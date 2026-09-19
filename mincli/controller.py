"""ChatController —— mincli 纯逻辑层（无任何 UI 依赖）。

管理多棵对话树、设置、流式输出、工具调用与持久化。
Textual TUI 通过本控制器驱动；UI 通过 ControllerEvent 回调接收增量更新
（流式内容 / 工具调用 / 状态 / 完成）。

多对话树：
- 每棵树一个文件（见 mincli/trees.py），容量固定为「节点树 + 该树自己的设置」；
- 模型/温度/思考模式/系统提示词等是全局设置，所有树共享；
- 审核层级、file_confirm、工作目录、能力白名单、输入草稿是每棵树独立的；
- 同一时刻只允许一棵树在生成：后台生成期间切走，生成继续，但另一棵树
  必须等它结束才能发送（见 ChatApp._send_user_text）。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from mincli.config import (
    MODEL_FLASH,
    VISION_MODELS,
    MODELS_AVAILABLE,
    COMPACT_MAX_TOKENS,
    COMPACT_SOURCE_MAX_CHARS,
    COMPACT_REASONING_MAX_CHARS,
    COMPACT_TOOL_RESULT_MAX_CHARS,
    EXEC_DEFAULT_TIMEOUT,
    VISION_DEFAULT_DETAIL,
    VISION_LOW_INLINE_BUDGET_BYTES,
    VISION_MAX_SIDE,
    VISION_MAX_SIDE_MANY,
    VISION_MANY_IMAGES_THRESHOLD,
    VISION_REQUEST_IMAGES_MAX_COUNT,
    VISION_REQUEST_INLINE_TOTAL_MAX_BYTES,
    VISION_REQUEST_MAX_BYTES,
    VISION_REQUEST_TOTAL_MAX_BYTES,
    FILES_LIST_PAGE,
    WORKFLOWS_PATH,
    WF_EXTRACT_MAX_TOKENS,
    WF_SOURCE_MAX_CHARS,
    WF_REASONING_MAX_CHARS,
    WF_TOOL_ARGS_MAX_CHARS,
    WF_TOOL_RESULT_MAX_CHARS,
    WF_ANSWER_MAX_CHARS,
    load_models,
    normalize_model_name,
)
from mincli.helpers import (
    convert_formulas,
    estimate_input_price,
    estimate_prompt_tokens,
    estimate_tokens,
    generate_conversation_title,
    get_balance,
    is_peak_hour,
    save_conversation_to_file,
)
from mincli.models import ConversationNode, ConversationTree
from mincli.streaming import stream_response
from mincli.tools.execute import audit_command, is_safe_readonly, matches_dangerous
from mincli.tools.file_ops import parse_file
from mincli.tools.files import (
    FilesAPIError,
    delete_file,
    list_all_files,
    list_files,
    retrieve_file,
    upload_image,
)
from mincli.tools.images import (
    ImageAttachment,
    collect_inline_bytes,
    estimate_image_tokens,
    image_placeholder_text,
    is_image_path,
    looks_like_image_target,
    make_path_attachment,
    make_url_attachment,
    oversize_inline_attachments,
    oversize_side_attachments,
    split_low_inline,
    total_tokens_est,
)
from mincli.tools.registry import TOOLS
from mincli.tools.web_fetch import fetch_webpage
from mincli.trees import TreeStore, color_of
from mincli.workflows import (
    FALLBACK_MARK,
    Workflow,
    WorkflowStore,
    doc_placeholders,
    now_iso,
    substitute,
    valid_name,
)

try:
    from mincli.mcp_client import BUNDLED_NAME, READY_TIMEOUT, McpToolClient
except ImportError:  # pragma: no cover
    McpToolClient = None
    BUNDLED_NAME = "mincli"
    READY_TIMEOUT = 30


@dataclass
class ControllerEvent:
    """控制器向 UI 发出的增量事件。

    kind:
        stream      —— 流式增量（content / reasoning 各为本次增量）
        tool        —— 工具调用（tool_name / tool_args / tool_summary；summary 为空表示开始）
        status      —— 状态信息（message）
        node_created —— 新节点已在树中创建并设为当前节点（流式输出归属该节点）
        done        —— 一条消息处理完成（node 为新节点）
        error       —— 出错（message）
    """

    kind: str
    content: str = ""
    reasoning: str = ""
    tool_name: str = ""
    tool_args: str = ""
    tool_summary: str = ""
    message: str = ""
    node: Optional[ConversationNode] = None
    # 事件所属的对话树编号（由 send_message 统一盖章）。后台生成期间切走时，
    # UI 据此丢弃不属于当前树的事件，避免流式内容串到别的树。
    tree: int = 0

    @classmethod
    def stream(cls, content: str, reasoning: str) -> "ControllerEvent":
        return cls(kind="stream", content=content, reasoning=reasoning)

    @classmethod
    def tool(cls, name: str, args: str, summary: str) -> "ControllerEvent":
        return cls(kind="tool", tool_name=name, tool_args=args, tool_summary=summary)

    @classmethod
    def status(cls, message: str) -> "ControllerEvent":
        return cls(kind="status", message=message)

    @classmethod
    def node_created(cls, node: ConversationNode) -> "ControllerEvent":
        return cls(kind="node_created", node=node)

    @classmethod
    def done(cls, node: ConversationNode) -> "ControllerEvent":
        return cls(kind="done", node=node)

    @classmethod
    def error(cls, message: str) -> "ControllerEvent":
        return cls(kind="error", message=message)


EventSink = Callable[[ControllerEvent], None]

AUDIT_LABELS = {
    1: "最高（AI审核 + 用户确认）",
    2: "中等（AI审核，低风险自动执行）",
    3: "最低（文本匹配，高风险询问）",
    4: "无（直接执行）",
}


@dataclass
class TreeState:
    """一棵对话树的运行时状态（编号 0 表示「尚未创建任何对话树」的占位）。"""

    number: int
    color: int
    tree: ConversationTree
    draft: str = ""
    # 每棵树独立的运行设置（其余设置是全局的）
    audit_level: int = 1
    file_confirm: bool = True
    workspace: Optional[str] = None
    # 能力挂载：系统工具整组开关 + 外置 MCP 工具白名单（工具名）
    system_tools: bool = True
    mcp_tools: List[str] = field(default_factory=list)

    @property
    def is_placeholder(self) -> bool:
        return self.number <= 0


class ChatController:
    """对话引擎：状态 + 多对话树 + 流式输出 + 工具调用 + 持久化。"""

    # 工作流持久化文件（测试子类可重写为临时路径）
    WORKFLOWS_FILE = WORKFLOWS_PATH

    def __init__(
        self,
        client: OpenAI,
        default_system: str,
        default_temperature: float,
        default_model: str = MODEL_FLASH,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        auto_start_mcp: bool = True,
        trees_dir: Optional[str] = None,
    ) -> None:
        self.client = client
        # ---------------- 全局设置（所有对话树共享） ----------------
        self.current_system = default_system
        self.current_temperature = default_temperature
        self.current_model = normalize_model_name(default_model)
        # 从磁盘加载时，若旧模型名被自动改写，这里记录原始名（供 UI 提示）
        self.model_migrated_from: Optional[str] = None
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort
        # 用户主动打断标志：streaming 循环在每个 chunk 处检查，工具调用后也会检查。
        # 每次 send_message 开始时复位；interrupt() 置位并顺带终止正在执行的命令。
        self._interrupt_requested: bool = False

        # 多模态：待发送图片（/import 导入的图片填充；发送后绑定到节点并清空）
        self.pending_images: List[ImageAttachment] = []
        # 图片 detail 全局默认（/set detail 可调；low 省 token，auto≈original 最清晰）
        self.image_detail: str = VISION_DEFAULT_DETAIL

        # ---------------- 多对话树 ----------------
        self._store = TreeStore(trees_dir)
        self._states: Dict[int, TreeState] = {}
        self._active: int = 0
        # 生成线程的树上下文：send_message 在 worker 线程里设置它，使 self.tree /
        # 每树设置在整个生成过程中都指向发起生成的那棵树（用户切走也不串台）
        self._tls = threading.local()
        # 正在生成的树编号（0 = 空闲）
        self._gen_tree: int = 0
        # 后台生成结束标记：编号 → "done" | "error"（UI 取用后清除）
        self.tree_marks: Dict[int, str] = {}

        self._wf_store = WorkflowStore(self.WORKFLOWS_FILE)

        self.imported_content: Optional[str] = None
        # /import 导入的文本/网页文件元数据（{"kind": "text"|"web", "name": str}），
        # 用于输入框下方状态栏展示；内容拼接在 imported_content 中随下次发送附带
        self.imported_files: List[Dict[str, str]] = []
        self.temp_dir = tempfile.mkdtemp(prefix="mincli_")
        self.temp_files: Dict[str, str] = {}

        # UI 注入的确认回调（写文件 / 执行命令等工具用）。默认拒绝（安全）。
        self.confirm: Callable[[str, str], bool] = lambda title, text: False

        self._mcp: Optional[McpToolClient] = None
        self._mcp_tool_names: set = set()
        # MCP 工具列表是否已同步（后台连接完成后同步一次；重连后置回 False）
        self._mcp_synced: bool = False
        # MCP 连接日志出口：默认打印（无 UI 时）；TUI 会换成通知，避免
        # 后台线程直接写 stdout 冲乱界面
        self.mcp_logger: Callable[[str], None] = print
        self.llm_tools: List[Dict] = list(TOOLS)

        self.session_loaded = self._restore_from_store()
        if auto_start_mcp:
            self.start_mcp()

    # ---------------- 对话树：加载 / 保存 ----------------

    def _restore_from_store(self) -> bool:
        """从磁盘恢复全局设置与上次激活的对话树。

        返回是否恢复了已有对话树（False = 一棵都没有，UI 应先引导新建）。
        """
        g = self._store.global_settings()
        if g:
            self.current_system = g.get("system_prompt", self.current_system)
            self.current_temperature = g.get("temperature", self.current_temperature)
            saved_model = g.get("model", self.current_model)
            self.current_model = normalize_model_name(saved_model)
            if self.current_model != saved_model:
                self.model_migrated_from = saved_model
            self.thinking_enabled = bool(g.get("thinking_enabled", False))
            self.reasoning_effort = g.get("reasoning_effort", "high")
            self.image_detail = g.get("image_detail", VISION_DEFAULT_DETAIL)
            self.imported_content = g.get("imported_content")
            self.imported_files = g.get("imported_files") or []

        numbers = self._store.numbers()
        active = self._store.active()
        if active is None and numbers:
            active = numbers[-1]
        if active is not None and self._store.has_tree(active):
            self._load_state(active)
            self._active = int(active)
            self._store.set_active(self._active)
            return True

        # 一棵树都没有：放一棵占位空树，让 UI 有东西可渲染（发送前必须先建树）
        self._reset_placeholder()
        return False

    def _reset_placeholder(self) -> None:
        self._states[0] = TreeState(
            number=0, color=1, tree=ConversationTree(self.current_system)
        )
        self._active = 0

    def _load_state(self, number: int) -> TreeState:
        """按需加载一棵树（已加载则直接返回）。"""
        number = int(number)
        st = self._states.get(number)
        if st is not None:
            return st
        data = self._store.load_tree(number) or {}
        tree_data = data.get("tree")
        tree = (
            ConversationTree.from_dict(tree_data)
            if isinstance(tree_data, dict)
            else ConversationTree(self.current_system)
        )
        # 系统提示词是全局设置：以内存中的值为准，忽略树文件里的旧副本
        tree.system_prompt = self.current_system
        st = TreeState(
            number=number,
            color=self._store.color_of(number),
            tree=tree,
            draft=data.get("draft", "") or "",
            audit_level=int(data.get("audit_level") or 1),
            file_confirm=bool(data.get("file_confirm", True)),
            workspace=data.get("workspace") or None,
            system_tools=bool(data.get("system_tools", True)),
            mcp_tools=list(data.get("mcp_tools") or []),
        )
        self._states[number] = st
        return st

    @staticmethod
    def _snapshot_tree(tree: ConversationTree) -> Optional[Dict[str, Any]]:
        """把树序列化成 dict；生成线程正在增删节点时退避重试，仍失败则放弃本次写入。

        后台生成与 UI 切树可能同时发生（切树会保存「离开的那棵树」），而
        `to_dict` 要遍历节点表——遍历中字典被改会抛 RuntimeError。宁可跳过
        这一次保存（下一次切换/退出还会再写），也不要写出半棵树。
        """
        for attempt in range(3):
            try:
                return tree.to_dict()
            except RuntimeError:
                time.sleep(0.05 * (attempt + 1))
        return None

    def _tree_payload(self, st: TreeState) -> Optional[Dict[str, Any]]:
        tree_data = self._snapshot_tree(st.tree)
        if tree_data is None:
            return None
        return {
            "number": st.number,
            "color": st.color,
            "tree": tree_data,
            "draft": st.draft,
            "audit_level": st.audit_level,
            "file_confirm": st.file_confirm,
            "workspace": st.workspace,
            "system_tools": st.system_tools,
            "mcp_tools": list(st.mcp_tools),
            "saved_at": time.time(),
        }

    def _global_payload(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.current_system,
            "temperature": self.current_temperature,
            "model": self.current_model,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "image_detail": self.image_detail,
            "imported_content": self.imported_content,
            "imported_files": self.imported_files,
        }

    def save_tree(self, number: Optional[int] = None) -> bool:
        """保存指定（默认当前）对话树到磁盘。占位树（0）不落盘。"""
        st = self._state(number)
        if st is None or st.is_placeholder:
            return False
        payload = self._tree_payload(st)
        if payload is None:
            return False
        return self._store.save_tree(st.number, payload)

    def save_session(self) -> bool:
        """保存全部对话树 + 全局设置（退出时调用；返回是否全部成功）。"""
        ok = True
        for number, st in list(self._states.items()):
            if st.is_placeholder:
                continue
            payload = self._tree_payload(st)
            if payload is None or not self._store.save_tree(number, payload):
                ok = False
        self._store.set_global_settings(self._global_payload())
        return ok

    # ---------------- 对话树：访问与切换 ----------------

    def _state(self, number: Optional[int] = None) -> Optional[TreeState]:
        """取某棵树的状态（默认当前树）。

        生成线程里默认取「发起生成的那棵树」，因此 send_message 及其调用的
        工具实现全部自动作用于正确的树，用户中途切走也不受影响。
        """
        if number is None:
            ctx = getattr(self._tls, "state", None)
            if ctx is not None:
                return ctx
        return self._states.get(self._active if number is None else int(number))

    @property
    def tree(self) -> ConversationTree:
        st = self._state()
        if st is None:
            self._load_state(self._active) if self._active > 0 else self._reset_placeholder()
            st = self._state()
        return st.tree

    @tree.setter
    def tree(self, value: ConversationTree) -> None:
        st = self._state()
        if st is None:
            self._states[self._active] = TreeState(
                number=self._active, color=color_of(self._active), tree=value
            )
        else:
            st.tree = value

    @property
    def active_number(self) -> int:
        """当前激活的对话树编号（0 = 尚未创建）。"""
        return self._active

    @property
    def generating_number(self) -> int:
        """正在生成回答的树编号（0 = 空闲）。"""
        return self._gen_tree

    @property
    def has_trees(self) -> bool:
        return self._store.has_trees()

    def tree_numbers(self) -> List[int]:
        return self._store.numbers()

    def tree_color(self, number: Optional[int] = None) -> int:
        n = self._active if number is None else int(number)
        if n <= 0:
            return 1
        return self._store.color_of(n)

    def tree_summary(self, number: int) -> dict:
        """列表展示用摘要（不把整棵树留在内存里）。"""
        number = int(number)
        st = self._states.get(number)
        tree = st.tree if st is not None else None
        if tree is None:
            data = self._store.load_tree(number) or {}
            tree_data = data.get("tree")
            tree = (
                ConversationTree.from_dict(tree_data)
                if isinstance(tree_data, dict)
                else None
            )
        title = ""
        count = 0
        if tree is not None:
            count = len(tree.nodes)
            if tree.root is not None:
                title = tree.root.title or ""
        return {"number": number, "nodes": count, "title": title}

    def switch_tree(self, number: int) -> bool:
        """切换当前对话树（编号不存在返回 False）。"""
        number = int(number)
        if not self._store.has_tree(number):
            return False
        if number == self._active:
            self.tree_marks.pop(number, None)
            return True
        self.save_tree()  # 离开前先保存当前树，避免崩溃丢改动
        self._load_state(number)
        self._active = number
        self._store.set_active(number)
        self._interrupt_requested = False  # 打断标志只属于具体那棵树
        self.tree_marks.pop(number, None)
        self._rebuild_llm_tools()
        return True

    def create_tree(
        self,
        system_tools: bool = True,
        mcp_tools: Optional[List[str]] = None,
        activate: bool = True,
    ) -> int:
        """新建一棵空对话树并（默认）切过去，返回编号。"""
        if activate:
            self.save_tree()
        number = self._store.allocate()
        st = TreeState(
            number=number,
            color=self._store.color_of(number),
            tree=ConversationTree(self.current_system),
            system_tools=bool(system_tools),
            mcp_tools=list(mcp_tools or []),
        )
        self._states[number] = st
        # 刚建的空树没有并发改动，快照一定成功
        self._store.save_tree(number, self._tree_payload(st) or {})
        if activate:
            self._states.pop(0, None)
            self._active = number
            self._store.set_active(number)
            self._rebuild_llm_tools()
        return number

    def delete_tree(self, number: int) -> dict:
        """删除整棵对话树（含数据文件）。编号不复用。"""
        number = int(number)
        if not self._store.has_tree(number):
            return {"ok": False, "reason": "missing"}
        was_active = number == self._active
        self._store.delete_tree(number)
        self._states.pop(number, None)
        self.tree_marks.pop(number, None)
        if was_active:
            remaining = self._store.numbers()
            if remaining:
                nxt = remaining[-1]
                self._load_state(nxt)
                self._active = nxt
                self._store.set_active(nxt)
            else:
                self._states.pop(0, None)
                self._reset_placeholder()
                self._store.set_active(None)
            self._rebuild_llm_tools()
        return {"ok": True, "was_active": was_active, "remaining": self._store.numbers()}

    def tree_caps(self, number: Optional[int] = None) -> dict:
        """某棵树的能力挂载配置（未加载的树按需读盘）。"""
        st = self._state(number)
        if st is None and number is not None and self._store.has_tree(int(number)):
            st = self._load_state(int(number))
        if st is None:
            return {"system_tools": True, "mcp_tools": []}
        return {"system_tools": st.system_tools, "mcp_tools": list(st.mcp_tools)}

    def set_tree_caps(
        self,
        number: Optional[int] = None,
        system_tools: Optional[bool] = None,
        mcp_tools: Optional[List[str]] = None,
    ) -> bool:
        """修改某棵树的能力挂载（对下一次请求生效）。"""
        st = self._state(number)
        if st is None and number is not None and self._store.has_tree(int(number)):
            st = self._load_state(int(number))
        if st is None or st.is_placeholder:
            return False
        if system_tools is not None:
            st.system_tools = bool(system_tools)
        if mcp_tools is not None:
            st.mcp_tools = list(mcp_tools)
        self.save_tree(st.number)
        if st.number == self._active:
            self._rebuild_llm_tools()
        return True

    def available_tool_groups(self) -> Dict[str, List[str]]:
        """当前可勾选的工具：server 名 → 工具名（内置 server 的键为 "mincli"）。"""
        if self._mcp is None:
            return {}
        return self._mcp.tools_by_server()

    def external_servers(self) -> List[str]:
        """已配置的第三方 MCP server 名（可能尚未连接完成）。"""
        if self._mcp is None:
            return []
        return self._mcp.configured_servers()

    # ---------------- 每棵树独立的设置 ----------------

    @property
    def audit_level(self) -> int:
        st = self._state()
        return st.audit_level if st is not None else 1

    @audit_level.setter
    def audit_level(self, value: int) -> None:
        st = self._state()
        if st is not None:
            st.audit_level = int(value)

    @property
    def file_confirm(self) -> bool:
        st = self._state()
        return st.file_confirm if st is not None else True

    @file_confirm.setter
    def file_confirm(self, value: bool) -> None:
        st = self._state()
        if st is not None:
            st.file_confirm = bool(value)

    @property
    def workspace(self) -> Optional[str]:
        st = self._state()
        return st.workspace if st is not None else None

    @workspace.setter
    def workspace(self, value: Optional[str]) -> None:
        st = self._state()
        if st is not None:
            st.workspace = value

    @property
    def draft(self) -> str:
        st = self._state()
        return st.draft if st is not None else ""

    @draft.setter
    def draft(self, value: str) -> None:
        st = self._state()
        if st is not None:
            st.draft = value or ""

    # ---------------- MCP ----------------

    def _mcp_log(self, message: str) -> None:
        """MCP 客户端日志回调（可能来自后台连接线程）。"""
        try:
            self.mcp_logger(message)
        except Exception:
            pass

    @property
    def mcp_started(self) -> bool:
        """MCP 客户端是否已创建（不代表已连接完成）。"""
        return self._mcp is not None

    @property
    def mcp_ready(self) -> bool:
        """后台连接是否已结束。"""
        return self._mcp is not None and self._mcp.ready

    @property
    def mcp_connecting(self) -> bool:
        """是否仍在后台连接（UI 据此显示「连接中」）。"""
        return self._mcp is not None and self._mcp.connecting

    @property
    def mcp_connected(self) -> bool:
        """是否至少连上一个 server（工具可用）。"""
        return self._mcp is not None and self._mcp.ok

    def start_mcp(self) -> None:
        """创建 MCP 客户端并**在后台**开始连接（立即返回，不阻塞首屏）。"""
        if McpToolClient is None or self._mcp is not None:
            return
        self._mcp = McpToolClient(log=self._mcp_log)
        self._mcp_synced = False
        try:
            self._mcp.start()
        except Exception:
            self._mcp = None
            self._mcp_tool_names = set()
        self._rebuild_llm_tools()

    def wait_mcp_ready(
        self, emit: Optional[EventSink] = None, timeout: Optional[float] = READY_TIMEOUT
    ) -> bool:
        """等待后台 MCP 连接结束并同步工具列表（发送消息前调用）。

        首次调用会阻塞到连接完成（通常启动后 1~2 秒内就已结束，此处瞬间返回）；
        之后只做一次同步。超时或连不上时返回 False，调用方退回“无 MCP 工具”
        继续，不因为 MCP 故障把发送卡死。
        """
        if self._mcp is None:
            return False
        if not self._mcp_synced:
            if self._mcp.connecting and emit is not None:
                emit(ControllerEvent.status("正在连接 MCP 服务…"))
            self._mcp.wait_ready(timeout=timeout)
            self._mcp_tool_names = self._mcp.tool_names()
            self._mcp_synced = True
            self._rebuild_llm_tools()
        return self._mcp.ok

    def _rebuild_llm_tools(self) -> None:
        """按当前对话树的能力挂载重组发给模型的工具列表。

        - 进程内工具（查询对话树）与内置 server 的工具同属「系统工具」整组开关；
        - 外置 server 的工具按该树保存的白名单逐个放行（白名单为空 = 一个都不挂）；
        - MCP 未就绪时只有进程内工具，其余等 wait_mcp_ready 同步后补上。
        """
        st = self._state()
        system_tools = True if st is None else st.system_tools
        allowed = set(st.mcp_tools) if st is not None else set()
        tools: List[Dict] = []
        if system_tools:
            tools.extend(TOOLS)
        if self._mcp is not None:
            for d in self._mcp.tools():
                name = d.get("function", {}).get("name")
                if not name:
                    continue
                if self._mcp.tool_owner(name) == BUNDLED_NAME:
                    if system_tools:
                        tools.append(d)
                elif name in allowed:
                    tools.append(d)
        self.llm_tools = tools

    def mcp_status(self) -> dict:
        return self._mcp.server_status() if self._mcp else {}

    def mcp_reload(self) -> None:
        """后台重连全部 MCP server（读取最新配置），立即返回。"""
        if not self._mcp:
            raise RuntimeError("MCP 客户端未就绪")
        self._mcp.reload()
        self._mcp_synced = False

    def close(self) -> None:
        if self._mcp:
            self._mcp.close()
            self._mcp = None

    # ---------------- 打断 ----------------

    def interrupt(self) -> bool:
        """请求打断当前轮生成/命令执行（线程安全，可从 UI 线程调用）。

        - 置位打断标志：流式循环在下一个 chunk 处停止读取（见 stream_response
          的 should_stop），工具执行后也会检查并结束本轮；
        - 立即请求内置 MCP server 终止正在运行的命令进程组，长任务不用再等
          超时（第三方 server 无该内部工具，只会停止后续流程）。

        返回是否发起了打断请求（当前是否处于可打断状态由 UI 侧感知）。
        """
        self._interrupt_requested = True
        if self._mcp is not None:
            try:
                self._mcp.cancel_running()
            except Exception:
                pass
        return True

    @property
    def interrupt_pending(self) -> bool:
        """本轮是否已请求过打断（UI 用于「再按一次强制退出」）。"""
        return self._interrupt_requested

    # ---------------- 设置 ----------------

    def set_system(self, system: str) -> None:
        """设置系统提示词（全局设置：立即作用于所有已加载的对话树）。"""
        self.current_system = system
        for st in self._states.values():
            st.tree.system_prompt = system

    def set_temperature(self, temp: float) -> None:
        self.current_temperature = temp

    def set_model(self, model: str) -> bool:
        """切换模型：支持 flash/pro/vision 简写与现役/已注册模型名。

        旧模型名（deepseek-v4-flash、deepseek-v4-flash-vision-exp 等）自动
        改写为现役名（vision → deepseek-flash，因为图片能力已并入 Flash）。
        """
        arg = (model or "").strip()
        normalized = normalize_model_name(arg)
        registered = load_models()
        if normalized in MODELS_AVAILABLE or normalized in registered:
            self.current_model = normalized
            return True
        # 未注册的自定义完整模型名（如 gpt-4o）原样接受
        if arg in registered:
            self.current_model = arg
            return True
        return False

    def set_detail(self, detail: str) -> bool:
        """设置图片 detail 全局默认（low/auto/high/original）。

        同时改写「待发送图片」的 detail 并重算 token 估算：用户多半是先导入图片
        再 `/set detail low`，若只改全局默认，已入队的图片仍会按旧档发送。
        """
        if detail not in ("low", "auto", "high", "original"):
            return False
        self.image_detail = detail
        for att in self.pending_images:
            att.detail = detail
            att.tokens_est = estimate_image_tokens(att.width, att.height, detail)
        return True

    def set_thinking(self, on: bool) -> None:
        self.thinking_enabled = on

    def set_effort(self, effort: str) -> bool:
        if effort in ("low", "high", "max"):
            self.reasoning_effort = effort
            return True
        return False

    def set_audit(self, level: int) -> bool:
        if level in (1, 2, 3, 4):
            self.audit_level = level
            return True
        return False

    def set_file_confirm(self, enabled: bool) -> None:
        """设置写文件/编辑文件时是否弹窗确认。"""
        self.file_confirm = bool(enabled)

    def set_workspace(self, path: str) -> bool:
        """设置命令执行默认工作目录（不存在则创建）。"""
        path = os.path.expanduser(path.strip())
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            return False
        self.workspace = os.path.abspath(path)
        return True

    # ---------------- 导入 / 导出 ----------------

    # 工具函数返回的错误文案前缀（视为导入失败而非内容）
    _IMPORT_FAIL_PREFIXES = (
        "文件不存在:",
        "无法获取网页内容:",
        "无法从网页中提取有效文本:",
        "抓取或解析失败:",
        "无法读取:",
    )

    def import_target(self, target: str) -> Optional[str]:
        """导入单个文件/网页为上下文（兼容入口）。

        图片路径/URL 转为待发送图片附件。成功返回 None，失败返回错误信息。
        """
        res = self.import_targets([target])
        if res["errors"] and not res["images_added"] and not res["text_added"]:
            return res["errors"][0]
        return None

    def import_targets(self, targets: List[str]) -> dict:
        """批量导入（/import 多文件）：图片→待发送图片，文本/网页→下次发送附带。

        返回 {"images_added": int, "text_added": int, "errors": [str, ...]}。
        """
        images_before = len(self.pending_images)
        text_before = len(self.imported_files)
        errors: List[str] = []
        for t in targets:
            err = self._import_one(t)
            if err:
                errors.append(err)
        return {
            "images_added": len(self.pending_images) - images_before,
            "text_added": len(self.imported_files) - text_before,
            "errors": errors,
        }

    def _import_one(self, target: str) -> Optional[str]:
        """导入单个目标。成功返回 None，失败返回错误信息。"""
        target = target.strip()
        if re.match(r"^https?://", target):
            # 网页 URL：图片扩展名 → 图片附件；否则抓取网页文本
            if looks_like_image_target(target):
                added, errors = self.add_pending_images([target])
                if added:
                    return None
                return errors[0] if errors else f"无法读取: {target}"
            result = fetch_webpage(target)
            if result and not result.startswith(self._IMPORT_FAIL_PREFIXES):
                self._append_imported(result, {"kind": "web", "name": target[:120]})
                return None
            return result or f"无法读取: {target}"
        # 本地文件：先展开 ~（否则 ~/图片 会漏过图片嗅探走进文本解析）
        path = os.path.expanduser(target)
        if os.path.isfile(path) and is_image_path(target):
            added, errors = self.add_pending_images([target])
            if added:
                return None
            return errors[0] if errors else f"无法读取: {target}"
        result = parse_file(path)
        if result and not result.startswith(self._IMPORT_FAIL_PREFIXES):
            self._append_imported(result, {"kind": "text", "name": os.path.basename(path)})
            return None
        return result or f"无法读取: {target}"

    def _append_imported(self, content: str, meta: Dict[str, str]) -> None:
        """把一段导入内容拼接到 imported_content，并记录文件元数据。"""
        if self.imported_content:
            self.imported_content += "\n\n---\n\n"
        self.imported_content = (self.imported_content or "") + content
        self.imported_files.append(meta)

    def clear_imports(self) -> int:
        """清空待导入内容（图片 + 文本/网页），返回清除的文件数。"""
        n = len(self.pending_images) + len(self.imported_files)
        self.pending_images = []
        self.imported_content = None
        self.imported_files = []
        return n

    def import_summary(self) -> str:
        """状态条中段文本：已导入文件数量 + 前 2 个文件名 + 图片 token 估算。"""
        names = [a.name for a in self.pending_images] + [
            f["name"] for f in self.imported_files
        ]
        if not names:
            return ""
        shown = "、".join(names[:2])
        extra = "…" if len(names) > 2 else ""
        return (
            f"已导入 {len(names)} 个文件：{shown}{extra}{self.images_tokens_hint()}"
            " · /import clear 清除"
        )

    def images_tokens_hint(self) -> str:
        """待发送图片的 token 估算提示（无图片返回空串）。

        按官方图片预处理公式逐张估算，让用户在发送前就知道这轮图片大概多少
        token（只算图片本身，文本与工具定义不计）。
        """
        if not self.pending_images:
            return ""
        tokens = total_tokens_est(self.pending_images)
        if not tokens:
            return ""
        return f" · 图片约 {tokens} tokens"

    def import_file_list(self) -> List[Dict[str, str]]:
        """完整导入文件列表（图片在前、文本/网页在后），供悬停弹窗展示。"""
        items = [
            {"kind": "image", "name": a.name, "tokens": a.tokens_est or 0}
            for a in self.pending_images
        ]
        items += [
            {"kind": f["kind"], "name": f["name"], "tokens": 0}
            for f in self.imported_files
        ]
        return items

    # ---------------- 多模态：待发送图片 ----------------

    def add_pending_images(self, targets: List[str]) -> tuple:
        """把路径/URL 添加为待发送图片。返回 (成功数, 错误信息列表)。"""
        added = 0
        errors: List[str] = []
        for t in targets:
            try:
                if t.startswith(("http://", "https://")):
                    att = make_url_attachment(t, self.image_detail)
                else:
                    att = make_path_attachment(t, self.image_detail)
                self.pending_images.append(att)
                added += 1
            except ValueError as e:
                errors.append(str(e))
        return added, errors

    def save_node(self, node_id: str) -> Optional[str]:
        """导出节点为 Markdown 文件，返回文件路径；节点不存在返回 None。"""
        node = self.tree.nodes.get(node_id)
        if not node:
            return None
        user_msg = convert_formulas(node.user_msg)
        assistant_msg = convert_formulas(node.assistant_msg)
        reasoning = convert_formulas(node.reasoning)
        content = f"# {node.title}\n\n"
        content += f"---\n\n**你：**\n\n{user_msg}\n\n"
        if node.user_images:
            marks = "；".join(image_placeholder_text(att) for att in node.user_images)
            content += f"（附图：{marks}）\n\n"
        if node.reasoning:
            content += f"---\n\n**DeepSeek 思考过程：**\n\n{reasoning}\n\n"
        content += f"---\n\n**DeepSeek：**\n\n{assistant_msg}\n\n"
        token_stats = {
            "input_tokens": node.input_tokens,
            "output_tokens": node.output_tokens,
        }
        return save_conversation_to_file(content, node.title, node.id, token_stats)

    def get_node_markdown_file(self, node_id: str) -> Optional[str]:
        """把节点回答写入临时文件（/view 用），返回路径；无内容返回 None。"""
        node = self.tree.nodes.get(node_id)
        if not node or not node.assistant_msg:
            return None
        if node_id in self.temp_files:
            return self.temp_files[node_id]
        filepath = os.path.join(self.temp_dir, f"mincli_{node_id}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(node.assistant_msg)
        self.temp_files[node_id] = filepath
        return filepath

    def reset(self) -> None:
        """清空当前对话树的对话历史（/clear）。"""
        self._cleanup_temp_files()
        self.tree = ConversationTree(self.current_system)
        self.save_tree()

    # ---------------- 上下文压缩 ----------------

    _COMPACT_SYSTEM_INSTR = (
        "你负责把对话历史压缩成一份详尽、信息密度高的上下文摘要。"
        "这份摘要将替代原文作为后续对话的背景，因此必须尽可能完整地保留信息，宁可长、不要短。"
    )

    _COMPACT_PROMPT = """请把下面的对话历史压缩成一份【详尽】的上下文摘要，要求：

1. 按主题组织，使用 Markdown 标题和列表，条理清晰，便于后续检索。
2. 必须完整保留（不得省略）：
   - 用户的总体目标、具体需求与每轮提出的问题
   - 背景事实与关键约束
   - 已做出的决定与结论
   - 执行过的命令、涉及的代码/文件路径、关键代码片段要点
   - 重要的数据、数字、编号、专有名词与 ID
   - 用户的偏好与风格要求
   - 尚未解决的问题、待办事项与当前进度
3. 保留对话所使用的语言（中文对话用中文输出）。
4. 目标长度约为原文的 1/3 到 1/2，不要刻意精简；信息密度优先，重复内容可合并。
5. 只输出摘要本身，不要任何解释性开场白。

对话历史：
{source}"""

    def compact_history(self, emit: Optional[EventSink] = None) -> Optional[dict]:
        """把当前分支全部对话压缩成详细摘要，并新建摘要节点（/compact）。

        压缩后新建一个子节点（用户消息 = 摘要）并设为当前节点；只有摘要节点
        及其子节点发送消息时用摘要替代全部历史，其他节点仍发送完整原始消息
        （不再保留任何原文轮次）。返回统计信息 dict：
            {"blocked": "already_compact"} —— 当前节点已是摘要节点，禁止重复压缩
            None —— 无可压缩内容 / 压缩失败
            其他 —— 成功（含 node_id / before_tokens / after_tokens 等）

        统计口径：before/after 均用 estimate_tokens 估算压缩前后「发送给模型
        的完整消息列表」（与 usage_stats 摘要节点口径同函数、同来源）。
        """
        if self.tree is None or self.tree.current_node is None:
            return None
        comp = self.tree.compaction
        if comp and comp.get("boundary_id") == self.tree.current_node.id:
            return {"blocked": "already_compact"}
        path = self._path_to_root(self.tree.current_node)
        if not any(n.user_msg or n.assistant_msg for n in path):
            return None  # 没有任何可压缩的内容
        to_compress = path  # 全部压缩，不保留任何原文

        source = self._build_compact_source(to_compress)
        if len(source) > COMPACT_SOURCE_MAX_CHARS:
            half = COMPACT_SOURCE_MAX_CHARS // 2
            source = (
                source[:half]
                + f"\n\n…（原文过长，中间 {len(source) - COMPACT_SOURCE_MAX_CHARS} 字符已省略）…\n\n"
                + source[-half:]
            )

        if emit:
            emit(ControllerEvent.status(
                f"正在压缩上下文：{len(to_compress)} 轮 → 详细摘要…"
            ))

        summary = self._call_summarize(source)
        if not summary:
            return None

        before_msgs = self.tree.get_messages_for_node(self.tree.current_node)
        # 新建摘要节点（用户消息 = 摘要）并设为当前节点
        node = self.tree.add_child(
            self.tree.current_node, summary, "", "", "上下文压缩摘要", 0, 0
        )
        self.tree.compaction = {"summary": summary, "boundary_id": node.id}
        self.tree.current_node = node
        after_msgs = self.tree.get_messages_for_node(node)
        before_tok = estimate_tokens(before_msgs)
        after_tok = estimate_tokens(after_msgs)

        return {
            "summary": summary,
            "node_id": node.id,
            "boundary_id": node.id,
            "nodes_compressed": len(to_compress),
            "summary_chars": len(summary),
            "before_tokens": before_tok,
            "after_tokens": after_tok,
            "saved_tokens": max(0, before_tok - after_tok),
        }

    # ---------------- 实时用量统计（输入栏状态条） ----------------

    def usage_stats(self) -> dict:
        """输入栏状态条数据（纯本地计算，不联网）。

        缓存命中率取当前节点累计的 usage.prompt_cache_hit/miss_tokens。
        「下一次输入」优先用真实 usage 推算：普通节点 = 本节点**最后一次请求**
        的 prompt_tokens + 该轮 output_tokens（工具定义已计在 prompt_tokens 里，
        回答与思考会在下一次请求里作为历史回传并同样计费），因此与对话结束
        显示的「输入/输出」严格同口径（实测误差 ≤5 token）。只有拿不到 usage
        的节点（/compact 新建的摘要节点、旧存档、请求还没跑完）才回退到本地
        估算 estimate_prompt_tokens（= tiktoken 估算 + 工具定义开销，中文会
        高估，仅供参考）。用户新输入内容量小，忽略不计。
        预计价格按 DeepSeek 峰谷分时定价 × 缓存命中率折算。
        """
        stats: dict = {
            "cache_hit_rate": None,
            "next_input_tokens": 0,
            "estimated_price": None,
            "peak": is_peak_hour(),
            "model": self.current_model,
        }
        node = self.tree.current_node if self.tree else None
        if node is None:
            return stats
        hit = node.cache_hit_tokens
        miss = node.cache_miss_tokens
        total = hit + miss
        if total > 0:
            stats["cache_hit_rate"] = hit / total
        if node.last_prompt_tokens > 0:
            # 真实 usage 口径：再发一次就是「上次的 prompt + 本轮回答（含回传的思考）」
            next_in = node.last_prompt_tokens + node.last_output_tokens
        else:
            try:
                next_in = estimate_prompt_tokens(
                    self.tree.get_messages_for_node(node), self.llm_tools
                )
            except Exception:
                next_in = 0
        stats["next_input_tokens"] = next_in
        stats["estimated_price"] = estimate_input_price(
            self.current_model, next_in, stats["cache_hit_rate"], stats["peak"]
        )
        return stats

    def fetch_balance(self) -> Optional[dict]:
        """拉取 DeepSeek 账户余额（网络请求，调用方应放入后台线程）。

        返回 balance_infos 中的一项（优先 CNY）；失败返回 None。
        """
        infos = get_balance(self.client)
        if not infos:
            return None
        for info in infos:
            if info.get("currency") == "CNY":
                return info
        return infos[0]

    def _path_to_root(self, node: ConversationNode) -> List[ConversationNode]:
        path: List[ConversationNode] = []
        cur: Optional[ConversationNode] = node
        while cur is not None:
            path.append(cur)
            cur = self.tree.nodes.get(cur.parent_id) if cur.parent_id else None
        path.reverse()
        return path

    def _build_compact_source(self, nodes: List[ConversationNode]) -> str:
        parts = []
        for i, node in enumerate(nodes, start=1):
            title = (node.title or "").strip()
            head = f"--- 第 {i} 轮（节点 {node.id}）{('：' + title) if title else ''} ---"
            parts.append(head)
            user_line = node.user_msg
            if node.user_images:
                # 图片不进压缩请求（避免 base64 撑爆）；以占位符说明图片存在
                img_marks = "；".join(
                    image_placeholder_text(att) for att in node.user_images
                )
                user_line = f"{user_line}\n（附图：{img_marks}）"
            parts.append(f"用户: {user_line}")
            for tm in node.tool_messages:
                if tm.get("role") == "tool":
                    content = str(tm.get("content", ""))[:COMPACT_TOOL_RESULT_MAX_CHARS]
                    parts.append(f"工具结果: {content}")
            if node.reasoning:
                parts.append(f"思考过程: {node.reasoning[:COMPACT_REASONING_MAX_CHARS]}")
            if node.assistant_msg:
                parts.append(f"回答: {node.assistant_msg}")
        return "\n\n".join(parts)

    def _call_summarize(self, source: str) -> str:
        """调用模型生成详尽摘要；成功返回摘要文本，失败返回空串。"""
        messages = [
            {"role": "system", "content": self._COMPACT_SYSTEM_INSTR},
            {"role": "user", "content": self._COMPACT_PROMPT.format(source=source)},
        ]
        for max_tokens in (COMPACT_MAX_TOKENS, 4096):
            try:
                resp = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception:
                continue
        return ""

    def _cleanup_temp_files(self, keep_ids: Optional[set] = None) -> None:
        for nid, filepath in list(self.temp_files.items()):
            if keep_ids is None or nid not in keep_ids:
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                del self.temp_files[nid]

    # ---------------- 工作流（/wf） ----------------

    _WF_EXTRACT_INSTR = (
        "你是操作流程提炼助手。把一段“已完成任务的完整记录”提炼成可复用的"
        "工作流规范文档。规范用于以后每次执行同类任务时指导模型照做，因此必须"
        "保留工作内容、执行步骤、每一步调用命令/工具的细节，同时剔除只会出现"
        "在本次执行中的具体实例数据。"
    )

    _WF_EXTRACT_PROMPT = """请把下面的操作记录提炼成工作流规范文档，要求：

1. 严格按此格式输出（只输出文档本身，禁止任何开场白或解释）：
   目标：<一句话：这类任务要完成什么>
   （空行）
   变量：
   - {{变量名}}：含义（示例：一个典型值）
   （空行）
   步骤：
   1. <本步目的与动作>；若需命令，写明命令或工具与用法，如：`git log {{旧版本}}..HEAD`
   2. …

2. 步骤必须保留“命令/工具调用的细节”：工具名、关键参数/命令本身、该步要达成的
   目的、结果如何处理，让没有上下文的人也能照做。

3. 【重要】凡是每次执行都会变化的具体数据——版本号、日期、绝对路径、文件名、
   本次的正文/数据内容、ID、URL 等——一律不得写死，改写成 {{变量名}} 并在
   “变量”节说明含义与示例。例如不得写“v1.2.0”或“/Users/me/run.sh”，应写成
   {{旧版本}}、{{脚本路径}}。

4. 不包含思考过程、失败重试的花絮、只与本次记录相关的闲聊。

5. 保留原记录语言（中文记录用中文输出）。

操作记录：
{source}"""

    _WF_REVISE_INSTR = (
        "你是工作流文档修订助手。根据用户的修改要求，把给定的工作流规范文档改写为"
        "完整的新版本文档（输出整篇文档，不是 diff）。沿用原文档的格式与语言，"
        "沿用其中 {{变量}} 的既有约定。"
    )

    _WF_REVISE_PROMPT = """以下是当前的工作流规范文档：

{doc}

用户修改要求：
{request}

请输出完整的新版规范文档（格式与原文一致：目标 / 变量 / 步骤）。"""

    def wf_list(self) -> List[dict]:
        """返回工作流摘要列表（用于 /wf list）。"""
        out = []
        for wf in self._wf_store.list():
            goal, steps = (wf.goal(), len(
                [ln for ln in wf.doc.splitlines()
                 if re.match(r"^\s*\d+[.、]", ln)]))
            out.append({
                "name": wf.name,
                "goal": goal,
                "steps": steps,
                "vars": wf.placeholders(),
                "run_count": wf.run_count,
                "updated_at": wf.updated_at,
            })
        return out

    def wf_get(self, name: str) -> Optional[Workflow]:
        return self._wf_store.get(name)

    def _wf_build_source(self, nodes: List[ConversationNode]) -> str:
        parts = []
        for i, node in enumerate(nodes, start=1):
            title = (node.title or "").strip()
            head = f"--- 第 {i} 轮（节点 {node.id}）{('：' + title) if title else ''} ---"
            parts.append(head)
            user_line = node.user_msg or ""
            if node.user_images:
                img_marks = "；".join(
                    image_placeholder_text(att) for att in node.user_images
                )
                user_line = f"{user_line}\n（附图：{img_marks}）"
            parts.append(f"用户: {user_line}")
            for tm in node.tool_messages:
                role = tm.get("role")
                if role == "assistant":
                    for tc in tm.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        args = str(fn.get("arguments", ""))
                        if len(args) > WF_TOOL_ARGS_MAX_CHARS:
                            args = args[:WF_TOOL_ARGS_MAX_CHARS] + "…（已截断）"
                        parts.append(f"调用工具: {fn.get('name', '?')} {args}")
                elif role == "tool":
                    content = str(tm.get("content", ""))
                    if len(content) > WF_TOOL_RESULT_MAX_CHARS:
                        content = content[:WF_TOOL_RESULT_MAX_CHARS] + "…（已截断）"
                    parts.append(f"工具结果: {content}")
            if node.reasoning:
                parts.append(
                    f"思考: {node.reasoning[:WF_REASONING_MAX_CHARS]}"
                )
            if node.assistant_msg:
                ans = node.assistant_msg
                if len(ans) > WF_ANSWER_MAX_CHARS:
                    ans = ans[:WF_ANSWER_MAX_CHARS] + "…（已截断）"
                parts.append(f"回答: {ans}")
        return "\n\n".join(parts)

    def _wf_call_model(self, messages: List[Dict], max_tokens: int) -> str:
        """调用当前模型生成工作流文档（提炼/修订）；失败返回空串。"""
        for tokens in (max_tokens, 2000):
            try:
                resp = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception:
                continue
        return ""

    def wf_save(self, name: str, start_id: Optional[str] = None,
                force: bool = False) -> dict:
        """把当前节点（或 start_id→当前节点的连续链）提炼为工作流并保存。

        返回：
            {"status": "exists"} —— 同名已存在且未 force（UI 确认后带 force 重试）
            {"status": "error", "message": ...}
            {"status": "saved", "from": "extract"|"fallback", "name": ...,
             "nodes": N, "doc": ..., "placeholders": [...]}
        """
        if not valid_name(name):
            return {"status": "error",
                    "message": "工作流名需为 1-32 位字母/数字/_/-（如 release、git-changelog）"}
        if self._wf_store.has(name) and not force:
            return {"status": "exists"}

        if self.tree is None or self.tree.current_node is None:
            return {"status": "error", "message": "当前没有可提炼的对话"}
        current = self.tree.current_node

        if start_id:
            path = self._path_to_root(current)
            path_ids = [n.id for n in path]
            if start_id not in path_ids:
                return {"status": "error",
                        "message": f"起点节点 {start_id} 不在当前节点链上（应为当前节点或其祖先）"}
            nodes = path[path_ids.index(start_id):]
        else:
            nodes = [current]
        if not any(n.user_msg or n.assistant_msg or n.tool_messages for n in nodes):
            return {"status": "error", "message": "所选对话没有可提炼的内容"}

        source = self._wf_build_source(nodes)
        if len(source) > WF_SOURCE_MAX_CHARS:
            half = WF_SOURCE_MAX_CHARS // 2
            source = (
                source[:half]
                + f"\n\n…（原文过长，中间 {len(source) - WF_SOURCE_MAX_CHARS} 字符已省略）…\n\n"
                + source[-half:]
            )

        messages = [
            {"role": "system", "content": self._WF_EXTRACT_INSTR},
            {"role": "user",
             "content": self._WF_EXTRACT_PROMPT.format(source=source)},
        ]
        doc = self._wf_call_model(messages, WF_EXTRACT_MAX_TOKENS)
        from_status = "extract"
        if not doc:
            doc = (
                f"{FALLBACK_MARK}\n\n目标：未能自动提炼\n\n"
                f"原始操作记录：\n\n{source}"
            )
            from_status = "fallback"

        wf = Workflow(
            name=name,
            doc=doc,
            source_nodes=[n.id for n in nodes],
        )
        if not self._wf_store.put(wf, overwrite=True):
            return {"status": "error", "message": f"工作流「{name}」保存失败（文件不可写）"}
        return {
            "status": "saved",
            "from": from_status,
            "name": name,
            "nodes": len(nodes),
            "doc": doc,
            "placeholders": doc_placeholders(doc),
        }

    def wf_delete(self, name: str) -> bool:
        return self._wf_store.delete(name)

    def wf_rename(self, old: str, new: str) -> Optional[str]:
        """重命名；成功返回 None，失败返回错误信息。"""
        return self._wf_store.rename(old, new)

    def wf_revise(self, name: str, request: str) -> dict:
        """按用户要求让模型修订工作流文档并保存。"""
        wf = self._wf_store.get(name)
        if wf is None:
            return {"status": "error", "message": f"工作流「{name}」不存在"}
        messages = [
            {"role": "system", "content": self._WF_REVISE_INSTR},
            {"role": "user",
             "content": self._WF_REVISE_PROMPT.format(
                 doc=wf.doc, request=request)},
        ]
        new_doc = self._wf_call_model(messages, WF_EXTRACT_MAX_TOKENS)
        if not new_doc:
            return {"status": "error",
                    "message": "修订失败（模型未返回内容），工作流未改变"}
        wf.doc = new_doc
        self._wf_store.put(wf, overwrite=True)
        return {"status": "revised", "name": name, "doc": new_doc}

    def wf_export_temp(self, name: str) -> Optional[str]:
        """把工作流文档写入临时 .md 文件（供系统编辑器打开）；不存在返回 None。"""
        wf = self._wf_store.get(name)
        if wf is None:
            return None
        from mincli.workflows import write_doc_tempfile

        return write_doc_tempfile(wf.doc)

    def wf_import_text(self, name: str, text: str) -> bool:
        """编辑器改回后更新工作流文档。"""
        wf = self._wf_store.get(name)
        if wf is None:
            return False
        wf.doc = text.strip() or wf.doc
        return self._wf_store.put(wf, overwrite=True)

    def wf_compose(self, name: str, typed: str = "",
                   values: Optional[Dict[str, str]] = None) -> Optional[str]:
        """为“下一次输入/立即运行”构造执行消息文本；不存在返回 None。

        附带工作流规范全文（{占位符} 表示每次会变化的内容）。统计 run_count。
        """
        wf = self._wf_store.get(name)
        if wf is None:
            return None
        doc, missing = substitute(wf.doc, values or {})
        wf.run_count += 1
        wf.last_run_at = now_iso()
        self._wf_store.put(wf, overwrite=True)

        extras = ""
        if missing:
            extras = "（未提供值的变量：" + "、".join(
                "{" + m + "}" for m in missing) + "，请结合当前情况合理推断）"
        head = (
            f"请执行工作流「{wf.name}」{extras}。\n\n"
            "下面是该工作流的完整规范，其中 {变量} 表示每次会变化的内容，"
            "请结合本次输入与当前环境确定取值后严格按步骤执行，最后汇报结果：\n\n"
            + doc
        )
        if typed and typed.strip():
            return f"{head}\n\n本次输入：{typed}"
        return head

    # ---------------- 发送消息 ----------------

    def send_message(
        self, user_input: str, emit: EventSink
    ) -> Optional[ConversationNode]:
        """在当前对话树里发送一条消息（可能触发多轮工具调用），返回新节点。

        多对话树：本方法在 worker 线程里执行，进入时把「当前树」绑定到发起
        生成的那棵树（线程局部），因此用户中途切到别的树也不会让生成写错树；
        事件统一盖上树编号，UI 据此丢弃不属于当前树的事件。切走后生成继续，
        结束后在 tree_marks 里留下「完成/出错」标记供 UI 提示。
        """
        state = self._state()
        if state is None:
            self._reset_placeholder()
            state = self._state()
        if state is None or state.is_placeholder:
            raise RuntimeError("尚未创建对话树，请先新建对话树（/tree new）")

        self._tls.state = state
        self._gen_tree = state.number
        failed = False

        def _emit(ev: ControllerEvent) -> None:
            ev.tree = state.number
            emit(ev)

        try:
            return self._send_message_impl(user_input, _emit)
        except BaseException:
            failed = True
            raise
        finally:
            self._tls.state = None
            self._gen_tree = 0
            if state.number != self._active:
                # 后台生成：给侧栏留一个「完成/出错」标记，并把结果落盘
                self.tree_marks[state.number] = "error" if failed else "done"
            self.save_tree(state.number)

    def _send_message_impl(
        self, user_input: str, emit: EventSink
    ) -> Optional[ConversationNode]:
        """发送消息的实际实现（已绑定好树上下文，见 send_message）。

        节点在流式输出前即创建并设为当前节点（UI 可立即“进入”新节点进行
        流式输出）；**出错时节点也会保留**（写入已生成的部分内容与 node.error），
        用户可以直接输入「继续」接着生成，不再整轮回滚。emit 会收到
        node_created / stream / tool / status / done / error 事件。

        多模态：待发送图片先上传为 Files API file_id（请求体极小、序列化稳定、
        不破坏前缀缓存），上传失败回退 base64 内联；图片消息自动切换视觉模型。

        用户可随时通过 interrupt() 打断（停止流式输出 / 终止正在执行的命令），
        已生成部分同样落盘为「中断」节点，可直接继续。
        """
        self._interrupt_requested = False
        if self.imported_content:
            user_input = self.imported_content + "\n\n" + user_input
            self.imported_content = None
            self.imported_files = []

        # 本轮待发送图片（已在上传前从待发送区摘出，避免重复发送）
        this_turn_images = list(self.pending_images)
        self.pending_images = []

        # 前置创建节点并设为当前节点：UI 立即进入新节点，流式输出归属该节点
        node = self._begin_node(user_input)
        node.user_images = this_turn_images
        emit(ControllerEvent.node_created(node))

        # 图片落地方式统一规划：detail=low 走内联（file 块不支持 detail），其余
        # 上传为 Files API file_id（含历史节点缺 file_id 的补传）
        self._prepare_attachments(node, emit)

        # MCP 后台连接的收口：首次发送时若还没连完，这里等它（状态提示已发出），
        # 保证本轮请求带上完整工具列表——工具中途就绪会导致前后两轮工具集不一致。
        self.wait_mcp_ready(emit=emit)

        # 构建发送消息（历史链 + 本轮；图片构造为 OpenAI 兼容内容块）
        messages = self.tree.get_messages_for_node(node)

        # 图片消息：仅 deepseek-flash 支持；其余模型给出明确提示（不自动切换）
        if self._messages_contain_images(messages):
            guard_err = self._ensure_vision_model(emit)
            if guard_err:
                emit(ControllerEvent.error(guard_err))
                return self._finalize_interrupted(node, guard_err, emit)
            limit_err = self._validate_request_images(node)
            if limit_err:
                emit(ControllerEvent.error(limit_err))
                return self._finalize_interrupted(node, limit_err, emit)

        # 请求体大小预检（内联 base64 才是大请求体来源：detail=low 内联或上传回退）
        inline_bytes = collect_inline_bytes(messages)
        if inline_bytes > VISION_REQUEST_MAX_BYTES:
            low_inlined = any(
                att.detail == "low" and not att.file_id and not att.is_url
                for att in self._attachments_for_node(node)
            )
            hint = (
                "输入 /set detail auto 改用 Files API 上传（file_id）可绕开请求体限制，"
                if low_inlined else "请减少图片数量或压缩后重试"
            )
            size_err = (
                f"图片 base64 总量超限（约 {inline_bytes // 1024 // 1024} MiB "
                f"> {VISION_REQUEST_MAX_BYTES // 1024 // 1024} MiB），{hint}"
            )
            emit(ControllerEvent.error(size_err))
            return self._finalize_interrupted(node, size_err, emit)

        final_answer: Optional[str] = None
        accumulated_content = ""
        accumulated_reasoning = ""
        accumulated_in_tok = 0
        accumulated_out_tok = 0
        accumulated_cache_hit = 0
        accumulated_cache_miss = 0
        # 最后一轮（产出最终回答的那次请求）的真实 usage：状态条「下次输入」用它推算
        last_prompt_tok = 0
        last_output_tok = 0
        tool_messages: List[Dict] = []
        file_degraded = False  # file_id 失效降级重试标志（只重试一次）

        def _on_chunk(content_delta: str, reasoning_delta: str) -> None:
            """累积本轮正文（出错时也要落盘到节点）+ 转发 UI。"""
            nonlocal accumulated_content
            if content_delta:
                accumulated_content += content_delta
            emit(ControllerEvent.stream(content_delta, reasoning_delta))

        def _finalize(reason: str) -> ConversationNode:
            """把当前已生成的部分落盘为「中断」节点（打断/错误共用）。"""
            return self._finalize_interrupted(
                node, reason, emit,
                content=accumulated_content,
                reasoning=accumulated_reasoning,
                input_tokens=accumulated_in_tok,
                output_tokens=accumulated_out_tok,
                cache_hit_tokens=accumulated_cache_hit,
                cache_miss_tokens=accumulated_cache_miss,
                last_prompt_tokens=last_prompt_tok,
                last_output_tokens=last_output_tok,
                tool_messages=tool_messages,
            )

        def _interrupted() -> bool:
            return self._interrupt_requested

        try:
            while True:
                if _interrupted():
                    emit(ControllerEvent.status("⏹ 已打断生成"))
                    return _finalize("用户已打断")
                sr = stream_response(
                    self.client,
                    messages,
                    self.current_model,
                    self.current_temperature,
                    user_input,
                    thinking_enabled=self.thinking_enabled,
                    reasoning_effort=self.reasoning_effort,
                    tools=self.llm_tools,
                    on_chunk=_on_chunk,
                    should_stop=_interrupted,
                )
                if _interrupted():
                    # 本轮已流出的思考/用量也计入中断节点（正文经 _on_chunk 已累积）
                    accumulated_reasoning += sr.reasoning or ""
                    accumulated_in_tok += sr.input_tokens
                    accumulated_out_tok += sr.output_tokens
                    accumulated_cache_hit += sr.cache_hit_tokens
                    accumulated_cache_miss += sr.cache_miss_tokens
                    last_prompt_tok = sr.input_tokens
                    last_output_tok = sr.output_tokens
                    emit(ControllerEvent.status("⏹ 已打断生成"))
                    return _finalize("用户已打断")
                if sr.error:
                    # file_id 失效兜底：消息含 file 块且尚未降级过时，清掉 file_id 用 base64 重试一次
                    if not file_degraded and self._messages_contain_file_blocks(messages):
                        file_degraded = True
                        emit(ControllerEvent.status(
                            "⚠️ 图片 file_id 似乎已失效，正在降级为 base64 内联重试…"
                        ))
                        self._invalidate_file_ids_in_chain(node)
                        messages = self.tree.get_messages_for_node(node)
                        # 降级后重新预检 base64 总量
                        inline_bytes = collect_inline_bytes(messages)
                        if inline_bytes > VISION_REQUEST_MAX_BYTES:
                            degrade_err = "降级后图片 base64 总量超限，无法发送"
                            emit(ControllerEvent.error(degrade_err))
                            return self._finalize_interrupted(
                                node, degrade_err, emit,
                                content=accumulated_content,
                                reasoning=accumulated_reasoning,
                                tool_messages=tool_messages,
                            )
                        continue
                    emit(ControllerEvent.error(sr.error))
                    return self._finalize_interrupted(
                        node, sr.error, emit,
                        content=accumulated_content,
                        # 失败轮的思考只在 sr 里（成功轮已累加到 accumulated_reasoning）
                        reasoning=accumulated_reasoning + (sr.reasoning or ""),
                        input_tokens=accumulated_in_tok + sr.input_tokens,
                        output_tokens=accumulated_out_tok + sr.output_tokens,
                        cache_hit_tokens=accumulated_cache_hit + sr.cache_hit_tokens,
                        cache_miss_tokens=accumulated_cache_miss + sr.cache_miss_tokens,
                        last_prompt_tokens=sr.input_tokens,
                        last_output_tokens=sr.output_tokens,
                        tool_messages=tool_messages,
                    )

                reasoning = sr.reasoning or ""
                if reasoning:
                    accumulated_reasoning += reasoning
                accumulated_in_tok += sr.input_tokens
                accumulated_out_tok += sr.output_tokens
                accumulated_cache_hit += sr.cache_hit_tokens
                accumulated_cache_miss += sr.cache_miss_tokens
                # 覆盖式记录（不是累加）：状态条只关心最后一次请求的真实 prompt
                last_prompt_tok = sr.input_tokens
                last_output_tok = sr.output_tokens

                if sr.tool_calls:
                    assistant_msg: Dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    }
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning

                    tool_results: List[Dict] = []
                    for tc in sr.tool_calls:
                        name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        args_repr = tc["function"]["arguments"]
                        emit(ControllerEvent.tool(name, args_repr, ""))
                        tool_result = self._run_tool(name, args, emit)
                        summary = (tool_result or "").strip()[:100].replace("\n", " ")
                        emit(ControllerEvent.tool(name, args_repr, summary or "（完成）"))

                        assistant_msg["tool_calls"].append(
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                        )
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_result if tool_result else "执行失败或无结果",
                            }
                        )

                    messages.append(assistant_msg)
                    messages.extend(tool_results)
                    tool_messages.append(assistant_msg)
                    tool_messages.extend(tool_results)
                    # 工具批次跑完后再检查打断：保持 assistant.tool_calls 与
                    # 结果消息一一对应（历史合法），已执行的命令也一并落盘
                    if _interrupted():
                        emit(ControllerEvent.status("⏹ 已打断生成"))
                        return _finalize("用户已打断")
                    continue

                if sr.content is not None:
                    final_answer = sr.content
                    break

                empty_err = "回答生成失败，请重试"
                emit(ControllerEvent.error(empty_err))
                return self._finalize_interrupted(
                    node, empty_err, emit,
                    content=accumulated_content,
                    reasoning=accumulated_reasoning,
                    input_tokens=accumulated_in_tok,
                    output_tokens=accumulated_out_tok,
                    cache_hit_tokens=accumulated_cache_hit,
                    cache_miss_tokens=accumulated_cache_miss,
                    last_prompt_tokens=last_prompt_tok,
                    last_output_tokens=last_output_tok,
                    tool_messages=tool_messages,
                )
        except KeyboardInterrupt:
            # 外部直接中断（如测试或未来前端）：同样保留已生成的部分，
            # 再抛出交给前端（TUI 走 interrupt() 标志，不会走到这里）。
            self._interrupt_requested = True
            _finalize("用户已打断")
            raise
        except Exception as e:
            # 未预期的异常：节点仍然保留（含已生成的部分内容），
            # 异常继续抛给调用方显示（调用方负责提示用户）
            self._finalize_interrupted(
                node, str(e) or e.__class__.__name__, emit,
                content=accumulated_content,
                reasoning=accumulated_reasoning,
                input_tokens=accumulated_in_tok,
                output_tokens=accumulated_out_tok,
                cache_hit_tokens=accumulated_cache_hit,
                cache_miss_tokens=accumulated_cache_miss,
                last_prompt_tokens=last_prompt_tok,
                last_output_tokens=last_output_tok,
                tool_messages=tool_messages,
            )
            raise

        title = generate_conversation_title(self.client, user_input)
        node.assistant_msg = final_answer
        node.reasoning = accumulated_reasoning
        node.input_tokens = accumulated_in_tok
        node.output_tokens = accumulated_out_tok
        node.last_prompt_tokens = last_prompt_tok
        node.last_output_tokens = last_output_tok
        node.cache_hit_tokens = accumulated_cache_hit
        node.cache_miss_tokens = accumulated_cache_miss
        node.title = title
        if tool_messages:
            node.tool_messages = tool_messages
        # 本节点消息是在 assistant 输出前构建并缓存的（缺本节点回答/工具消息），
        # 失效缓存让下一次发送重建完整消息链（含本节点 assistant/tool 消息）。
        node.cached_messages = None
        self.tree.current_node = node
        self._auto_title_subtree(node, emit)
        emit(ControllerEvent.done(node))
        return node

    # ---------------- 多模态：图片上传与模型守卫 ----------------

    def _prepare_attachments(self, node: ConversationNode, emit: EventSink) -> None:
        """决定本轮链上每张本地图片的落地方式，并完成上传。

        - ``detail=low``：只有内联 ``image_url`` 才支持 detail（``file`` 块没有该
          字段，官方明确通过 file_id 传图时 detail 被忽略），因此在请求级预算内
          优先 base64 内联，让「省 token」真正生效；
        - 其余情况（detail 非 low、超过内联单图上限、预算不够）：上传 Files API，
          换 file_id 以便跨轮复用、请求体极小；
        - 历史节点里还没上传过的图片一并补传（尽力而为，静默失败）。

        上传会改变消息序列化结果（base64 → file 块），因此只要有任何变化就失效
        链上节点缓存——子节点缓存包含父节点消息。
        """
        chain = self._path_to_root(node)
        images = [att for n in chain for att in n.user_images]
        inline, to_upload = split_low_inline(images, VISION_LOW_INLINE_BUDGET_BYTES)
        uploadable = [
            att for att in to_upload
            if os.path.exists(os.path.expanduser(att.source))
        ]
        missing = len(to_upload) - len(uploadable)
        if inline:
            emit(ControllerEvent.status(
                f"detail=low：{len(inline)} 张图片以内联 base64 发送（省 token）"
            ))
        if uploadable:
            emit(ControllerEvent.status(f"正在上传图片（{len(uploadable)} 张）…"))
        changed = False
        for i, att in enumerate(uploadable, start=1):
            emit(ControllerEvent.status(
                f"正在上传图片 {i}/{len(uploadable)}：{att.name}…"
            ))
            try:
                att.file_id = upload_image(self.client, att.source)
                changed = True
            except FilesAPIError as e:
                emit(ControllerEvent.status(f"[WARN] {e}（将以内联 base64 发送）"))
        if missing:
            emit(ControllerEvent.status(
                f"[WARN] {missing} 张历史图片的本地文件已不存在，未上传"
            ))
        low_uploaded = [att for att in uploadable if att.detail == "low"]
        if low_uploaded:
            emit(ControllerEvent.status(
                f"detail=low 的 {len(low_uploaded)} 张图片超出内联预算"
                "（或单图超 32MiB），已改用 file_id 上传，其 detail 将被忽略"
            ))
        if changed:
            for n in chain:
                n.cached_messages = None

    def _invalidate_file_ids_in_chain(self, node: ConversationNode) -> None:
        """将当前节点链上所有图片的 file_id 置为 None（降级为 base64），并失效消息缓存。

        用于 Files API file_id 失效（过期/被删/API 端异常）时的自动回退。
        只要清掉了任何 file_id，就失效链上所有节点缓存——子节点缓存包含父节点消息。
        """
        any_changed = False
        for n in self._path_to_root(node):
            for att in n.user_images:
                if att.file_id:
                    att.file_id = None
                    any_changed = True
        if any_changed:
            for n in self._path_to_root(node):
                n.cached_messages = None

    @staticmethod
    def _messages_contain_file_blocks(messages: List[Dict]) -> bool:
        """消息列表中是否存在 Files API file 块（type=file）。"""
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "file":
                        return True
        return False

    @staticmethod
    def _messages_contain_images(messages: List[Dict]) -> bool:
        """消息列表中是否存在图片内容块（image_url / file）。"""
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image_url", "file"):
                        return True
        return False

    def _ensure_vision_model(self, emit: EventSink) -> Optional[str]:
        """检查当前模型是否支持图片理解；不支持时返回错误信息（不自动切换）。

        2026-09 起仅 `deepseek-flash` 支持图片（Pro 不支持），因此不再自动切换
        模型，而是明确提示用户手动 `/set model flash`。
        """
        if self.current_model in VISION_MODELS:
            return None
        return (
            f"当前模型 {self.current_model} 不支持图片理解，"
            f"请先 /set model flash 切换后重试（或移除图片）；"
            f"本轮提问与图片已保存在该节点，切换后直接继续即可"
        )

    def _attachments_for_node(self, node: ConversationNode) -> List[ImageAttachment]:
        """当前请求链（根 → node）上的全部图片附件（按发送顺序）。"""
        return [att for n in self._path_to_root(node) for att in n.user_images]

    def _validate_request_images(self, node: ConversationNode) -> Optional[str]:
        """按官方限制校验本次请求的图片集；超限返回错误信息，否则 None。

        官方限制（2026-09）：单请求 ≤600 张；内联/URL 单图 ≤32MiB、Files API
        单图 ≤64MiB；不含 file_id 的图片总量 ≤64MiB、含 file_id ≤200MiB；
        单边 ≤8192px（单请求 ≥15 张时 ≤4096px）。
        """
        attachments = self._attachments_for_node(node)
        if not attachments:
            return None
        count = len(attachments)
        if count > VISION_REQUEST_IMAGES_MAX_COUNT:
            return (
                f"图片数量超限（{count} > {VISION_REQUEST_IMAGES_MAX_COUNT} 张/请求），"
                "请减少图片后重试"
            )
        side_limit = (
            VISION_MAX_SIDE_MANY
            if count >= VISION_MANY_IMAGES_THRESHOLD
            else VISION_MAX_SIDE
        )
        oversize_side = oversize_side_attachments(attachments, side_limit)
        if oversize_side:
            return (
                f"图片尺寸超限（单边最大 {side_limit}px"
                + (f"；{count} 张时降为 {VISION_MAX_SIDE_MANY}px"
                   if count >= VISION_MANY_IMAGES_THRESHOLD else "")
                + f"）：{'、'.join(oversize_side[:3])}"
                + ("…" if len(oversize_side) > 3 else "")
            )
        oversize_inline = oversize_inline_attachments(attachments)
        if oversize_inline:
            return (
                "图片超过内联上限（32MiB）且未能上传到 Files API："
                + "、".join(oversize_inline[:3])
                + ("…" if len(oversize_inline) > 3 else "")
                + "；请压缩图片或检查网络后重试"
            )
        inline_total = sum(
            att.size_bytes or 0
            for att in attachments
            if not att.file_id and not att.is_url
        )
        if inline_total > VISION_REQUEST_INLINE_TOTAL_MAX_BYTES:
            return (
                f"内联图片总量超限（{inline_total // 1024 // 1024} MiB > "
                f"{VISION_REQUEST_INLINE_TOTAL_MAX_BYTES // 1024 // 1024} MiB），"
                "请减少图片或压缩后重试"
            )
        total = sum(att.size_bytes or 0 for att in attachments)
        if total > VISION_REQUEST_TOTAL_MAX_BYTES:
            return (
                f"图片总大小超限（{total // 1024 // 1024} MiB > "
                f"{VISION_REQUEST_TOTAL_MAX_BYTES // 1024 // 1024} MiB），"
                "请减少图片后重试"
            )
        return None

    def _begin_node(self, user_input: str) -> ConversationNode:
        """在流式输出前前置创建新节点并设为当前节点（UI 立即进入新节点）。"""
        title = (user_input.strip()[:24] or "新对话")
        if self.tree.current_node is None:
            node = self.tree.create_root(user_input, "", "", title, 0, 0)
        else:
            node = self.tree.add_child(
                self.tree.current_node, user_input, "", "", title, 0, 0
            )
        self.tree.current_node = node
        return node

    def _finalize_interrupted(
        self,
        node: ConversationNode,
        error: str,
        emit: EventSink,
        *,
        content: str = "",
        reasoning: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
        last_prompt_tokens: int = 0,
        last_output_tokens: int = 0,
        tool_messages: Optional[List[Dict]] = None,
    ) -> ConversationNode:
        """生成失败/中断时保留节点（含已生成的部分内容），供用户接着「继续」。

        历史行为是「出错即回滚节点」，结果是半截回答连同提问一起消失，用户
        只能从头再问。现在改为落盘：已流式生成的部分正文/思考、已产生的
        token 用量、以及已经跑完的工具调用/结果都写进节点，并在 node.error
        记下失败原因；UI 据此在树上标「⚠ 中断」、在正文里提示可继续。空回答
        节点不会再作为 assistant 消息发回模型（ConversationNode.get_messages
        只在 assistant_msg 非空时追加），所以下一轮输入「继续」时历史仍然合法。
        """
        node.assistant_msg = content
        node.reasoning = reasoning
        node.error = error
        node.input_tokens = input_tokens
        node.output_tokens = output_tokens
        node.last_prompt_tokens = last_prompt_tokens
        node.last_output_tokens = last_output_tokens
        node.cache_hit_tokens = cache_hit_tokens
        node.cache_miss_tokens = cache_miss_tokens
        if tool_messages:
            # 保留已完成的工具轮：用户「继续」时模型能接着用已检索到的资料
            node.tool_messages = tool_messages
        # 本节点消息可能是在流式输出前构建并缓存的（缺本节点 assistant 消息），
        # 失效缓存让下一次发送重建完整消息链。
        node.cached_messages = None
        self.tree.current_node = node
        emit(ControllerEvent.done(node))
        return node

    # ---------------- 多模态：Files API 文件管理 ----------------

    def files_list(self, limit: int = FILES_LIST_PAGE) -> Dict:
        """列出最近上传的图片文件（/files list）。默认最新在前。"""
        return list_files(self.client, limit=limit, order="desc")

    def files_all(self) -> List[Dict]:
        """翻页取回全部已上传文件（/files clean 需要全量清单）。"""
        return list_all_files(self.client)

    def files_retrieve(self, file_id: str) -> Dict:
        """查询单个已上传文件的信息（/files info）。"""
        return retrieve_file(self.client, file_id)

    def files_referenced_ids(self) -> set:
        """所有对话树（含未保存的内存改动）里引用到的 file_id 集合。"""
        self.save_tree()  # 当前树的未保存改动也要算进去
        ids: set = set()
        for number in self._store.numbers():
            st = self._states.get(number)
            tree = st.tree if st is not None else None
            if tree is None:
                data = self._store.load_tree(number) or {}
                tree_data = data.get("tree")
                tree = (
                    ConversationTree.from_dict(tree_data)
                    if isinstance(tree_data, dict) else None
                )
            if tree is None:
                continue
            for n in tree.nodes.values():
                for att in n.user_images:
                    if att.file_id:
                        ids.add(att.file_id)
        return ids

    def files_clean(self) -> Dict:
        """找出「没有任何对话树引用」的远端文件（不删除）。

        返回 ``{"unused": [...], "used": int, "total": int}``；远端文件数超过官方
        配额时上传会失败，因此需要这个入口定期回收（官方配额：10000 个 / 25GiB）。
        """
        remote = self.files_all()
        referenced = self.files_referenced_ids()
        unused = [f for f in remote if f.get("id") not in referenced]
        return {
            "unused": unused,
            "used": len(remote) - len(unused),
            "total": len(remote),
        }

    def files_delete(self, file_id: str) -> bool:
        """删除一个已上传的图片文件（/files delete），并同步清除对话树中的失效引用。"""
        delete_file(self.client, file_id)
        # 同步：把所有节点中引用该 file_id 的图片标记为失效，下次发送自动回退 base64
        any_changed = False
        for n in self.tree.nodes.values():
            for att in n.user_images:
                if att.file_id == file_id:
                    att.file_id = None
                    any_changed = True
        # 子节点缓存包含父节点消息，任一节点引用失效则全树缓存失效
        if any_changed:
            for n in self.tree.nodes.values():
                n.cached_messages = None
        return True

    def delete_node(self, node_id: str) -> bool:
        """删除节点及其子节点，并尽力删除其关联的 Files API 文件。

        比 tree.delete_node 多了文件清理；失败静默（文件可 /files list 手工清理）。
        """
        node = self.tree.nodes.get(node_id)
        if node is None:
            return False
        if node_id == "main" or (self.tree.root and node_id == self.tree.root.id):
            return False
        to_delete: set = set()
        self.tree._collect_descendants(node, to_delete)
        file_ids: List[str] = []
        for nid in to_delete:
            n = self.tree.nodes.get(nid)
            if n:
                for att in n.user_images:
                    if att.file_id:
                        file_ids.append(att.file_id)
        if not self.tree.delete_node(node_id):
            return False
        self._cleanup_compaction_boundary()
        for fid in file_ids:
            try:
                delete_file(self.client, fid)
            except FilesAPIError:
                pass
        return True

    def _cleanup_compaction_boundary(self) -> None:
        """若压缩摘要的 boundary 节点已被删除，则清除压缩状态（避免悬挂）。"""
        if self.tree.compaction:
            bid = self.tree.compaction.get("boundary_id")
            if not bid or bid not in self.tree.nodes:
                self.tree.compaction = None

    def delete_nodes(self, node_ids: List[str]) -> dict:
        """批量删除节点（/delete a1 b3 g5）。

        同一批中若某节点是另一待删节点的子孙，则仅删除祖先（子孙随祖先级联
        删除，不会因「父节点已删、子节点找不到」而报错）；根节点跳过。
        返回 {"deleted": [ids], "skipped": [ids]}。
        """
        tree = self.tree
        seen: set = set()
        ordered: List[str] = []
        for nid in node_ids:
            nid = nid.strip()
            if nid and nid not in seen:
                seen.add(nid)
                ordered.append(nid)
        # 先分出根节点（跳过）与可删候选，避免根节点被误当作「祖先覆盖」
        deletable: List[str] = []
        skipped: List[str] = []
        for nid in ordered:
            node = tree.nodes.get(nid)
            if node is None:
                continue  # 已不存在（可能已被本批更早的父节点删除级联移除）→ 不报错
            if nid == "main" or (tree.root and nid == tree.root.id):
                skipped.append(nid)
            else:
                deletable.append(nid)
        deleted: List[str] = []
        for nid in deletable:
            # 该节点是其他待删节点的子孙 → 交给祖先级联删除
            if any(
                other != nid
                and other in tree.nodes
                and self._is_node_descendant(nid, other)
                for other in deletable
            ):
                continue
            node = tree.nodes.get(nid)
            if node is None:
                continue  # 已被本批更早的祖先节点级联删除
            to_delete: set = set()
            tree._collect_descendants(node, to_delete)
            file_ids: List[str] = []
            for did in to_delete:
                n = tree.nodes.get(did)
                if n:
                    for att in n.user_images:
                        if att.file_id:
                            file_ids.append(att.file_id)
            if tree.delete_node(nid):
                deleted.append(nid)
                for fid in file_ids:
                    try:
                        delete_file(self.client, fid)
                    except FilesAPIError:
                        pass
        self._cleanup_compaction_boundary()
        return {"deleted": deleted, "skipped": skipped}

    def _is_node_descendant(self, node_id: str, ancestor_id: str) -> bool:
        """node_id 是否为 ancestor_id 的子孙节点。"""
        cur = self.tree.nodes.get(node_id)
        while cur is not None and cur.parent_id:
            if cur.parent_id == ancestor_id:
                return True
            cur = self.tree.nodes.get(cur.parent_id)
        return False

    def _run_tool(self, name: str, args: dict, emit: EventSink) -> str:
        """执行一个工具，返回文本结果。"""
        if name == "query_conversation_tree":
            return self._query_conversation_tree(
                args.get("root", ""), args.get("search", ""), args.get("tree", "")
            )
        if name == "read_conversation_nodes":
            return self._read_conversation_nodes(
                args.get("node_ids", ""), args.get("tree", "")
            )
        if name == "write_file":
            return self._write_file(args.get("filepath", ""), args.get("content", ""))
        if name == "edit_file":
            return self._edit_file(
                args.get("filepath", ""),
                args.get("old_string", ""),
                args.get("new_string", ""),
            )
        if name == "execute_command":
            return self._execute_command_tool(args, emit)
        if self._mcp is not None and name in self._mcp_tool_names:
            return self._mcp.call(name, args)
        return f"未知工具: {name}"

    # ---------------- 工具实现 ----------------

    def _tree_for_arg(self, tree_arg: Any) -> Optional[ConversationTree]:
        """解析跨树查询工具的 tree 参数（编号）：空 = 当前树，非法/不存在 = None。"""
        text = str(tree_arg if tree_arg is not None else "").strip()
        if not text:
            return self.tree
        try:
            number = int(text)
        except ValueError:
            return None
        if not self._store.has_tree(number):
            return None
        return self._load_state(number).tree

    def tree_overview_lines(self) -> List[str]:
        """所有对话树的编号 + 首条提问标题（跨树查询工具返回给模型的索引）。"""
        lines = []
        for number in self._store.numbers():
            st = self._states.get(number)
            tree = st.tree if st is not None else None
            if tree is None:
                data = self._store.load_tree(number) or {}
                tree_data = data.get("tree")
                tree = (
                    ConversationTree.from_dict(tree_data)
                    if isinstance(tree_data, dict)
                    else None
                )
            title = ""
            if tree is not None and tree.root is not None:
                title = tree.root.title or ""
            mark = "（当前对话树）" if number == self._active else ""
            lines.append(f"树 {number}{mark}: {title or '（空）'}")
        return lines

    def _query_conversation_tree(
        self, root: str = "", search: str = "", tree_arg: Any = ""
    ) -> str:
        tree = self._tree_for_arg(tree_arg)
        if tree is None:
            return f"（对话树 {tree_arg} 不存在；可用编号: {'、'.join(str(n) for n in self._store.numbers()) or '无'}）"
        scope = f"树 {tree_arg}" if str(tree_arg).strip() else f"树 {self._active}"
        if not tree.root:
            return f"（{scope} 暂无对话记录）"

        if search:
            results = []
            kw = search.lower()
            for nid, node in tree.nodes.items():
                if kw in (node.title or "").lower() or kw in (node.user_msg or "").lower():
                    results.append(f"{nid}: {node.title}")
            return "\n".join(results) if results else f"（{scope} 中未找到包含「{search}」的节点）"

        if root:
            nodes_in_tree = []
            root_id = next(
                (
                    nid
                    for nid in tree.nodes
                    if tree._node_letter_prefix(nid) == root
                    and tree.nodes[nid].parent_id == "main"
                ),
                None,
            )
            if not root_id:
                return f"（{scope} 的子对话树 {root} 不存在）"
            descendants = set()
            tree._collect_descendants(tree.nodes[root_id], descendants)
            for nid in sorted(descendants):
                node = tree.nodes[nid]
                depth = 0
                cur = node
                while cur.parent_id and cur.parent_id != "main":
                    depth += 1
                    cur = tree.nodes.get(cur.parent_id)
                nodes_in_tree.append(f"{'  ' * depth}{nid}: {node.title}")
            return "\n".join(nodes_in_tree)

        lines = []
        # 不指定 tree 时顺带列出所有对话树，让模型知道有哪几棵树可以查
        if not str(tree_arg).strip():
            overview = self.tree_overview_lines()
            if len(overview) > 1:
                lines.append("全部对话树（可用 tree 参数指定编号查询）：")
                lines.extend(f"  {ln}" for ln in overview)
                lines.append("")
        lines.append(f"main: {tree.root.title}")
        for child in tree.root.children:
            prefix = tree._get_subtree_root_prefix(child.id)
            if prefix:
                count = tree.count_subtree_nodes(prefix)
                suffix = tree.subtree_titles.get(prefix, child.title)
                lines.append(f"  {prefix}: {suffix}（{count}个节点）")
        return "\n".join(lines)

    def _read_conversation_nodes(self, node_ids: str, tree_arg: Any = "") -> str:
        tree = self._tree_for_arg(tree_arg)
        if tree is None:
            return f"（对话树 {tree_arg} 不存在）"
        parts = []
        for nid in node_ids.split(","):
            nid = nid.strip()
            if not nid:
                continue
            node = tree.nodes.get(nid)
            if not node:
                parts.append(f"--- {nid} ---\n（节点不存在）")
            else:
                parts.append(
                    f"--- {nid} ---\n"
                    f"用户: {node.user_msg}\n"
                    + (f"思考过程: {node.reasoning}\n" if node.reasoning else "")
                    + f"回答: {node.assistant_msg}"
                )
        return "\n\n".join(parts) if parts else "（未指定节点）"

    def _write_file(self, filepath: str, content: str) -> str:
        filepath = os.path.expanduser(filepath)
        exists = os.path.exists(filepath)
        mode = "覆盖已有文件" if exists else "创建新文件"
        line_count = content.count("\n") + 1
        preview = content
        if line_count > 10:
            preview_lines = content.split("\n")[:5]
            preview = "\n".join(preview_lines) + f"\n…（共 {line_count} 行）"
        details = f"路径: {filepath}\n操作: {mode}\n内容: {line_count} 行, {len(content)} 字符\n预览:\n{preview}"
        if self.file_confirm and not self.confirm(f"即将{'覆盖' if exists else '写入'}文件", details):
            return "用户已取消操作"
        if self._interrupt_requested:
            return "用户已打断，未写入文件"
        if self._mcp is not None and "write_file" in self._mcp_tool_names:
            return self._mcp.call("write_file", {"filepath": filepath, "content": content})
        return "写文件工具不可用"

    def _edit_file(self, filepath: str, old_string: str, new_string: str) -> str:
        filepath = os.path.expanduser(filepath)
        if not os.path.exists(filepath):
            return f"文件不存在: {filepath}"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"读取文件失败: {e}"
        if old_string not in content:
            return "未找到匹配的原文，请确保 old_string 与文件内容完全一致（包括空格和换行）"
        details = f"路径: {filepath}\n替换内容:\n"
        for line in old_string.split("\n"):
            details += f"  - {line}\n"
        details += "  替换为:\n"
        for line in new_string.split("\n"):
            details += f"  + {line}\n"
        if self.file_confirm and not self.confirm("即将修改文件", details):
            return "用户已取消操作"
        if self._interrupt_requested:
            return "用户已打断，未修改文件"
        if self._mcp is not None and "edit_file" in self._mcp_tool_names:
            return self._mcp.call(
                "edit_file",
                {"filepath": filepath, "old_string": old_string, "new_string": new_string},
            )
        return "编辑文件工具不可用"

    def _execute_command_tool(self, args: dict, emit: EventSink) -> str:
        command = args.get("command", "")
        timeout = args.get("timeout", EXEC_DEFAULT_TIMEOUT)
        # 工作目录优先级：模型显式 cwd > /set workspace > 启动目录（服务端兜底）
        cwd = args.get("cwd") or self.workspace or None
        call_args: Dict[str, Any] = {"command": command, "timeout": timeout}
        for key in ("cwd", "env", "shell", "max_output"):
            if args.get(key) is not None:
                call_args[key] = args[key]
        if self.workspace and not call_args.get("cwd"):
            call_args["cwd"] = self.workspace

        def mcp_call() -> str:
            """执行命令；若在审核/确认期间用户已打断，则不再启动进程。"""
            if self._mcp is None or "execute_command" not in self._mcp_tool_names:
                return "执行命令工具不可用"
            if self._interrupt_requested:
                return "用户已打断，未执行此命令"
            return self._mcp.call("execute_command", call_args)

        def _ctx_line() -> str:
            parts = [f"工作目录: {cwd or '（启动目录）'}"]
            if args.get("shell") and args["shell"] != "sh":
                parts.append(f"shell: {args['shell']}")
            return " | ".join(parts)

        if self.audit_level == 4:
            emit(ControllerEvent.status("▸ execute_command（无审核）"))
            return mcp_call()

        # 高危硬门：命中正则模式时，任何审核级别（除无审核）都强制用户确认
        if matches_dangerous(command):
            if self.confirm(
                "高危命令",
                f"命令: {command}\n\n⚠️ 匹配到高危命令模式，确认执行？\n{_ctx_line()}",
            ):
                return mcp_call()
            return "用户未确认执行此命令"

        if self.audit_level == 3:
            emit(ControllerEvent.status("▸ execute_command（文本审核通过）"))
            return mcp_call()

        # 只读快速通道：无副作用命令跳过 AI 审核（level-1 仍确认，level-2 自动执行）
        if is_safe_readonly(command):
            if self.audit_level == 2:
                emit(ControllerEvent.status(f"▸ {command[:60]}（只读命令，自动执行）"))
                return mcp_call()
            if self.confirm(
                "执行确认",
                f"命令: {command}\n\n审核: 只读命令快速通道（无副作用）\n{_ctx_line()}",
            ):
                return mcp_call()
            return "用户未确认执行此命令"

        level, desc, risk, audit_reasoning = audit_command(self.client, command)
        if audit_reasoning:
            emit(ControllerEvent.status(f"🧠 审核思考: {audit_reasoning}"))
        risk_text = f"\n⚠️ {risk}" if risk else ""
        if self.audit_level == 2 and level <= 2:
            emit(ControllerEvent.status(f"▸ {desc}（等级{level}/5，自动执行）"))
            return mcp_call()
        if self.confirm(
            "执行确认",
            f"命令: {command}\n\n审核: 等级 {level}/5 | {desc}{risk_text}\n{_ctx_line()}",
        ):
            return mcp_call()
        return "用户未确认执行此命令"

    # ---------------- 内部 ----------------

    def _auto_title_subtree(self, node: ConversationNode, emit: EventSink) -> None:
        if not self.tree.root or node.id == "main":
            return
        prefix = self.tree._get_subtree_root_prefix(node.id)
        if not prefix or prefix in self.tree.subtree_titles:
            return
        count = self.tree.count_subtree_nodes(prefix)
        if count == 3:
            root_id = next(
                (
                    nid
                    for nid in self.tree.nodes
                    if self.tree._node_letter_prefix(nid) == prefix
                    and self.tree.nodes[nid].parent_id == "main"
                ),
                None,
            )
            if not root_id:
                return
            descendants = set()
            self.tree._collect_descendants(self.tree.nodes[root_id], descendants)
            titles = []
            for nid in sorted(descendants):
                n = self.tree.nodes.get(nid)
                if n and n.title:
                    titles.append(f"{nid}: {n.title}")
            prompt = (
                "以下是一组对话中各部分的标题，请为这组对话取一个不超过10字的总标题，"
                "只输出标题，不要有其他解释。\n\n" + "\n".join(titles)
            )
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_FLASH,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=30,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                title = resp.choices[0].message.content.strip()
                if title:
                    self.tree.subtree_titles[prefix] = title
                    emit(ControllerEvent.status(f"已自动为对话树「{prefix}」命名: {title}"))
            except Exception:
                pass
