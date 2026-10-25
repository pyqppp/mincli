"""ChatController —— mincli 纯逻辑层（无任何 UI 依赖）。

管理多棵对话树、设置、流式输出、工具调用与持久化。
Textual TUI 通过本控制器驱动；UI 通过 ControllerEvent 回调接收增量更新
（流式内容 / 工具调用 / 状态 / 完成）。

多对话树：
- 每棵树一个文件（见 mincli/trees.py），只存节点树与该树的输入草稿；
- 模型/温度/思考模式/能力开关都是全局的，统一来自 ~/.mincli/config.toml；
- 系统提示词全局唯一（~/.mincli/prompt.md），客户端不显示、不可编辑；
- 工具只有三组固定内置：文件、联网、对话树查询，按配置下发（见 tools/builtin.py）；
- 同一时刻只允许一棵树在生成：后台生成期间切走，生成继续，但另一棵树
  必须等它结束才能发送（见 ChatApp._send_user_text）。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from mincli.attachments import (
    KIND_IMAGE,
    Attachment,
    collect_stale_attachments,
    store_attachment_text,
)
from mincli.config import (
    COMPACT_MAX_TOKENS,
    COMPACT_SOURCE_MAX_CHARS,
    COMPACT_REASONING_MAX_CHARS,
    COMPACT_TOOL_RESULT_MAX_CHARS,
    EFFORTS,
    FILE_MODES,
    FILES_LIST_PAGE,
    MODEL_FLASH,
    MODEL_PRO,
    VISION_DEFAULT_DETAIL,
    VISION_LOW_INLINE_BUDGET_BYTES,
    VISION_MAX_SIDE,
    VISION_MAX_SIDE_MANY,
    VISION_MANY_IMAGES_THRESHOLD,
    VISION_REQUEST_IMAGES_MAX_COUNT,
    VISION_REQUEST_INLINE_TOTAL_MAX_BYTES,
    VISION_REQUEST_MAX_BYTES,
    VISION_REQUEST_TOTAL_MAX_BYTES,
    VISION_MODELS,
    Config,
    load_config,
    normalize_model_name,
    save_config,
    system_prompt,
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
from mincli.tool_markup import MARKUP_FALLBACK, ToolMarkupFilter, strip_tool_markup
from mincli.tools.builtin import ToolContext, available_tools, dispatch_tool
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
    image_placeholder_text,
    oversize_inline_attachments,
    oversize_side_attachments,
    split_low_inline,
)
from mincli.trees import TreeStore, color_of


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

@dataclass
class TreeState:
    """一棵对话树的运行时状态（编号 0 表示「尚未创建任何对话树」的占位）。"""

    number: int
    color: int
    tree: ConversationTree
    draft: str = ""
    # 草稿里的附件元数据（切换树再回来时恢复输入框里的引用块）
    draft_attachments: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_placeholder(self) -> bool:
        return self.number <= 0


class ChatController:
    """对话引擎：状态 + 多对话树 + 流式输出 + 工具调用 + 持久化。"""

    def __init__(
        self,
        client: OpenAI,
        config: Optional[Config] = None,
        default_system: Optional[str] = None,
        trees_dir: Optional[str] = None,
    ) -> None:
        self.client = client
        # 配置是唯一事实来源；界面开关改动后立即写回 config.toml
        self.config: Config = config or load_config()
        # ---------------- 全局设置（所有对话树共享） ----------------
        self.current_system = default_system or system_prompt(self.config)
        self.current_temperature = self.config.temperature
        self.current_model = normalize_model_name(self.config.model)
        # 从磁盘加载时，若旧模型名被自动改写，这里记录原始名（供 UI 提示）
        self.model_migrated_from: Optional[str] = None
        self.thinking_enabled = self.config.thinking
        self.reasoning_effort = self.config.effort
        # 用户主动打断标志：streaming 循环在每个 chunk 处检查，工具调用后也会检查。
        self._interrupt_requested: bool = False

        # 图片 detail 默认值（low 省 token，auto≈original 最清晰）
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

        self.temp_dir = tempfile.mkdtemp(prefix="mincli_")
        self.temp_files: Dict[str, str] = {}

        # UI 注入的确认回调（写文件等工具用）。默认拒绝（安全）。
        self.confirm: Callable[[str, str], bool] = lambda title, text: False

        self.llm_tools: List[Dict] = available_tools(self.config, has_attachments=False)

        self.session_loaded = self._restore_from_store()
        self._collect_attachments()

    # ---------------- 对话树：加载 / 保存 ----------------

    def _restore_from_store(self) -> bool:
        """从磁盘恢复上次激活的对话树。

        全局设置以 config.toml 为准，这里只恢复「上次在哪棵树」。
        返回是否恢复了已有对话树（False = 一棵都没有，UI 应先引导新建）。
        """
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

    def _collect_attachments(self) -> int:
        """启动时回收没有任何会话引用的附件正文文件，返回删除数量。"""
        referenced = set()
        for number in self._store.numbers():
            data = self._store.load_tree(number) or {}
            nodes = (data.get("tree") or {}).get("nodes") or {}
            for node_data in nodes.values():
                if not isinstance(node_data, dict):
                    continue
                for meta in node_data.get("attachments") or []:
                    if isinstance(meta, dict) and meta.get("id"):
                        referenced.add(str(meta["id"]))
        return collect_stale_attachments(self.config.attachment_dir, referenced)

    def _reset_placeholder(self) -> None:
        st = TreeState(number=0, color=1, tree=ConversationTree(self.current_system))
        self._apply_system_prompt(st)
        self._states[0] = st
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
        # 系统提示词由「全局设置 + 本树能力挂载」共同决定，树文件里的旧副本忽略
        st = TreeState(
            number=number,
            color=self._store.color_of(number),
            tree=tree,
            draft=data.get("draft", "") or "",
            draft_attachments=list(data.get("draft_attachments") or []),
        )
        self._apply_system_prompt(st)
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
            "draft_attachments": list(st.draft_attachments),
            "saved_at": time.time(),
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
            st = TreeState(
                number=self._active, color=color_of(self._active), tree=value
            )
            self._states[self._active] = st
        else:
            st.tree = value
        self._apply_system_prompt(st)

    def _apply_system_prompt(self, st: TreeState) -> None:
        """提示词全局唯一：所有树用同一份。"""
        st.tree.system_prompt = self.current_system

    @property
    def active_system_prompt(self) -> str:
        """当前生效的系统提示词（来自 ~/.mincli/prompt.md）。"""
        return self.current_system

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
        return True

    def create_tree(self, activate: bool = True) -> int:
        """新建一棵空对话树并（默认）切过去，返回编号。所有树完全一样。"""
        if activate:
            self.save_tree()
        number = self._store.allocate()
        st = TreeState(
            number=number,
            color=self._store.color_of(number),
            tree=ConversationTree(self.current_system),
        )
        self._apply_system_prompt(st)
        self._states[number] = st
        # 刚建的空树没有并发改动，快照一定成功
        self._store.save_tree(number, self._tree_payload(st) or {})
        if activate:
            self._states.pop(0, None)
            self._active = number
            self._store.set_active(number)
        return number

    def delete_tree(self, number: int) -> dict:
        """删除整棵对话树（含数据文件）。编号会被下一个新树复用。"""
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
        return {"ok": True, "was_active": was_active, "remaining": self._store.numbers()}

    @property
    def draft(self) -> str:
        st = self._state()
        return st.draft if st is not None else ""

    @draft.setter
    def draft(self, value: str) -> None:
        st = self._state()
        if st is not None:
            st.draft = value or ""

    @property
    def draft_attachments(self) -> List[Dict[str, Any]]:
        """草稿里的附件元数据（供恢复引用块）。"""
        st = self._state()
        return list(st.draft_attachments) if st is not None else []

    @draft_attachments.setter
    def draft_attachments(self, value: List[Dict[str, Any]]) -> None:
        st = self._state()
        if st is not None:
            st.draft_attachments = list(value or [])

    # ---------------- 打断 ----------------

    def interrupt(self) -> bool:
        """请求打断当前轮生成（线程安全，可从 UI 线程调用）。

        置位打断标志：流式循环在下一个 chunk 处停止读取（见 stream_response
        的 should_stop），工具执行后也会检查并结束本轮。返回是否发起了打断请求。
        """
        self._interrupt_requested = True
        return True

    @property
    def interrupt_pending(self) -> bool:
        """本轮是否已请求过打断（UI 用于「再按一次强制退出」）。"""
        return self._interrupt_requested

    # ---------------- 设置 ----------------

    def set_temperature(self, temp: float) -> None:
        self.current_temperature = temp
        self.config.temperature = temp
        self._save_config()

    def set_model(self, model: str) -> bool:
        """在 flash / pro 之间切换（支持简写与旧名）。"""
        normalized = normalize_model_name(model or "")
        if normalized not in (MODEL_FLASH, MODEL_PRO):
            return False
        self.current_model = normalized
        self.config.model = normalized
        self._save_config()
        return True

    def set_detail(self, detail: str) -> bool:
        """设置图片 detail 默认值（low/auto/high/original）。"""
        if detail not in ("low", "auto", "high", "original"):
            return False
        self.image_detail = detail
        return True

    def set_thinking(self, on: bool) -> None:
        self.thinking_enabled = bool(on)
        self.config.thinking = bool(on)
        self._save_config()

    def set_effort(self, effort: str) -> bool:
        if effort not in EFFORTS:
            return False
        self.reasoning_effort = effort
        self.config.effort = effort
        self._save_config()
        return True

    # ---------------- 能力开关（底部状态栏三路入口共用） ----------------

    def set_files_mode(self, mode: str) -> bool:
        if mode not in FILE_MODES:
            return False
        self.config = self.config.with_files_mode(mode)
        self._save_config()
        self._rebuild_llm_tools()
        return True

    def set_web(self, enabled: bool) -> bool:
        self.config = self.config.with_web(enabled)
        self._save_config()
        self._rebuild_llm_tools()
        return True

    def set_recall(self, enabled: bool) -> bool:
        self.config = self.config.with_recall(enabled)
        self._save_config()
        self._rebuild_llm_tools()
        return True

    def _save_config(self) -> None:
        try:
            save_config(self.config)
        except OSError:
            pass

    def _rebuild_llm_tools(self) -> None:
        """按当前能力状态刷新工具列表（发送时按本轮附件再算一次）。"""
        self.llm_tools = available_tools(self.config, has_attachments=False)

    def _capability_prompt(self, tools: List[Dict]) -> str:
        """本轮能力状态：告诉模型哪些工具真的可用，避免它「凭空调用」。

        这段内容只随请求发送，不写进树、不落盘；开关变化时内容跟着变。
        """
        cfg = self.config
        names = [str(t.get("function", {}).get("name", "")) for t in tools]
        lines = ["## 本轮能力状态"]
        if names:
            lines.append(
                "- 本轮只提供这些工具：" + "、".join(names)
                + "。未列出的工具不可用，不要调用，也不要输出任何工具调用标记"
                "（如 <|DSML|>、<tool_call>）。"
            )
        else:
            lines.append(
                "- 本轮没有提供任何工具。不要调用工具，也不要输出任何工具调用标记"
                "（如 <|DSML|>、<tool_call>）。"
            )
        if not cfg.web_enabled:
            lines.append(
                "- 联网已关闭：web_search / web_fetch 不可用。需要联网时直接说明，"
                "并告诉用户按 Ctrl+W 或 /web on 打开。"
            )
        if not cfg.files_read_enabled:
            lines.append(
                "- 文件权限已关闭：read_file / list_directory 不可用，只能读本轮附件。"
                "需要读文件时请用户按 Ctrl+R 或 /file read。"
            )
        elif not cfg.files_write_enabled:
            lines.append(
                "- 文件权限为只读：可以读文件与列目录，不能写入。"
                "需要写入时请用户 /file confirm 或 /file direct。"
            )
        elif cfg.files_write_needs_confirm:
            lines.append("- 文件权限为写需确认：每次写入会弹窗，用户可能拒绝。")
        if not cfg.recall_enabled:
            lines.append(
                "- 历史对话查询已关闭：不要声称读过历史节点；需要时请用户 /recall on。"
            )
        if self.current_model == MODEL_PRO:
            lines.append("- 当前模型 deepseek-v4-pro 不支持图片。")
        return "\n".join(lines)

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
        self.tree = ConversationTree(self.active_system_prompt)
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

    # ---------------- 发送消息 ----------------

    def send_message(
        self,
        user_input: str,
        emit: EventSink,
        attachments: Optional[List[Attachment]] = None,
    ) -> Optional[ConversationNode]:
        """在当前对话树里发送一条消息（可能触发多轮工具调用），返回新节点。

        ``attachments`` 是本轮输入框里的附件：文本正文已由调用方用
        compose_message 拼进 ``user_input``；这里负责把图片登记到节点、把正文
        落盘供 read_attachment 续读，并随节点元数据持久化。

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
            return self._send_message_impl(user_input, _emit, attachments)
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
        self,
        user_input: str,
        emit: EventSink,
        attachments: Optional[List[Attachment]] = None,
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
        turn_attachments = list(attachments or [])

        # 前置创建节点并设为当前节点：UI 立即进入新节点，流式输出归属该节点
        node = self._begin_node(user_input)
        node.user_images = [
            att.image
            for att in turn_attachments
            if att.kind == KIND_IMAGE and att.image is not None
        ]
        # 正文先落盘，元数据里才带得上 stored_path
        for att in turn_attachments:
            if att.kind != KIND_IMAGE:
                store_attachment_text(att, self.config.attachment_dir)
        node.attachments = [att.metadata() for att in turn_attachments]
        emit(ControllerEvent.node_created(node))

        # 图片落地方式统一规划：detail=low 走内联（file 块不支持 detail），其余
        # 上传为 Files API file_id（含历史节点缺 file_id 的补传）
        self._prepare_attachments(node, emit)

        # 本轮工具集：能力开关 + 是否有附件（附件续读与文件权限无关）
        tools = available_tools(
            self.config, has_attachments=self._node_has_attachments(node)
        )
        self.llm_tools = tools

        # 构建发送消息（历史链 + 本轮；图片构造为 OpenAI 兼容内容块）。
        # system 里追加「本轮能力状态」：工具没下发的功能明确告诉模型不可用。
        messages = self.tree.get_messages_for_node(node)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": (
                    self.current_system + "\n\n" + self._capability_prompt(tools)
                ).strip(),
            }

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
            hint = "请减少图片数量或压缩后重试"
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

        markup = ToolMarkupFilter()

        def _on_chunk(content_delta: str, reasoning_delta: str) -> None:
            """累积本轮正文（出错时也要落盘到节点）+ 转发 UI。

            正文先过滤掉泄漏的工具调用标记：不给用户看乱码，也不写进历史。
            """
            nonlocal accumulated_content
            clean = markup.feed(content_delta) if content_delta else ""
            if clean:
                accumulated_content += clean
            if clean or reasoning_delta:
                emit(ControllerEvent.stream(clean, reasoning_delta))

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
                    tools=tools,
                    on_chunk=_on_chunk,
                    should_stop=_interrupted,
                )
                tail = markup.flush()
                if tail:
                    accumulated_content += tail
                    emit(ControllerEvent.stream(tail, ""))
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
                        tool_result = self._run_tool(name, args, emit, node)
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
                    cleaned, leaked, names = strip_tool_markup(sr.content)
                    if leaked:
                        markup.leaked = True
                        for name in names:
                            if name not in markup.names:
                                markup.names.append(name)
                    final_answer = (
                        cleaned
                        if cleaned.strip()
                        else (MARKUP_FALLBACK if leaked else sr.content)
                    )
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

        if markup.leaked:
            names = "、".join(markup.names) or "未提供的工具"
            emit(ControllerEvent.status(
                f"模型尝试调用未提供的工具（{names}），已忽略这次调用"
            ))
            if not (final_answer or "").strip():
                final_answer = MARKUP_FALLBACK

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
        模型，而是明确提示用户手动 `/model flash`。
        """
        if self.current_model in VISION_MODELS:
            return None
        return (
            f"当前模型 {self.current_model} 不支持图片理解，"
            f"请先 /model flash 切换后重试（或移除图片）；"
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

    def _node_has_attachments(self, node: ConversationNode) -> bool:
        """当前节点链条上是否有附件（决定是否下发 read_attachment）。"""
        return any(n.attachments for n in self._path_to_root(node))

    def _tool_context(self, node: ConversationNode) -> ToolContext:
        """一次工具调用需要的上下文：本会话全部附件 + 写确认 + 树查询。"""
        pool: List[Attachment] = []
        seen = set()
        for n in self._path_to_root(node):
            for meta in n.attachments:
                if not isinstance(meta, dict):
                    continue
                key = str(meta.get("id") or "") or str(meta.get("stored_path") or "")
                if key and key in seen:
                    continue
                seen.add(key)
                pool.append(Attachment.from_metadata(meta))
        return ToolContext(
            config=self.config,
            attachments=pool,
            confirm=self._confirm_write,
            recall=self._recall_dispatch,
        )

    def _confirm_write(self, description: str) -> bool:
        """写文件确认：只有「写需确认」模式才弹窗，其余模式直接放行。"""
        if not self.config.files_write_enabled:
            return False
        if not self.config.files_write_needs_confirm:
            return True
        return self.confirm("确认写入", description)

    def _recall_dispatch(self, kind: str, args: dict) -> str:
        if kind == "query":
            return self._query_conversation_tree(
                str(args.get("root", "") or ""),
                str(args.get("search", "") or ""),
                args.get("tree", ""),
            )
        return self._read_conversation_nodes(
            str(args.get("node_ids", "") or ""), args.get("tree", "")
        )

    def _run_tool(
        self,
        name: str,
        args: dict,
        emit: EventSink,
        node: Optional[ConversationNode] = None,
    ) -> str:
        """在进程内执行一个工具（无 MCP、无子进程）。"""
        if node is None:
            ctx = ToolContext(config=self.config)
        else:
            ctx = self._tool_context(node)
        return dispatch_tool(name, args, ctx)

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
