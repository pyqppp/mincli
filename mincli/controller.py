"""ChatController —— mincli 纯逻辑层（无任何 UI 依赖）。

管理对话树、设置、流式输出、工具调用与会话持久化。
Textual TUI 与纯文本前端都通过本控制器驱动；UI 通过
ControllerEvent 回调接收增量更新（流式内容 / 工具调用 / 状态 / 完成）。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from mincli.config import (
    MODEL_V4_FLASH,
    MODEL_V4_PRO,
    MODEL_V4_VISION,
    MODELS_AVAILABLE,
    COMPACT_MAX_TOKENS,
    COMPACT_SOURCE_MAX_CHARS,
    COMPACT_REASONING_MAX_CHARS,
    COMPACT_TOOL_RESULT_MAX_CHARS,
    EXEC_DEFAULT_TIMEOUT,
    VISION_DEFAULT_DETAIL,
    VISION_REQUEST_MAX_BYTES,
    load_models,
)
from mincli.helpers import (
    convert_formulas,
    estimate_input_price,
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
from mincli.tools.files import FilesAPIError, delete_file, list_files, upload_image
from mincli.tools.images import (
    ImageAttachment,
    collect_inline_bytes,
    image_placeholder_text,
    is_image_path,
    looks_like_image_target,
    make_path_attachment,
    make_url_attachment,
)
from mincli.tools.registry import TOOLS
from mincli.tools.web_fetch import fetch_webpage

try:
    from mincli.mcp_client import McpToolClient
except ImportError:  # pragma: no cover
    McpToolClient = None


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


class ChatController:
    """对话引擎：状态 + 对话树 + 流式输出 + 工具调用 + 持久化。"""

    SAVE_FILE = os.path.expanduser("~/.mincli_session.json")

    def __init__(
        self,
        client: OpenAI,
        default_system: str,
        default_temperature: float,
        default_model: str = MODEL_V4_FLASH,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        auto_start_mcp: bool = True,
    ) -> None:
        self.client = client
        self.current_system = default_system
        self.current_temperature = default_temperature
        self.current_model = default_model
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort
        self.audit_level: int = 1
        # 命令执行默认工作目录（/set workspace 设置；None 时用 mincli 启动目录）
        self.workspace: Optional[str] = None

        # 多模态：待发送图片（/import 导入的图片填充；发送后绑定到节点并清空）
        self.pending_images: List[ImageAttachment] = []
        # 图片 detail 全局默认（/set detail 可调；low 省 token，auto≈original 最清晰）
        self.image_detail: str = VISION_DEFAULT_DETAIL

        self.tree = ConversationTree(default_system)

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
        self.llm_tools: List[Dict] = list(TOOLS)

        self.session_loaded = self.load_session()
        if auto_start_mcp:
            self.start_mcp()

    # ---------------- MCP ----------------

    def start_mcp(self) -> None:
        if McpToolClient is None or self._mcp is not None:
            return
        self._mcp = McpToolClient()
        try:
            self._mcp.start()
            self._mcp_tool_names = self._mcp.tool_names()
        except Exception:
            self._mcp = None
            self._mcp_tool_names = set()
        self._rebuild_llm_tools()

    def _rebuild_llm_tools(self) -> None:
        self.llm_tools = list(TOOLS) + (self._mcp.tools() if self._mcp else [])

    def mcp_status(self) -> dict:
        return self._mcp.server_status() if self._mcp else {}

    def mcp_reload(self) -> None:
        if not self._mcp:
            raise RuntimeError("MCP 客户端未就绪")
        self._mcp.reload()
        self._mcp_tool_names = self._mcp.tool_names()
        self._rebuild_llm_tools()

    def close(self) -> None:
        if self._mcp:
            self._mcp.close()
            self._mcp = None

    # ---------------- 持久化 ----------------

    def save_session(self) -> None:
        filepath = self.SAVE_FILE
        try:
            data = {
                "system_prompt": self.current_system,
                "temperature": self.current_temperature,
                "model": self.current_model,
                "thinking_enabled": self.thinking_enabled,
                "reasoning_effort": self.reasoning_effort,
                "audit_level": self.audit_level,
                "workspace": self.workspace,
                "tree": self.tree.to_dict(),
                "imported_content": self.imported_content,
                "imported_files": self.imported_files,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def load_session(self) -> bool:
        self.session_loaded = False
        if not os.path.exists(self.SAVE_FILE):
            return False
        try:
            with open(self.SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            try:
                os.remove(self.SAVE_FILE)
            except Exception:
                pass
            return False

        self.current_system = data.get("system_prompt", self.current_system)
        self.current_temperature = data.get("temperature", self.current_temperature)
        self.current_model = data.get("model", self.current_model)
        self.thinking_enabled = data.get("thinking_enabled", False)
        self.reasoning_effort = data.get("reasoning_effort", "high")
        self.audit_level = data.get("audit_level", 1)
        self.workspace = data.get("workspace") or None

        tree_data = data.get("tree")
        if tree_data:
            self.tree = ConversationTree.from_dict(tree_data)
        else:
            self.tree = ConversationTree(self.current_system)

        self.imported_content = data.get("imported_content")
        self.imported_files = data.get("imported_files") or []
        self.session_loaded = True
        return True

    def delete_session_file(self) -> None:
        try:
            if os.path.exists(self.SAVE_FILE):
                os.remove(self.SAVE_FILE)
        except Exception:
            pass

    # ---------------- 设置 ----------------

    def set_system(self, system: str) -> None:
        self.current_system = system
        self.tree.system_prompt = system

    def set_temperature(self, temp: float) -> None:
        self.current_temperature = temp

    def set_model(self, model: str) -> bool:
        arg = model.lower()
        if arg in ("flash", "v4-flash", "f"):
            self.current_model = MODEL_V4_FLASH
            return True
        if arg in ("pro", "v4-pro", "p"):
            self.current_model = MODEL_V4_PRO
            return True
        if arg in ("vision", "v-flash-vision", "v4-vision"):
            self.current_model = MODEL_V4_VISION
            return True
        # 支持注册的自定义模型名 / 完整模型名（如 gpt-4o）
        registered = load_models()
        if arg in registered or arg in MODELS_AVAILABLE:
            self.current_model = arg
            return True
        return False

    def set_detail(self, detail: str) -> bool:
        """设置图片 detail 全局默认（low/auto/high/original）。"""
        if detail in ("low", "auto", "high", "original"):
            self.image_detail = detail
            return True
        return False

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
        """输入框下方状态栏文本：已导入文件数量 + 前 2 个文件名；无导入返回空串。"""
        names = [a.name for a in self.pending_images] + [
            f["name"] for f in self.imported_files
        ]
        if not names:
            return ""
        shown = "、".join(names[:2])
        extra = "…" if len(names) > 2 else ""
        return f"📎 已导入 {len(names)} 个文件：{shown}{extra} · /import clear 清除"

    def import_file_list(self) -> List[Dict[str, str]]:
        """完整导入文件列表（图片在前、文本/网页在后），供悬停弹窗展示。"""
        items = [{"kind": "image", "name": a.name} for a in self.pending_images]
        items += [{"kind": f["kind"], "name": f["name"]} for f in self.imported_files]
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
        """清空对话历史（/clear）。"""
        self._cleanup_temp_files()
        self.tree = ConversationTree(self.current_system)
        self.delete_session_file()

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
        「下一次输入」token 量采用与相邻数据直接对应的口径：
        - 普通节点：= 本节点 input_tokens + output_tokens（API 真实口径，
          即「上次完整输入 + 本节点输出」，随对话推进持续更新，不会卡住）；
        - 摘要节点（/compact 新建、本身无 API 用量）：= 实际发送的摘要
          上下文估算（estimate_tokens），与 /compact 报告 after_tokens 一致；
        用户新输入内容量小，忽略不计。预计价格按 DeepSeek 峰谷分时定价
        × 缓存命中率折算。
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
        comp = self.tree.compaction
        if comp and comp.get("boundary_id") and node.id == comp["boundary_id"]:
            # 摘要节点本身：按实际发送的摘要上下文实时估算（不随时间冻结）
            try:
                next_in = estimate_tokens(self.tree.get_messages_for_node(node))
            except Exception:
                next_in = 0
        else:
            next_in = node.input_tokens + node.output_tokens
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
        self, user_input: str, emit: EventSink
    ) -> Optional[ConversationNode]:
        """发送一条消息（可能触发多轮工具调用），完成后返回新节点。

        节点在流式输出前即创建并设为当前节点（UI 可立即“进入”新节点进行
        流式输出）；出错时回滚该节点。emit 会收到 node_created / stream /
        tool / status / done / error 事件。API 错误或无回答时返回 None。

        多模态：待发送图片先上传为 Files API file_id（请求体极小、序列化稳定、
        不破坏前缀缓存），上传失败回退 base64 内联；图片消息自动切换视觉模型。
        """
        if self.imported_content:
            user_input = self.imported_content + "\n\n" + user_input
            self.imported_content = None
            self.imported_files = []

        # 待发送图片：先上传为 file_id（成功后历史重放/后续请求体保持极小）
        if self.pending_images:
            emit(ControllerEvent.status(
                f"正在上传图片（{len(self.pending_images)} 张）…"
            ))
            self._upload_attachments(self.pending_images, emit)
        this_turn_images = list(self.pending_images)
        self.pending_images = []

        # 前置创建节点并设为当前节点：UI 立即进入新节点，流式输出归属该节点
        node = self._begin_node(user_input)
        node.user_images = this_turn_images
        emit(ControllerEvent.node_created(node))

        # 历史节点的本地图片若未上传过（file_id 缺失），补传一次（尽力而为）
        self._ensure_chain_uploads(node, emit)

        # 构建发送消息（历史链 + 本轮；图片构造为 OpenAI 兼容内容块）
        messages = self.tree.get_messages_for_node(node)

        # 图片消息必须使用视觉模型（flash/pro 自动切换，其他模型报错）
        if self._messages_contain_images(messages):
            guard_err = self._ensure_vision_model(emit)
            if guard_err:
                self.pending_images = this_turn_images + self.pending_images
                emit(ControllerEvent.error(guard_err))
                self._discard_node(node)
                return None

        # 请求体大小预检（仅内联 base64 回退路径会产生大请求体）
        inline_bytes = collect_inline_bytes(messages)
        if inline_bytes > VISION_REQUEST_MAX_BYTES:
            self.pending_images = this_turn_images + self.pending_images
            emit(ControllerEvent.error(
                f"图片 base64 总量超限（约 {inline_bytes // 1024 // 1024} MiB "
                f"> {VISION_REQUEST_MAX_BYTES // 1024 // 1024} MiB），"
                "请减少图片数量或压缩后重试"
            ))
            self._discard_node(node)
            return None

        final_answer: Optional[str] = None
        accumulated_reasoning = ""
        accumulated_in_tok = 0
        accumulated_out_tok = 0
        accumulated_cache_hit = 0
        accumulated_cache_miss = 0
        tool_messages: List[Dict] = []

        def _restore_pending() -> None:
            """发送失败时把本轮图片放回待发送队列（下次发送自动重试上传）。"""
            if this_turn_images:
                self.pending_images = this_turn_images + self.pending_images

        try:
            while True:
                sr = stream_response(
                    self.client,
                    messages,
                    self.current_model,
                    self.current_temperature,
                    user_input,
                    thinking_enabled=self.thinking_enabled,
                    reasoning_effort=self.reasoning_effort,
                    tools=self.llm_tools,
                    on_chunk=lambda c, r: emit(ControllerEvent.stream(c, r)),
                )
                if sr.error:
                    _restore_pending()
                    emit(ControllerEvent.error(sr.error))
                    self._discard_node(node)
                    return None

                reasoning = sr.reasoning or ""
                if reasoning:
                    accumulated_reasoning += reasoning
                accumulated_in_tok += sr.input_tokens
                accumulated_out_tok += sr.output_tokens
                accumulated_cache_hit += sr.cache_hit_tokens
                accumulated_cache_miss += sr.cache_miss_tokens

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
                    continue

                if sr.content is not None:
                    final_answer = sr.content
                    break

                _restore_pending()
                emit(ControllerEvent.error("回答生成失败，请重试"))
                self._discard_node(node)
                return None
        except Exception:
            # 未预期的异常：回滚前置创建的节点后重新抛出（由调用方显示错误）
            _restore_pending()
            self._discard_node(node)
            raise

        title = generate_conversation_title(self.client, user_input)
        node.assistant_msg = final_answer
        node.reasoning = accumulated_reasoning
        node.input_tokens = accumulated_in_tok
        node.output_tokens = accumulated_out_tok
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

    def _upload_attachments(
        self, attachments: List[ImageAttachment], emit: EventSink
    ) -> None:
        """尽力上传 path 附件为 Files API file_id；失败保留 file_id=None。

        发送时对无 file_id 的 path 附件回退 base64 内联（见 images.build_image_block）。
        """
        for i, att in enumerate(attachments, start=1):
            if att.file_id or att.is_url:
                continue
            emit(ControllerEvent.status(
                f"正在上传图片 {i}/{len(attachments)}：{att.name}…"
            ))
            try:
                att.file_id = upload_image(self.client, att.source)
            except FilesAPIError as e:
                emit(ControllerEvent.status(f"⚠️ {e}（将以内联 base64 发送）"))

    def _ensure_chain_uploads(self, node: ConversationNode, emit: EventSink) -> None:
        """历史节点的本地图片若缺 file_id 则补传（尽力而为，静默失败）。

        上传成功会使该节点消息序列化改变（base64 → file 块），因此失效其
        cached_messages 以便下次构建时使用 file_id。
        """
        for n in self._path_to_root(node):
            changed = False
            for att in n.user_images:
                if (
                    att.file_id is None
                    and not att.is_url
                    and os.path.exists(os.path.expanduser(att.source))
                ):
                    try:
                        att.file_id = upload_image(self.client, att.source)
                        changed = True
                    except FilesAPIError:
                        pass
            if changed:
                n.cached_messages = None

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
        """图片消息要求视觉模型：flash/pro 自动切换；其他模型返回错误信息。"""
        if self.current_model == MODEL_V4_VISION:
            return None
        if self.current_model in (MODEL_V4_FLASH, MODEL_V4_PRO):
            self.current_model = MODEL_V4_VISION
            emit(ControllerEvent.status(
                f"已自动切换到视觉模型 {MODEL_V4_VISION}（图片消息）"
            ))
            return None
        return (
            f"当前模型 {self.current_model} 不支持图片，请先 /set model vision 切换"
        )

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

    def _discard_node(self, node: ConversationNode) -> None:
        """出错时回滚前置创建的节点（连带重置当前节点）。"""
        if node.id in self.tree.nodes:
            self.tree.delete_node(node.id)

    # ---------------- 多模态：Files API 文件管理 ----------------

    def files_list(self) -> List[Dict]:
        """列出已上传的图片文件（/files list）。"""
        return list_files(self.client)

    def files_delete(self, file_id: str) -> bool:
        """删除一个已上传的图片文件（/files delete）。"""
        delete_file(self.client, file_id)
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
            return self._query_conversation_tree(args.get("root", ""), args.get("search", ""))
        if name == "read_conversation_nodes":
            return self._read_conversation_nodes(args.get("node_ids", ""))
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

    def _query_conversation_tree(self, root: str = "", search: str = "") -> str:
        if not self.tree or not self.tree.root:
            return "（暂无对话记录）"

        if search:
            results = []
            kw = search.lower()
            for nid, node in self.tree.nodes.items():
                if kw in (node.title or "").lower() or kw in (node.user_msg or "").lower():
                    results.append(f"{nid}: {node.title}")
            return "\n".join(results) if results else f"（未找到包含「{search}」的节点）"

        if root:
            nodes_in_tree = []
            root_id = next(
                (
                    nid
                    for nid in self.tree.nodes
                    if self.tree._node_letter_prefix(nid) == root
                    and self.tree.nodes[nid].parent_id == "main"
                ),
                None,
            )
            if not root_id:
                return f"（子对话树 {root} 不存在）"
            descendants = set()
            self.tree._collect_descendants(self.tree.nodes[root_id], descendants)
            for nid in sorted(descendants):
                node = self.tree.nodes[nid]
                depth = 0
                cur = node
                while cur.parent_id and cur.parent_id != "main":
                    depth += 1
                    cur = self.tree.nodes.get(cur.parent_id)
                nodes_in_tree.append(f"{'  ' * depth}{nid}: {node.title}")
            return "\n".join(nodes_in_tree)

        lines = [f"main: {self.tree.root.title}"]
        for child in self.tree.root.children:
            prefix = self.tree._get_subtree_root_prefix(child.id)
            if prefix:
                count = self.tree.count_subtree_nodes(prefix)
                suffix = self.tree.subtree_titles.get(prefix, child.title)
                lines.append(f"  {prefix}: {suffix}（{count}个节点）")
        return "\n".join(lines)

    def _read_conversation_nodes(self, node_ids: str) -> str:
        parts = []
        for nid in node_ids.split(","):
            nid = nid.strip()
            if not nid:
                continue
            node = self.tree.nodes.get(nid)
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
        if not self.confirm(f"即将{'覆盖' if exists else '写入'}文件", details):
            return "用户已取消操作"
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
        if not self.confirm("即将修改文件", details):
            return "用户已取消操作"
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
        mcp_call = (
            (lambda: self._mcp.call("execute_command", call_args))
            if self._mcp is not None and "execute_command" in self._mcp_tool_names
            else (lambda: "执行命令工具不可用")
        )

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
                    model=MODEL_V4_FLASH,
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
