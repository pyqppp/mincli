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
    MODELS_AVAILABLE,
    COMPACT_DEFAULT_KEEP,
    COMPACT_MAX_TOKENS,
    COMPACT_SOURCE_MAX_CHARS,
    COMPACT_REASONING_MAX_CHARS,
    COMPACT_TOOL_RESULT_MAX_CHARS,
    EXEC_DEFAULT_TIMEOUT,
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

        self.tree = ConversationTree(default_system)

        self.imported_content: Optional[str] = None
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
        # 支持注册的自定义模型名 / 完整模型名（如 gpt-4o）
        registered = load_models()
        if arg in registered or arg in MODELS_AVAILABLE:
            self.current_model = arg
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
        """导入文件或网页为上下文，成功返回 None，失败返回错误信息。"""
        if re.match(r"^https?://", target):
            result = fetch_webpage(target)
        else:
            result = parse_file(target)
        if result and not result.startswith(self._IMPORT_FAIL_PREFIXES):
            self.imported_content = result
            return None
        return result or f"无法读取: {target}"

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

    def compact_history(
        self, keep: int = COMPACT_DEFAULT_KEEP, emit: Optional[EventSink] = None
    ) -> Optional[dict]:
        """把当前分支的早期对话压缩成详细摘要（/compact）。

        keep=保留最近 N 轮（含当前节点）的原始内容；0=全部压缩。
        摘要写入 self.tree.compaction，之后发送消息时自动用摘要替代
        boundary 及之前节点的原始消息。返回统计信息 dict；无可压缩
        内容或压缩失败时返回 None。

        统计口径：before/after 均用 estimate_tokens 估算压缩前后「发送给模型
        的完整消息列表」（与 usage_stats 的 next_input_tokens 同函数、同来源），
        因此压缩后状态条的「下次输入」会自动等于 after_tokens。
        """
        if self.tree is None or self.tree.current_node is None:
            return None
        path = self._path_to_root(self.tree.current_node)
        n = len(path)
        if keep is None or keep < 0:
            keep = COMPACT_DEFAULT_KEEP
        if n - keep < 1:
            return None  # 对话太短，没有可压缩的轮次
        boundary = path[n - keep - 1]   # 最后一个被压缩进摘要的节点
        to_compress = path[: n - keep]  # 被压缩进摘要的节点（含 boundary）

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
                f"正在压缩上下文：{len(to_compress)} 轮 → 详细摘要（保留最近 {keep} 轮原文）…"
            ))

        summary = self._call_summarize(source)
        if not summary:
            return None

        before_msgs = self.tree.get_messages_for_node(self.tree.current_node)
        self.tree.compaction = {
            "summary": summary,
            "boundary_id": boundary.id,
            "next_input_tokens": 0,  # 占位，下面填充
        }
        after_msgs = self.tree.get_messages_for_node(self.tree.current_node)
        before_tok = estimate_tokens(before_msgs)
        after_tok = estimate_tokens(after_msgs)
        # 让状态条「下次输入」在压缩后直接采用压缩报告口径（与 after_tokens 严格一致）
        self.tree.compaction["next_input_tokens"] = after_tok

        return {
            "summary": summary,
            "boundary_id": boundary.id,
            "nodes_compressed": len(to_compress),
            "nodes_kept": keep,
            "summary_chars": len(summary),
            "before_tokens": before_tok,
            "after_tokens": after_tok,
            "saved_tokens": max(0, before_tok - after_tok),
        }

    def clear_compaction(self) -> bool:
        """清除上下文压缩摘要（/compact off），恢复完整原始消息。"""
        return self.tree.clear_compaction() if self.tree else False

    # ---------------- 实时用量统计（输入栏状态条） ----------------

    def usage_stats(self) -> dict:
        """输入栏状态条数据（纯本地计算，不联网）。

        缓存命中率取当前节点累计的 usage.prompt_cache_hit/miss_tokens。
        「下一次输入」token 量采用与相邻数据直接对应的口径：
        - 未压缩：= 本节点 input_tokens + output_tokens（API 真实口径，
          即「上次完整输入 + 本节点输出」，与对话结束显示的输入/输出严格对应）；
        - 已压缩：= 压缩报告 after_tokens（压缩时存储），与 /compact
          报告的数字严格一致，直观体现压缩节省；
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
        if comp and comp.get("next_input_tokens"):
            next_in = comp["next_input_tokens"]
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
            parts.append(f"用户: {node.user_msg}")
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
        """
        if self.imported_content:
            user_input = self.imported_content + "\n\n" + user_input
            self.imported_content = None

        if self.tree.current_node is None:
            messages: List[Dict] = [
                {"role": "system", "content": self.current_system},
                {"role": "user", "content": user_input},
            ]
        else:
            messages = self.tree.get_messages_for_node(self.tree.current_node)
            messages.append({"role": "user", "content": user_input})

        # 前置创建节点并设为当前节点：UI 立即进入新节点，流式输出归属该节点
        node = self._begin_node(user_input)
        emit(ControllerEvent.node_created(node))

        final_answer: Optional[str] = None
        accumulated_reasoning = ""
        accumulated_in_tok = 0
        accumulated_out_tok = 0
        accumulated_cache_hit = 0
        accumulated_cache_miss = 0
        tool_messages: List[Dict] = []

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

                emit(ControllerEvent.error("回答生成失败，请重试"))
                self._discard_node(node)
                return None
        except Exception:
            # 未预期的异常：回滚前置创建的节点后重新抛出（由调用方显示错误）
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
        self.tree.current_node = node
        self._auto_title_subtree(node, emit)
        emit(ControllerEvent.done(node))
        return node

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
