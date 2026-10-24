"""内置工具表：定义、门控与本地分派。

只有这一处决定「模型能看到哪些工具」：
- 文件四态决定文件工具；
- 联网开关决定搜索/抓取；
- 树查询开关决定对话树工具；
- 只要本轮有附件，``read_attachment`` 就可用（与文件权限无关）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from mincli.config import Config
from mincli.tools import file_tools, web_tools

DEFAULT_ATTACHMENT_LIMIT = 200
MAX_ATTACHMENT_LIMIT = 2000


@dataclass
class ToolContext:
    """一次工具调用需要的运行期上下文。"""

    config: Config
    attachments: List[object] = field(default_factory=list)
    confirm: Optional[Callable[[str], bool]] = None
    recall: Optional[Callable[[str, dict], str]] = None


# ---------------- 附件续读 ----------------


def _attachment_text(att) -> str:
    """取附件正文：优先内存内容，其次落盘副本。"""
    text = getattr(att, "text", "") or ""
    if text:
        return text
    stored = getattr(att, "stored_path", "") or ""
    if stored:
        try:
            with open(stored, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""
    return ""


def _match_attachment(attachments: List[object], name: str):
    """按引用块名称/文件名匹配附件；重名时返回候选列表。"""
    wanted = (name or "").strip().strip("[]")
    if not wanted:
        return None
    hits = []
    for att in attachments:
        candidates = {
            str(getattr(att, "name", "")),
            str(getattr(att, "chip_label", "")).strip("[]"),
        }
        path = str(getattr(att, "path", ""))
        if path:
            parts = path.rstrip("/").split("/")
            candidates.add(parts[-1])
            if len(parts) >= 2:
                # 同名文件可用一级目录区分，如 papers/报告.docx
                candidates.add(f"{parts[-2]}/{parts[-1]}")
        if wanted in candidates:
            hits.append(att)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return hits
    return None


def read_attachment(
    name: str,
    offset: int = 1,
    limit: int = DEFAULT_ATTACHMENT_LIMIT,
    attachments: Optional[List[object]] = None,
) -> str:
    """按行读取附件正文（当前会话内的附件都可读）。"""
    from mincli.tools.file_tools import _as_int, _split_lines

    pool = list(attachments or [])
    if not pool:
        return "当前会话没有可读取的附件"
    wanted = str(name or "").strip()
    if not wanted:
        available = "、".join(str(getattr(a, "name", "")) for a in pool[:10])
        return f"请提供附件名称。当前可读：{available}"

    matched = _match_attachment(pool, wanted)
    if matched is None:
        available = "、".join(str(getattr(a, "name", "")) for a in pool[:10])
        return f"未找到附件：{wanted}。当前可读：{available}"
    if isinstance(matched, list):
        names = "、".join(str(getattr(a, "name", "")) for a in matched)
        return f"有多个同名附件：{names}，请用带目录的名字区分"

    att = matched
    text = _attachment_text(att)
    if not text:
        kind = str(getattr(att, "kind", ""))
        if kind == "image":
            return f"{getattr(att, 'name', wanted)} 是图片，以多模态方式发送，没有可续读的文本"
        return f"{getattr(att, 'name', wanted)} 没有可读取的正文"

    lines = _split_lines(text)
    total = len(lines)
    if total == 0:
        return f"附件：{getattr(att, 'name', wanted)}（空文件）"

    start = _as_int(offset, 1, 1, total + 1)
    count = _as_int(limit, DEFAULT_ATTACHMENT_LIMIT, 1, MAX_ATTACHMENT_LIMIT)
    if start > total:
        return f"起始行 {start} 超出范围：{getattr(att, 'name', wanted)} 共 {total} 行"
    selected = lines[start - 1 : start - 1 + count]
    end = start + len(selected) - 1
    header = (
        f"附件：{getattr(att, 'name', wanted)}"
        f"（共 {total} 行，以下为第 {start}-{end} 行）"
    )
    body = "\n".join(selected)
    tail = f"（用 offset={end + 1} 继续读取）" if end < total else "（已到附件末尾）"
    return f"{header}\n{body}\n{tail}"


# ---------------- 工具定义 ----------------


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


READ_ATTACHMENT = _tool(
    "read_attachment",
    "按行读取本轮附件（[DOCX ...]、[PDF ...]、[WEB ...] 等）的完整正文。"
    "消息里标注「已截断」时用它从提示的行继续读取，可分多次读。",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "附件名称，即引用块里的名字，如 '报告.docx' 或 'example.com/标题'",
            },
            "offset": {"type": "integer", "description": "起始行号，从 1 开始，默认 1"},
            "limit": {"type": "integer", "description": "读取行数，默认 200"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
)

READ_FILE = _tool(
    "read_file",
    "按行读取本地文件的一段，返回带行号的内容。支持文本、Markdown、CSV、PDF、DOCX。"
    "长文件用 offset / limit 分段读。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，支持 ~ 开头"},
            "offset": {"type": "integer", "description": "起始行号，从 1 开始，默认 1"},
            "limit": {"type": "integer", "description": "读取行数，默认 200，上限 2000"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)

WRITE_FILE = _tool(
    "write_file",
    "写入（覆盖）本地文件，父目录不存在时自动创建。整体重写才用它；"
    "改几行请用 edit_file。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，支持 ~ 开头"},
            "content": {"type": "string", "description": "完整的新内容"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)

EDIT_FILE = _tool(
    "edit_file",
    "把文件里唯一出现的一段原文替换成新内容。old_string 必须与文件完全一致"
    "（含空格与换行），且只能匹配一处；匹配多处时请补充前后文。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，支持 ~ 开头"},
            "old_string": {"type": "string", "description": "要被替换的原文"},
            "new_string": {"type": "string", "description": "替换后的内容"},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    },
)

LIST_DIRECTORY = _tool(
    "list_directory",
    "列出目录下的文件与子目录（不递归），目录在前，文件带大小。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径，默认当前目录"},
        },
        "additionalProperties": False,
    },
)

WEB_SEARCH = _tool(
    "web_search",
    "联网搜索网页，返回标题、链接与摘要。需要区分多来源、确认最新信息时使用；"
    "拿到链接后可再用 web_fetch 读全文。",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索词，1-128 字符，支持 site: / after: 等语法"},
            "count": {"type": "integer", "description": "返回条数，1-50，默认 5"},
            "need_content": {
                "type": "boolean",
                "description": "是否同时返回正文摘录，默认 false（摘要通常够用）",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

WEB_FETCH = _tool(
    "web_fetch",
    "抓取单个网页并返回 Markdown 正文。已知具体链接时直接用它，比搜索更省。",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "网页地址，http/https"},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
)

QUERY_TREE = _tool(
    "query_conversation_tree",
    "查询对话节点树的结构，返回文本。不传参数时返回所有对话树的编号索引 + 当前树的"
    "子树索引；传 root 只查某子树（如 root='a'）；传 search 按标题或内容关键词搜索节点；"
    "传 tree 指定对话树编号（不传=当前对话树）",
    {
        "type": "object",
        "properties": {
            "root": {"type": "string", "description": "可选。子对话树的字母前缀，如 'a'、'b'"},
            "search": {"type": "string", "description": "可选。关键词，搜索节点标题与用户消息"},
            "tree": {"type": "string", "description": "可选。对话树编号，如 '2'"},
        },
        "additionalProperties": False,
    },
)

READ_NODES = _tool(
    "read_conversation_nodes",
    "读取指定对话节点的完整内容（用户输入+思考过程+AI回答）。可一次读多个节点，"
    "用逗号分隔节点 ID；读别的对话树时需同时给 tree 编号",
    {
        "type": "object",
        "properties": {
            "node_ids": {"type": "string", "description": "节点 ID，多个用逗号分隔，如 'a1,a2'"},
            "tree": {"type": "string", "description": "可选。对话树编号，如 '2'"},
        },
        "required": ["node_ids"],
        "additionalProperties": False,
    },
)

_ALL_TOOLS = {
    "read_attachment": READ_ATTACHMENT,
    "read_file": READ_FILE,
    "write_file": WRITE_FILE,
    "edit_file": EDIT_FILE,
    "list_directory": LIST_DIRECTORY,
    "web_search": WEB_SEARCH,
    "web_fetch": WEB_FETCH,
    "query_conversation_tree": QUERY_TREE,
    "read_conversation_nodes": READ_NODES,
}

# 各类开关打开后额外下发的工具（顺序即下发给模型的顺序）
_READ_TOOLS = ("read_file", "list_directory")
_WRITE_TOOLS = ("write_file", "edit_file")
_WEB_TOOLS = ("web_search", "web_fetch")
_RECALL_TOOLS = ("query_conversation_tree", "read_conversation_nodes")


def available_tool_names(cfg: Config, has_attachments: bool) -> List[str]:
    """按当前能力状态决定下发给模型的工具名列表。"""
    names: List[str] = []
    if has_attachments:
        names.append("read_attachment")
    if cfg.files_read_enabled:
        names.extend(_READ_TOOLS)
    if cfg.files_write_enabled:
        names.extend(_WRITE_TOOLS)
    if cfg.web_enabled:
        names.extend(_WEB_TOOLS)
    if cfg.recall_enabled:
        names.extend(_RECALL_TOOLS)
    return names


def available_tools(cfg: Config, has_attachments: bool) -> List[dict]:
    """按当前能力状态下发工具定义（OpenAI tools 格式）。"""
    return [_ALL_TOOLS[name] for name in available_tool_names(cfg, has_attachments)]


# ---------------- 分派 ----------------


def dispatch_tool(name: str, args: dict, ctx: ToolContext) -> str:
    """在进程内执行一个工具，返回给模型的文本结果。"""
    args = args or {}
    try:
        if name == "read_attachment":
            return read_attachment(
                str(args.get("name", "")),
                args.get("offset", 1),
                args.get("limit", DEFAULT_ATTACHMENT_LIMIT),
                ctx.attachments,
            )
        if name == "read_file":
            return file_tools.read_file(
                str(args.get("path", "")),
                args.get("offset", 1),
                args.get("limit", file_tools.DEFAULT_READ_LIMIT),
            )
        if name == "list_directory":
            return file_tools.list_directory(str(args.get("path", ".") or "."))
        if name == "write_file":
            return file_tools.write_file(
                str(args.get("path", "")),
                str(args.get("content", "")),
                ctx.confirm,
            )
        if name == "edit_file":
            return file_tools.edit_file(
                str(args.get("path", "")),
                str(args.get("old_string", "")),
                str(args.get("new_string", "")),
                ctx.confirm,
            )
        if name == "web_search":
            return web_tools.web_search(
                str(args.get("query", "")),
                args.get("count", web_tools.DEFAULT_RESULTS),
                bool(args.get("need_content", False)),
            )
        if name == "web_fetch":
            return web_tools.web_fetch(str(args.get("url", "")))
        if name == "query_conversation_tree":
            if ctx.recall is None:
                return "对话树查询当前不可用"
            return ctx.recall("query", args)
        if name == "read_conversation_nodes":
            if ctx.recall is None:
                return "对话树查询当前不可用"
            return ctx.recall("read", args)
    except Exception as exc:  # 工具异常不应中断整轮对话
        return f"工具执行失败：{exc}"
    return f"未知工具：{name}"
