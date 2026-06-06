#!/usr/bin/env python3
"""
mincli - 树状对话 AI 助手（支持推理过程）

特性：
    - 流式输出，实时刷新 Markdown 渲染。
    - 树状对话：全局唯一节点ID，分支自由切换。
    - 支持 /set 命令动态修改系统提示词、温度、模型及思考开关。
    - 支持 /save 保存节点对话到 Markdown 文件。
    - 交互过程中可随时切换思考/非思考模型，并完整保留推理内容。

使用方法：
    python main.py chat             # 进入树状对话模式
    python main.py info             # 显示配置信息
"""

import os
import re
import csv
import sys
import datetime
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import json
import subprocess
import tempfile

import requests
import tiktoken
import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.theme import Theme
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree as RichTree
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
import trafilatura

# ---------- 初始化 ----------
load_dotenv()                                   # 当前目录 .env（最低优先级）
load_dotenv(os.path.expanduser("~/.mincli/.env"))  # 用户配置（中优先级）
# 环境变量已在 os.environ 中，优先级最高

MD_THEME = Theme({
    "markdown.h1": "cyan",
    "markdown.h2": "cyan",
    "markdown.h3": "cyan",
    "markdown.h4": "cyan",
    "markdown.h5": "cyan",
    "markdown.h6": "cyan",
    "markdown.block_quote": "bright_black",
})
console = Console(stderr=True, highlight=False, theme=MD_THEME)
app = typer.Typer(help="mincli - 树状对话 AI 助手")

# 模型常量（V4）
MODEL_V4_FLASH = "deepseek-v4-flash"   # 轻量快速
MODEL_V4_PRO = "deepseek-v4-pro"       # 旗舰性能
DEFAULT_MODEL = MODEL_V4_FLASH

# 保存路径配置
SAVE_BASE_DIR = os.path.expanduser(
    os.getenv("MINCLI_SAVE_PATH", "~/Documents/mincli_Conversations")
)

# 显示/截断常量
TITLE_MAX_TOKENS = 30
TITLE_MAX_LENGTH = 30
DISPLAY_BODY_PADDING = 8
DISPLAY_BODY_MIN = 30
PREVIEW_USER_MSG_LEN = 100
PREVIEW_ASSISTANT_MSG_LEN = 200
WEBPAGE_MAX_LENGTH = 5000
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
BOCHA_API_BASE = "https://api.bocha.cn/v1/web-search"

# 工具定义（Tool Calls）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件的内容，支持 txt、md、py、bat、sh、csv、pdf、docx 格式，返回文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "文件路径，支持绝对路径和 ~ 开头的路径",
                    }
                },
                "required": ["filepath"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "抓取指定 URL 的网页内容并提取正文，返回网页标题和文本内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "网页 URL，如 https://example.com",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录的内容，可选择是否包含隐藏文件（以 . 开头的文件），默认不包含隐藏文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "目录路径，支持绝对路径和 ~ 开头的路径",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "是否包含隐藏文件，默认 false",
                    },
                },
                "required": ["directory"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件。如果文件不存在则创建新文件，存在则覆盖原内容。写入前会请求用户确认",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "文件路径，支持绝对路径和 ~ 开头的路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容",
                    },
                },
                "required": ["filepath", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "在文件中搜索 old_string 并替换为 new_string（仅替换第一个匹配项）。old_string 必须与文件内容精确匹配（包括空格和换行）。操作前会请求用户确认",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "文件路径，支持绝对路径和 ~ 开头的路径",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要被替换的精确原文（区分大小写、包含空格和换行）",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的新内容",
                    },
                },
                "required": ["filepath", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网信息。注意：该工具调用需要用户事先通过 /search 命令授权",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "freshness": {
                        "type": "string",
                        "description": "时间范围: noLimit(不限)/oneDay(一天内)/oneWeek(一周内)/oneMonth(一月内)/oneYear(一年内)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "返回结果条数(1-50), 默认10",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": f"在用户电脑上执行 shell 命令。当前操作系统: {sys.platform}（{'Windows' if sys.platform == 'win32' else 'macOS/Linux'}）。每个命令在执行前会经过 AI 安全审核和用户确认。默认工作目录为用户家目录。注意：若预计输出很长，请在命令中关闭输出（如 {'追加 >nul 2>&1' if sys.platform == 'win32' else '追加 >/dev/null 2>&1'}）以节省 token。必须设置 deadline（timeout 参数），超时后命令将被强制终止，但会返回已产生的部分输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "执行截止时间（秒）。必须设置，超时后命令会被强制终止，已产生的部分输出仍会返回",
                    },
                },
                "required": ["command", "timeout"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------- 辅助函数 ----------
def clear_screen() -> None:
    """清空终端屏幕。在 iTerm2 中同时重置滚动缓冲区。"""
    # 检测是否在 iTerm2 中（环境变量 TERM_PROGRAM=iTerm.app）
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        sys.stdout.write("\033]1337;ClearScrollback\007")
        sys.stdout.flush()
    else:
        # 跨平台通用清屏
        os.system('cls' if os.name == 'nt' else 'clear')


def clip_for_terminal(text: str, max_lines: int) -> str:
    """将文本裁剪到 max_lines 行，超出部分从顶部截断，保留最后 max_lines 行。"""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "…（上略）\n" + "\n".join(lines[-(max_lines - 1):])


def get_balance(client: OpenAI) -> Optional[Dict]:
    """查询 DeepSeek 账户余额，返回 balance_infos 列表或 None。"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        resp = requests.get("https://api.deepseek.com/user/balance", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("balance_infos")
    except Exception as e:
        return None


def format_balance(balance_infos: Optional[List[Dict]]) -> str:
    """格式化余额信息为可读字符串。"""
    if not balance_infos:
        return ""
    parts = []
    for info in balance_infos:
        currency = info.get("currency", "")
        total = info.get("total_balance", "0.00")
        granted = info.get("granted_balance", "0.00")
        topped_up = info.get("topped_up_balance", "0.00")
        parts.append(f"{currency} ¥{total}（赠金:¥{granted} 充值:¥{topped_up}）")
    return " | ".join(parts)


def estimate_tokens(messages: list) -> int:
    """估算消息列表的 token 数量（备用方案，用于流式响应未返回 usage 时）。"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return 0
    tokens = 0
    for msg in messages:
        tokens += 3
        for key, value in msg.items():
            if isinstance(value, str):
                tokens += len(encoding.encode(value))
            if key == "name":
                tokens += 1
    tokens += 3
    return tokens


def generate_conversation_title(client: OpenAI, user_msg: str, assistant_msg: str) -> str:
    """调用 DeepSeek 生成简短的对话标题（用于保存文件和树节点标识）。"""
    try:
        prompt = (
            "请用不超过30字的汉字为以下内容写一个标题，只输出标题，不要有其他解释，"
            "不要包含标点符号和特殊字符。\n\n"
            f"用户：{user_msg}\n助手：{assistant_msg}"
        )
        resp = client.chat.completions.create(
            model=MODEL_V4_FLASH,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=TITLE_MAX_TOKENS,
            extra_body={"thinking": {"type": "disabled"}}   # 显式关闭思考
        )
        title = resp.choices[0].message.content.strip()
        # 移除文件系统不安全的字符
        title = re.sub(r'[\\/*?:"<>|]', '', title)
        title = title.replace(' ', '_')
        if len(title) > TITLE_MAX_LENGTH:
            title = title[:TITLE_MAX_LENGTH]
        return title if title else f"对话_{datetime.datetime.now().strftime('%H%M%S')}"
    except Exception as e:
        console.print(f"[red]⚠️ 生成标题失败: {e}[/red]")
        return f"对话_{datetime.datetime.now().strftime('%H%M%S')}"


def save_conversation_to_file(
    content: str,
    title: str,
    extra_prefix: str = "",
    token_stats: Optional[Dict[str, int]] = None,
) -> str:
    """通用保存函数：将对话内容写入 Markdown 文件。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{extra_prefix}_" if extra_prefix else ""
    filename = f"{prefix}{title}_{timestamp}.md"
    
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_BASE_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        if token_stats:
            f.write(f"\n## Token 统计\n\n")
            f.write(f"- 输入 tokens: {token_stats.get('input_tokens', 0)}\n")
            f.write(f"- 输出 tokens: {token_stats.get('output_tokens', 0)}\n")
    
    return filepath


# ---------- 流式输出与 API 交互 ----------
@dataclass
class StreamResult:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: Optional[List[Dict]] = None


def stream_response(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    user_question: str,
    thinking_enabled: bool = False,
    reasoning_effort: str = "high",
    tools: Optional[List[Dict]] = None,
    silent: bool = False,
) -> StreamResult:
    estimated_input = estimate_tokens(messages)
    full_content = ""
    reasoning_text = ""
    usage_input = 0
    usage_output = 0
    accumulated_tool_calls: Dict[int, Dict] = {}

    def _process_chunk(chunk):
        nonlocal full_content, reasoning_text, usage_input, usage_output
        if hasattr(chunk, "usage") and chunk.usage:
            usage_input = chunk.usage.prompt_tokens
            usage_output = chunk.usage.completion_tokens
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
        if delta.content:
            full_content += delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                if tc.id:
                    accumulated_tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

    try:
        extra_body = {}
        if thinking_enabled:
            extra_body["thinking"] = {"type": "enabled"}
            extra_body["reasoning_effort"] = reasoning_effort
        else:
            extra_body["thinking"] = {"type": "disabled"}

        kwargs = dict(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            extra_body=extra_body,
        )
        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)

        if silent:
            for chunk in response:
                _process_chunk(chunk)
        else:
            with Live(auto_refresh=False, console=console, screen=True) as live:
                header = f"**你:**\n{user_question}\n\n"
                initial_display = header + "**DeepSeek:** "
                live.update(Markdown(initial_display), refresh=True)

                for chunk in response:
                    _process_chunk(chunk)

                    term_lines = shutil.get_terminal_size().lines
                    max_body = max(DISPLAY_BODY_MIN, term_lines - DISPLAY_BODY_PADDING)
                    display = header
                    if reasoning_text:
                        display += "[dim]**DeepSeek 思考过程:**\n "
                        display += clip_for_terminal(reasoning_text, max_body // 2) + "[/dim]\n\n"
                    display += f"**DeepSeek:** {clip_for_terminal(full_content, max_body // 2)}"
                    live.update(Markdown(display), refresh=True)

                term_lines = shutil.get_terminal_size().lines
                max_body = max(DISPLAY_BODY_MIN, term_lines - DISPLAY_BODY_PADDING)
                final_display = header
                if reasoning_text:
                    final_display += "[dim]**DeepSeek 思考过程:**\n "
                    final_display += clip_for_terminal(reasoning_text, max_body // 2) + "[/dim]\n\n"
                final_display += f"**DeepSeek:** {clip_for_terminal(full_content, max_body // 2)}"
                live.update(Markdown(final_display), refresh=True)

        if accumulated_tool_calls:
            return StreamResult(tool_calls=list(accumulated_tool_calls.values()), reasoning=reasoning_text,
                                input_tokens=usage_input, output_tokens=usage_output)

        if usage_input == 0 and usage_output == 0:
            input_tokens = estimated_input
            output_tokens = estimate_tokens([{"role": "assistant", "content": full_content}])
        else:
            input_tokens = usage_input
            output_tokens = usage_output

        return StreamResult(content=full_content, reasoning=reasoning_text,
                            input_tokens=input_tokens, output_tokens=output_tokens)

    except Exception as e:
        console.print(f"[red]API 调用失败: {e}[/red]")
        return StreamResult()


# ---------- 树状对话数据结构 ----------
@dataclass
class ConversationNode:
    """树状对话中的一个节点。"""
    id: str                           # 全局唯一的简短层级ID
    parent_id: Optional[str] = None
    user_msg: str = ""
    assistant_msg: str = ""
    reasoning: str = ""               # 推理内容
    title: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    children: List['ConversationNode'] = field(default_factory=list)
    cached_messages: Optional[List[Dict]] = None
    tool_messages: List[Dict] = field(default_factory=list)

    def get_messages(self, tree: 'ConversationTree') -> List[Dict]:
        """获取从根节点到当前节点的完整消息上下文（用于 API 调用）。"""
        if self.cached_messages is not None:
            return self.cached_messages
        
        msgs = []
        if self.parent_id:
            parent = tree.nodes.get(self.parent_id)
            if parent:
                msgs = parent.get_messages(tree).copy()
        
        msgs.append({"role": "user", "content": self.user_msg})
        for tm in self.tool_messages:
            msgs.append(tm)
        if self.assistant_msg:
            assistant_msg = {"role": "assistant", "content": self.assistant_msg}
            if self.reasoning:
                assistant_msg["reasoning_content"] = self.reasoning
            msgs.append(assistant_msg)
        
        self.cached_messages = msgs
        return msgs
    
    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为可 JSON 序列化的字典（忽略缓存和循环引用）。"""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "user_msg": self.user_msg,
            "assistant_msg": self.assistant_msg,
            "reasoning": self.reasoning,
            "title": self.title,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_messages": self.tool_messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationNode':
        """从字典恢复节点（不包含 children 和 cached_messages）。"""
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            user_msg=data["user_msg"],
            assistant_msg=data["assistant_msg"],
            reasoning=data.get("reasoning", ""),
            title=data["title"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            tool_messages=data.get("tool_messages", []),
        )


class ConversationTree:
    """管理树状对话结构，提供节点增删改查及 ID 生成逻辑。"""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.nodes: Dict[str, ConversationNode] = {}
        self.root: Optional[ConversationNode] = None
        self.current_node: Optional[ConversationNode] = None

    def _generate_child_id(self, parent: ConversationNode) -> str:
        """
        生成 ID 规则：
        - 父节点无子节点 → 继承父字母，数字 +1（主线发展）。
        - 父节点已有子节点 → 从 'a' 开始寻找全局未使用的字母，分配 '字母1' 作为新分支。
        """
        used_ids = set(self.nodes.keys())
    
        # 1. 主线发展：父节点无子节点
        if not parent.children:
            # 提取父 ID 的字母部分（忽略数字）
            match = re.match(r'^([a-z]+)(\d+)$', parent.id)
            if match:
                prefix = match.group(1)
                num = int(match.group(2)) + 1
                candidate = f"{prefix}{num}"
                while candidate in used_ids:
                    num += 1
                    candidate = f"{prefix}{num}"
                return candidate
            else:
                # 若父 ID 格式非字母+数字（如 main），则以 a1 作为主线起始
                candidate = "a1"
                while candidate in used_ids:
                    num = int(re.search(r'\d+$', candidate).group()) + 1
                    candidate = f"a{num}"
                return candidate
    
        # 2. 分支发展：父节点已有子节点 → 分配新字母
        # 收集全局已使用的字母
        used_letters = set()
        for nid in used_ids:
            match = re.match(r'^([a-z]+)\d+$', nid)
            if match:
                used_letters.add(match.group(1))
    
        # 从 'a' 开始寻找第一个未使用的字母
        for letter in range(ord('a'), ord('z') + 1):
            l = chr(letter)
            if l not in used_letters:
                candidate = f"{l}1"
                if candidate not in used_ids:
                    return candidate
                # 如果字母+1已被占用（极端情况），则尝试字母+2...
                num = 2
                while f"{l}{num}" in used_ids:
                    num += 1
                return f"{l}{num}"
    
        # 字母表耗尽（极其罕见）的兜底
        return f"z_{datetime.datetime.now().strftime('%H%M%S')}"

    def create_root(self, user_msg: str, assistant_msg: str, reasoning: str,
                    title: str, input_tokens: int, output_tokens: int) -> ConversationNode:
        """创建根节点，ID 固定为 'main'。"""
        node = ConversationNode(
            id="main",
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            reasoning=reasoning,
            title=title,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        self.nodes[node.id] = node
        self.root = node
        self.current_node = node
        return node
    
    def add_child(self, parent: ConversationNode, user_msg: str, assistant_msg: str,
                  reasoning: str, title: str, input_tokens: int, output_tokens: int) -> ConversationNode:
        """为指定父节点添加子节点。"""
        child_id = self._generate_child_id(parent)
        node = ConversationNode(
            id=child_id,
            parent_id=parent.id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            reasoning=reasoning,
            title=title,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        self.nodes[node.id] = node
        parent.children.append(node)
        return node

    def get_messages_for_node(self, node: ConversationNode) -> List[Dict]:
        """获取指定节点所需的完整 API 消息列表（含 system 提示）。"""
        msgs = node.get_messages(self)
        return [{"role": "system", "content": self.system_prompt}] + msgs

    def switch_to_node(self, node_id: str) -> bool:
        """切换当前节点至指定 ID，成功返回 True。"""
        if node_id in self.nodes:
            self.current_node = self.nodes[node_id]
            return True
        return False

    def delete_node(self, node_id: str) -> bool:
        """删除指定节点及其所有子节点。返回 True 表示成功。"""
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        # 收集所有要删除的节点 ID（包括自身和所有后代）
        to_delete = set()
        self._collect_descendants(node, to_delete)

        # 从父节点的 children 列表中移除该节点
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                parent.children = [c for c in parent.children if c.id != node_id]

        # 从 nodes 字典中删除
        for nid in to_delete:
            del self.nodes[nid]

        # 如果删除的节点是当前节点，将 current_node 切换到其父节点或根节点
        if self.current_node and self.current_node.id in to_delete:
            if node.parent_id and node.parent_id in self.nodes:
                self.current_node = self.nodes[node.parent_id]
            else:
                self.current_node = self.root

        return True

    def _collect_descendants(self, node: ConversationNode, result: set):
        """递归收集节点及其所有后代的 ID。"""
        result.add(node.id)
        for child in node.children:
            self._collect_descendants(child, result)

    def render_tree(self, highlight_id: Optional[str] = None) -> RichTree:
        """使用 Rich 库渲染树状图。"""
        if not self.root:
            return RichTree("[空树]")
        root_tree = RichTree(f"📁 {self.root.id}: {self.root.title}")
        self._add_node_to_rich_tree(root_tree, self.root, highlight_id)
        return root_tree

    def _add_node_to_rich_tree(self, rich_node: RichTree, node: ConversationNode,
                               highlight_id: Optional[str]):
        for child in node.children:
            label = f"{child.id}: {child.title}"
            if child.id == highlight_id:
                label = f"[bold cyan]➤ {label}[/bold cyan]"
            child_tree = rich_node.add(label)
            self._add_node_to_rich_tree(child_tree, child, highlight_id)

    def get_branch_total_tokens(self, node_id: str) -> Tuple[int, int]:
        total_in = 0
        total_out = 0
        node = self.nodes.get(node_id)
        while node:
            total_in += node.input_tokens
            total_out += node.output_tokens
            node = self.nodes.get(node.parent_id) if node.parent_id else None
        return total_in, total_out
    
    def to_dict(self) -> Dict[str, Any]:
        """将整棵树转换为可 JSON 序列化的字典。"""
        nodes_data = {}
        for nid, node in self.nodes.items():
            nodes_data[nid] = node.to_dict()
        return {
            "system_prompt": self.system_prompt,
            "nodes": nodes_data,
            "root_id": self.root.id if self.root else None,
            "current_node_id": self.current_node.id if self.current_node else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTree':
        """从字典恢复树结构，重建节点及其父子关系。"""
        tree = cls(system_prompt=data["system_prompt"])
        # 第一遍：创建所有节点对象
        for nid, node_data in data["nodes"].items():
            node = ConversationNode.from_dict(node_data)
            tree.nodes[nid] = node
        # 第二遍：建立 children 关系
        for nid, node in tree.nodes.items():
            if node.parent_id:
                parent = tree.nodes.get(node.parent_id)
                if parent:
                    parent.children.append(node)
        # 设置根节点和当前节点
        root_id = data.get("root_id")
        if root_id:
            tree.root = tree.nodes.get(root_id)
        current_id = data.get("current_node_id")
        if current_id:
            tree.current_node = tree.nodes.get(current_id)
        return tree


# ---------- 交互会话管理器 ----------
class InteractiveSession:
    """管理整个交互会话的状态与行为，包括线性模式与树状模式。"""
    SAVE_FILE = os.path.expanduser("~/.mincli_session.json")

    def __init__(
        self,
        client: OpenAI,
        default_system: str,
        default_temperature: float,
        default_model: str = DEFAULT_MODEL,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
    ):
        self.client = client
        self.current_system = default_system
        self.current_temperature = default_temperature
        self.current_model = default_model
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort

        self.tree = ConversationTree(default_system)

        self.history_file = os.path.expanduser("~/.mincli_history")
        self.session = PromptSession(history=FileHistory(self.history_file))

        self.search_quota: int = 0
        self.imported_content: Optional[str] = None
        self.temp_dir = tempfile.mkdtemp(prefix="mincli_")
        self.temp_files: Dict[str, str] = {}
        self._load_session()

    def _save_session(self) -> None:
        filepath = self.SAVE_FILE
        try:
            data = {
                "system_prompt": self.current_system,
                "temperature": self.current_temperature,
                "model": self.current_model,
                "thinking_enabled": self.thinking_enabled,
                "reasoning_effort": self.reasoning_effort,
                "tree": self.tree.to_dict(),
                "imported_content": self.imported_content,
                "search_quota": self.search_quota,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[red]⚠️ 会话保存失败: {e}[/red]")

    def _load_session(self) -> bool:
        if not os.path.exists(self.SAVE_FILE):
            return False
        try:
            with open(self.SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]⚠️ 会话文件损坏，已忽略: {e}[/red]")
            try:
                os.remove(self.SAVE_FILE)
            except:
                pass
            return False

        self.current_system = data.get("system_prompt", self.current_system)
        self.current_temperature = data.get("temperature", self.current_temperature)
        self.current_model = data.get("model", self.current_model)
        self.thinking_enabled = data.get("thinking_enabled", False)
        self.reasoning_effort = data.get("reasoning_effort", "high")

        tree_data = data.get("tree")
        if tree_data:
            self.tree = ConversationTree.from_dict(tree_data)
        else:
            self.tree = ConversationTree(self.current_system)

        self.imported_content = data.get("imported_content")
        self.search_quota = data.get("search_quota", 0)
        console.print("[dim]📂 已加载上次会话记录[/dim]")
        return True

    def _delete_session_file(self) -> None:
        try:
            if os.path.exists(self.SAVE_FILE):
                os.remove(self.SAVE_FILE)
        except Exception:
            pass
    


    def _render_conversation(self, user_msg: str, assistant_msg: str, reasoning: str,
                             title: str, input_tokens: int, output_tokens: int) -> None:
        console.print(Panel(title, style="bold cyan"))
        # 先显示用户问题
        console.print(Markdown(f"**你:** {user_msg}"))
        # 再显示思考过程（如果有）
        if reasoning:
            console.print(Markdown("\n**DeepSeek 思考过程:**"))
            console.print(f"[dim]{reasoning}[/dim]")
        # 最后显示正式回答
        console.print(Markdown(f"**DeepSeek:** {assistant_msg}"))
        # 查询并显示余额
        balance_infos = get_balance(self.client)
        balance_str = format_balance(balance_infos)
        console.print(
            f"[dim]📊 输入: {input_tokens} tokens | 输出: {output_tokens} tokens"
            f"{' | 💰 ' + balance_str if balance_str else ''}[/dim]"
        )

    def _display_tree_node(self, node: ConversationNode, branch_total: Optional[Tuple[int, int]] = None) -> None:
        clear_screen()
        self._render_conversation(node.user_msg, node.assistant_msg, node.reasoning,
                                  f"节点 {node.id}: {node.title}",
                                  node.input_tokens, node.output_tokens)
        if branch_total is not None:
            bt_in, bt_out = branch_total
            balance_infos = get_balance(self.client)
            balance_str = format_balance(balance_infos)
            console.print(
                f"[dim]📊 本分支总消耗: 输入 {bt_in} tokens | 输出 {bt_out} tokens"
                f"{' | 💰 ' + balance_str if balance_str else ''}[/dim]"
            )
        console.print("[bold]对话树：[/bold]")
        console.print(self.tree.render_tree(node.id))
        console.print(f"[dim]当前节点: {node.id} ({node.title})[/dim]")

    def _save_tree_node(self, node_id: str) -> None:
        node = self.tree.nodes.get(node_id) if self.tree else None
        if not node:
            console.print("[red]节点不存在[/red]")
            return
        
        content = (
            f"# 节点 {node.id}: {node.title}\n\n"
            f"**时间：** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## 用户问题\n\n{node.user_msg}\n\n"
        )
        if node.reasoning:
            content += f"## DeepSeek 思考过程\n\n{node.reasoning}\n\n"
        content += f"## DeepSeek 回答\n\n{node.assistant_msg}\n\n"
        token_stats = {
            'input_tokens': node.input_tokens,
            'output_tokens': node.output_tokens
        }
        filepath = save_conversation_to_file(content, node.title, node.id, token_stats)
        console.print(f"[green]✅ 节点已保存到 {filepath}[/green]")

    def handle_command(self, cmd: str) -> bool:
        cmd_stripped = cmd.strip()
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ["/exit", "/quit", "/q", "/e"]:
            console.print("再见！👋")
            return True

        if cmd_lower in ["/clear", "/c"]:
            self._clear_history()
            return True

        if cmd_lower in ["/show"]:
            self._show_current_node()
            return True

        if cmd_lower in ["/help", "/h"]:
            self._show_help()
            return True

        if cmd_lower.startswith("/set"):
            self._handle_set_command(cmd)
            return True

        if self.tree and self._handle_tree_command(cmd):
            return True

        if cmd_lower.startswith("/search"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                self.search_quota = int(parts[1])
                console.print(f"[green]✅ 已授权 {self.search_quota} 次搜索[/green]")
            else:
                console.print("[yellow]用法: /search <正整数>[/yellow]")
            return True

        if cmd_lower.startswith("/fetch"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]用法: /fetch <URL>[/yellow]")
            else:
                console.print(f"[dim]正在抓取 {parts[1].strip()}…[/dim]")
                result = self._fetch_webpage(parts[1].strip())
                if result:
                    self.imported_content = result
                    console.print("[green]✅ 网页内容已导入，将在下一次提问时自动附加。[/green]")
            return True

        if cmd_lower.startswith("/imp"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]用法: /imp <文件路径>[/yellow]")
            else:
                result = self._parse_file(parts[1].strip())
                if result:
                    self.imported_content = result
                    console.print("[green]✅ 文件内容已导入，将在下一次提问时自动附加。[/green]")
            return True

        if cmd_stripped.startswith("/"):
            console.print(f"[yellow]未知命令: {cmd_stripped}。输入 /help 查看可用命令。[/yellow]")
            return True

        return False

    def _clear_history(self) -> None:
        self._cleanup_temp_files()
        self.tree = ConversationTree(self.current_system)
        self._delete_session_file()
        clear_screen()
        console.print("[dim]对话历史已清除[/dim]")
        console.print("[dim]等待下一个问题...[/dim]\n")

    def _show_current_node(self) -> None:
        node = self.tree.current_node
        if not node or not node.assistant_msg:
            console.print("[yellow]当前节点没有可打开的回答内容[/yellow]")
            return
        nid = node.id
        if nid in self.temp_files:
            filepath = self.temp_files[nid]
        else:
            filepath = os.path.join(self.temp_dir, f"mincli_{nid}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(node.assistant_msg)
            self.temp_files[nid] = filepath
        try:
            subprocess.run(["open", filepath], check=True)
            console.print(f"[dim]已打开节点 {nid} 的回答[/dim]")
        except Exception as e:
            console.print(f"[red]打开文件失败: {e}[/red]")

    def _cleanup_temp_files(self, keep_ids: Optional[set] = None) -> None:
        for nid, filepath in list(self.temp_files.items()):
            if keep_ids is None or nid not in keep_ids:
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                del self.temp_files[nid]

    def _handle_set_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=2)
        if len(parts) < 2:
            console.print("[yellow]用法: /set system <提示词>  /set temp <值>  /set model <flash|pro>  /set thinking <on|off>  /set effort <high|max>  /set show[/yellow]")
            return
        
        sub = parts[1]
        if sub == "system" and len(parts) == 3:
            self.current_system = parts[2]
            self.tree.system_prompt = self.current_system
            console.print("[green]系统提示词已更新[/green]")
        
        elif sub == "temp" and len(parts) == 3:
            try:
                temp = float(parts[2])
                if temp < TEMPERATURE_MIN or temp > TEMPERATURE_MAX:
                    console.print(f"[yellow]温度建议在 {TEMPERATURE_MIN}~{TEMPERATURE_MAX} 之间[/yellow]")
                self.current_temperature = temp
                console.print(f"[green]温度已设置为 {self.current_temperature}[/green]")
            except ValueError:
                console.print("[red]温度须为数字[/red]")
        
        elif sub == "model" and len(parts) == 3:
            arg = parts[2].lower()
            if arg in ["flash", "v4-flash", "f"]:
                self.current_model = MODEL_V4_FLASH
                console.print(f"[green]模型已切换为: {MODEL_V4_FLASH}[/green]")
            elif arg in ["pro", "v4-pro", "p"]:
                self.current_model = MODEL_V4_PRO
                console.print(f"[green]模型已切换为: {MODEL_V4_PRO}[/green]")
            else:
                console.print("[yellow]用法: /set model <flash|pro>[/yellow]")
        
        elif sub == "thinking" and len(parts) == 3:
            arg = parts[2].lower()
            if arg in ["on", "1", "true"]:
                self.thinking_enabled = True
                console.print(f"[green]思考模式已开启（effort: {self.reasoning_effort}）[/green]")
            elif arg in ["off", "0", "false"]:
                self.thinking_enabled = False
                console.print("[green]思考模式已关闭[/green]")
            else:
                console.print("[yellow]用法: /set thinking <on|off>[/yellow]")
        
        elif sub == "effort" and len(parts) == 3:
            arg = parts[2].lower()
            if arg in ["high", "max"]:
                self.reasoning_effort = arg
                console.print(f"[green]推理强度已设置为: {arg}[/green]")
            else:
                console.print("[yellow]用法: /set effort <high|max>[/yellow]")
        
        elif sub == "show":
            self._show_config()
        else:
            console.print("[yellow]用法: /set system <提示词>  /set temp <值>  /set model <flash|pro>  /set thinking <on|off>  /set effort <high|max>  /set show[/yellow]")

    def _show_config(self) -> None:
        console.print(f"[cyan]系统提示词: {self.current_system}[/cyan]")
        console.print(f"[cyan]温度: {self.current_temperature}[/cyan]")
        console.print(f"[cyan]模型: {self.current_model}[/cyan]")
        console.print(f"[cyan]思考模式: {'开' if self.thinking_enabled else '关'} | 推理强度: {self.reasoning_effort}[/cyan]")
        console.print(f"[cyan]搜索配额: 剩余 {self.search_quota} 次[/cyan]")
        if self.tree and self.tree.current_node:
            console.print(f"[cyan]当前节点: {self.tree.current_node.id} ({self.tree.current_node.title})[/cyan]")

    def _show_help(self) -> None:
        help_text = """
        可用命令：
        /exit, /quit, /q, /e  - 退出程序
        /clear, /c            - 清除对话历史
        /set system <提示词>   - 设置系统提示词
        /set temp <值>        - 设置温度参数
        /set model <flash|pro>- 切换模型（flash 或 pro）
        /set thinking <on|off>- 开启/关闭思考模式
        /set effort <high|max>- 设置推理强度
        /set show             - 显示当前所有配置
        /search <次数>        - 为 AI 授权 N 次互联网搜索（调用 web_search 消耗配额）
        /show                 - 将当前节点的回答正文保存到临时文件，并使用系统默认编辑器打开
        /help, /h             - 显示此帮助
        /imp <文件路径>       - 导入文件内容（txt/md/py/bat/sh/csv/pdf/docx），下次提问自动附加
        /fetch <URL>          - 抓取网页内容，下次提问自动附加

        树状命令：
        /cd <节点ID>          - 切换到指定节点
        /list                 - 列出所有节点
        /info [节点ID]        - 查看节点详情
        /back                 - 返回父节点
        /root                 - 跳转到根节点
        /save [节点ID]        - 保存当前或指定节点
        /rm <节点ID>          - 删除节点及其所有子节点（根节点不可删除）
        """
        console.print(help_text.strip())

    def _handle_tree_command(self, cmd: str) -> bool:
        parts = cmd.split()
        cmd_lower = parts[0].lower()
        
        if cmd_lower == "/cd" and len(parts) == 2:
            node_id = parts[1]
            if self.tree.switch_to_node(node_id):
                bt = self.tree.get_branch_total_tokens(self.tree.current_node.id)
                self._display_tree_node(self.tree.current_node, bt)
                console.print("\n[bold green]--- 已切换节点 ---[/bold green]\n")
            else:
                console.print("[red]未找到该节点ID[/red]")
            return True
        
        if cmd_lower == "/list":
            table = Table(title="所有节点")
            table.add_column("ID", style="cyan")
            table.add_column("标题", style="green")
            table.add_column("父节点", style="dim")
            for nid, node in self.tree.nodes.items():
                table.add_row(nid, node.title, node.parent_id or "根")
            console.print(table)
            return True
        
        if cmd_lower.startswith("/info"):
            nid = parts[1] if len(parts) > 1 else self.tree.current_node.id
            node = self.tree.nodes.get(nid)
            if node:
                console.print(Panel(f"节点 {node.id}: {node.title}", style="bold"))
                console.print(f"用户: {node.user_msg[:PREVIEW_USER_MSG_LEN]}...")
                console.print(f"助手: {node.assistant_msg[:PREVIEW_ASSISTANT_MSG_LEN]}...")
                console.print(f"Tokens: 输入 {node.input_tokens} / 输出 {node.output_tokens}")
            else:
                console.print("[red]节点不存在[/red]")
            return True
        
        if cmd_lower == "/back":
            if self.tree.current_node and self.tree.current_node.parent_id:
                parent = self.tree.nodes.get(self.tree.current_node.parent_id)
                if parent:
                    self.tree.current_node = parent
                    bt = self.tree.get_branch_total_tokens(parent.id)
                    self._display_tree_node(parent, bt)
                    console.print("\n[bold green]--- 已返回父节点 ---[/bold green]\n")
            else:
                console.print("[yellow]已在根节点[/yellow]")
            return True
        
        if cmd_lower == "/root":
            if self.tree.root:
                self.tree.current_node = self.tree.root
                bt = self.tree.get_branch_total_tokens(self.tree.root.id)
                self._display_tree_node(self.tree.root, bt)
                console.print("\n[bold green]--- 已跳转到根节点 ---[/bold green]\n")
            return True
        
        if cmd_lower.startswith("/save"):
            nid = parts[1] if len(parts) > 1 else self.tree.current_node.id
            self._save_tree_node(nid)
            return True
        
        if cmd_lower.startswith("/rm"):
            nid = parts[1] if len(parts) > 1 else None
            if nid is None:
                console.print("[yellow]用法: /rm <节点ID>[/yellow]")
                return True
            if nid not in self.tree.nodes:
                console.print(f"[red]未找到节点 {nid}[/red]")
                return True
            # 禁止删除根节点
            if nid == "main" or nid == self.tree.root.id:
                console.print("[red]不能删除根节点[/red]")
                return True
            # 重新获取节点对象（已确认存在）
            node_to_delete = self.tree.nodes[nid]
            # 确认删除
            child_count = len(self.tree.nodes) - 1  # 粗略估算会删除多少节点
            console.print(f"[yellow]确定要删除节点 {nid} 及其所有子节点吗？(y/N)[/yellow]")
            try:
                confirm = console.input("").strip().lower()
            except (KeyboardInterrupt, EOFError):
                confirm = "n"
            if confirm != "y":
                console.print("[dim]取消删除[/dim]")
                return True
            if self.tree.delete_node(nid):
                self._cleanup_temp_files(keep_ids=set(self.tree.nodes.keys()))
                # 删除成功后，重新显示当前节点
                if self.tree.current_node:
                    bt = self.tree.get_branch_total_tokens(self.tree.current_node.id)
                    self._display_tree_node(self.tree.current_node, bt)
                console.print(f"[green]节点 {nid} 及其所有子节点已删除[/green]")
            else:
                console.print(f"[red]删除节点 {nid} 失败[/red]")
            return True
        
        return False

    def process_user_input(self, user_input: str) -> None:
        if self.imported_content:
            user_input = self.imported_content + "\n\n" + user_input
            self.imported_content = None
        self._process_tree_input(user_input)

    def _process_tree_input(self, user_input: str) -> None:
        if self.tree.current_node is None:
            messages = [{"role": "system", "content": self.current_system},
                        {"role": "user", "content": user_input}]
        else:
            messages = self.tree.get_messages_for_node(self.tree.current_node)
            messages.append({"role": "user", "content": user_input})

        final_answer = None
        accumulated_reasoning = ""
        accumulated_in_tok = 0
        accumulated_out_tok = 0
        tool_messages: List[Dict] = []

        while True:
            sr = stream_response(
                self.client, messages, self.current_model,
                self.current_temperature, user_input,
                thinking_enabled=self.thinking_enabled,
                reasoning_effort=self.reasoning_effort,
                tools=TOOLS,
            )

            content, reasoning, in_tok, out_tok, tool_calls = (
                sr.content, sr.reasoning, sr.input_tokens, sr.output_tokens, sr.tool_calls
            )
            if reasoning:
                accumulated_reasoning += ("\n" if accumulated_reasoning else "") + reasoning
            accumulated_in_tok += in_tok
            accumulated_out_tok += out_tok

            if tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [],
                }
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning

                tool_results: List[Dict] = []
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    console.print(f"[dim]🔧 调用 {name}…[/dim]")

                    if name == "read_file":
                        tool_result = self._parse_file(args.get("filepath", ""))
                    elif name == "fetch_webpage":
                        tool_result = self._fetch_webpage(args.get("url", ""))
                    elif name == "list_directory":
                        tool_result = self._list_directory(args.get("directory", ""), args.get("show_hidden", False))
                    elif name == "write_file":
                        tool_result = self._write_file(args.get("filepath", ""), args.get("content", ""))
                    elif name == "edit_file":
                        tool_result = self._edit_file(args.get("filepath", ""), args.get("old_string", ""), args.get("new_string", ""))
                    elif name == "web_search":
                        if self.search_quota <= 0:
                            console.print("[yellow]🔍 AI 请求搜索授权，输入 /search <次数> 授权，输入其他内容跳过[/yellow]")
                            try:
                                cmd = self.session.prompt("搜索授权> ")
                            except (KeyboardInterrupt, EOFError):
                                cmd = ""
                            if cmd.strip().lower().startswith("/search"):
                                parts = cmd.strip().split()
                                if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                                    self.search_quota = int(parts[1])
                                    console.print(f"[green]✅ 已授权 {self.search_quota} 次搜索[/green]")
                                else:
                                    tool_result = "用户未授权此次搜索"
                            else:
                                tool_result = "用户未授权此次搜索"
                        if self.search_quota > 0:
                            self.search_quota -= 1
                            query = args.get("query", "")
                            freshness = args.get("freshness", "noLimit")
                            count = args.get("count", 10)
                            tool_result = self._web_search(query, freshness, count)
                            console.print(f"[dim]剩余搜索配额: {self.search_quota}[/dim]")
                    elif name == "execute_command":
                        command = args.get("command", "")
                        timeout = args.get("timeout", 30)
                        level, desc, risk = self._audit_command(command)
                        level_icons = {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠", 5: "🔴"}
                        icon = level_icons.get(level, "⚪")
                        console.print(f"[bold]{icon} 审核建议: 等级 {level}/5 | {desc}[/bold]")
                        if risk:
                            console.print(f"[yellow]⚠️ 风险提示: {risk}[/yellow]")
                        console.print(f"[cyan]命令: {command}[/cyan]")
                        try:
                            confirm = console.input("是否执行？(y/n) ")
                        except (KeyboardInterrupt, EOFError):
                            confirm = "n"
                        if confirm.strip().lower() == "y":
                            tool_result = self._execute_command(command, timeout)
                        else:
                            tool_result = "用户未确认执行此命令"
                    else:
                        tool_result = f"未知工具: {name}"

                    assistant_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result if tool_result else "执行失败或无结果",
                    })

                    args_str = json.dumps(args, ensure_ascii=False)
                    accumulated_reasoning += f"\n\n[调用工具] {name}({args_str})"
                    if tool_result:
                        summary = tool_result.strip()[:200].replace("\n", " ")
                        accumulated_reasoning += f"\n[工具返回] {summary}{'…' if len(tool_result.strip()) > 200 else ''}"

                messages.append(assistant_msg)
                messages.extend(tool_results)
                tool_messages.append(assistant_msg)
                tool_messages.extend(tool_results)
                continue

            if content is not None:
                final_answer = content
                final_reasoning = accumulated_reasoning
                break

            console.print("[red]回答生成失败，请重试[/red]")
            return

        title = generate_conversation_title(self.client, user_input, final_answer)
        if not self.tree.root:
            node = self.tree.create_root(user_input, final_answer, final_reasoning, title, accumulated_in_tok, accumulated_out_tok)
        else:
            node = self.tree.add_child(
                self.tree.current_node, user_input, final_answer, final_reasoning, title, accumulated_in_tok, accumulated_out_tok
            )
        if tool_messages:
            node.tool_messages = tool_messages
        self.tree.current_node = node
        branch_total = self.tree.get_branch_total_tokens(node.id)
        self._display_tree_node(node, branch_total)
        console.print("\n[bold green]--- 请输入下一个问题或命令 ---[/bold green]\n")

    def run(self) -> None:
        self._show_welcome()
        
        try:
            while True:
                try:
                    prompt_text = self._get_prompt_text()
                    user_input = self.session.prompt(prompt_text)
                except (KeyboardInterrupt, EOFError):
                    console.print("\n再见！👋")
                    break

                cmd = user_input.strip()
                if not cmd:
                    continue

                if self.handle_command(cmd):
                    if cmd.lower() in ["/exit", "/quit", "/q", "/e"]:
                        break
                    continue

                self.process_user_input(cmd)
        finally:
            self._save_session()

    def _get_prompt_text(self) -> str:
        if self.tree and self.tree.current_node:
            return f"[{self.tree.current_node.id}] 你: "
        return "你: "

    def _show_welcome(self) -> None:
        clear_screen()
        console.print(Panel.fit("mincli 树状对话模式", style="bold green"))
        console.print(
            "命令: /set system <提示词>  /set temp <值>  /set model <flash|pro>  "
            "/set thinking <on|off>  /set effort <high|max>  /set show  /clear  /exit /imp <路径>  /fetch <URL>  /show"
        )
        console.print("树状命令: /cd <ID>  /list  /info [ID]  /back  /root  /save [ID] /rm <ID>")
        console.print(f"💡 当前模型: [bold]{self.current_model}[/bold] | 思考: [bold]{'开' if self.thinking_enabled else '关'}[/bold] (effort: {self.reasoning_effort})")
        console.print("[dim]等待第一个问题...[/dim]\n")
    
    def _fetch_webpage(self, url: str) -> str:

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return f"无法获取网页内容: {url}"
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if not text:
                return f"无法从网页中提取有效文本: {url}"

            text = text.strip()
            if len(text) > WEBPAGE_MAX_LENGTH:
                text = text[:WEBPAGE_MAX_LENGTH] + "\n\n...(已截断)"
            return text
        except Exception as e:
            return f"抓取或解析失败: {e}"

    def _list_directory(self, directory: str, show_hidden: bool = False) -> str:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            return f"目录不存在: {directory}"
        try:
            entries = []
            for entry in os.scandir(directory):
                if not show_hidden and entry.name.startswith("."):
                    continue
                prefix = "[目录] " if entry.is_dir() else "[文件] "
                entries.append(f"{prefix}{entry.name}")
            if not entries:
                return "(空目录)"
            return f"目录: {directory}\n" + "\n".join(entries)
        except PermissionError:
            return f"(权限不足，无法读取 {directory})"
        except Exception as e:
            return f"读取目录失败: {e}"

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
        console.print(f"[yellow]⚠️ 即将{'覆盖' if exists else '写入'}文件[/yellow]")
        console.print(details)
        console.print("[yellow]确认执行? (y/N)[/yellow]")
        try:
            confirm = console.input("").strip().lower()
        except (KeyboardInterrupt, EOFError):
            confirm = "n"
        if confirm != "y":
            return "用户已取消操作"

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"已成功写入 {len(content)} 字符到 {filepath}"
        except Exception as e:
            return f"写入失败: {e}"

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

        new_content = content.replace(old_string, new_string, 1)

        details = f"路径: {filepath}\n替换内容:\n"
        for line in old_string.split("\n"):
            details += f"  - {line}\n"
        details += "  替换为:\n"
        for line in new_string.split("\n"):
            details += f"  + {line}\n"

        console.print(f"[yellow]⚠️ 即将修改文件[/yellow]")
        console.print(details)
        console.print("[yellow]确认执行? (y/N)[/yellow]")
        try:
            confirm = console.input("").strip().lower()
        except (KeyboardInterrupt, EOFError):
            confirm = "n"
        if confirm != "y":
            return "用户已取消操作"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"已成功替换文件 {filepath}"
        except Exception as e:
            return f"写入失败: {e}"

    def _parse_file(self, filepath: str) -> str:
        filepath = os.path.expanduser(filepath)
        if not os.path.exists(filepath):
            return f"文件不存在: {filepath}"
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        content = ""

        try:
            if ext in ('.txt', '.md', '.py', '.bat', '.sh'):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            elif ext == '.csv':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    rows = [','.join(row) for row in reader]
                    content = '\n'.join(rows)
            elif ext == '.pdf':
                try:
                    from pdfminer.high_level import extract_text
                    content = extract_text(filepath)
                except ImportError:
                    return "需安装 pdfminer.six: pip install pdfminer.six"
            elif ext == '.docx':
                try:
                    from docx import Document
                    doc = Document(filepath)
                    content = '\n'.join([p.text for p in doc.paragraphs])
                except ImportError:
                    return "需安装 python-docx: pip install python-docx"
            elif ext == '.doc':
                return "不支持 .doc 格式，请转换为 .docx 或 .txt"
            else:
                return f"不支持的文件格式: {ext}"

            if not content.strip():
                return f"文件内容为空: {filename}"

            return f"{filename}：\n{content.strip()}"
        except Exception as e:
            return f"文件解析失败: {e}"

    def _web_search(self, query: str, freshness: str = "noLimit", count: int = 10) -> str:
        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "错误: 未配置 BOCHA_API_KEY"
        try:
            resp = requests.post(
                BOCHA_API_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": query, "freshness": freshness, "summary": True, "count": min(count, 50)},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("data", {}).get("webPages", {}).get("value", [])
            if not pages:
                return f"搜索 \"{query}\" 未找到相关结果"
            lines = [f"搜索 \"{query}\" 共找到 {len(pages)} 条结果：\n"]
            for i, p in enumerate(pages, 1):
                name = p.get("name", "")
                url = p.get("url", "")
                snippet = p.get("snippet", "")
                date = (p.get("dateLastCrawled") or "")[:10]
                lines.append(f"{i}. {name}\n   链接: {url}\n   摘要: {snippet}\n   日期: {date}\n")
            return "\n".join(lines)
        except Exception as e:
            return f"搜索请求失败: {e}"

    def _audit_command(self, command: str) -> Tuple[int, str, str]:
        audit_system = (
            "你是 mincli 的命令安全审核员，负责审查 shell 命令的安全性。\n"
            "请分析以下命令并以 JSON 格式回复：\n\n"
            '{\n'
            '  "level": <1-5 的数字>,\n'
            '  "description": "简要说明命令功能（50字左右）",\n'
            '  "risk": "风险说明（若无风险则留空）"\n'
            '}\n\n'
            "等级含义：\n"
            "1 = 强烈建议执行（完全安全、无害的命令）\n"
            "2 = 建议执行（基本安全）\n"
            "3 = 中性 / 不确定\n"
            "4 = 不建议执行（可能有风险）\n"
            "5 = 强烈禁止执行（危险命令）"
        )
        audit_messages = [
            {"role": "system", "content": audit_system},
            {"role": "user", "content": f"请审核以下命令：\n\n```bash\n{command}\n```"},
        ]
        sr = stream_response(
            self.client, audit_messages, MODEL_V4_FLASH,
            0.3, command,
            thinking_enabled=True,
            reasoning_effort="high",
            tools=None,
        )
        content = sr.content or ""
        if sr.reasoning:
            console.print(f"[dim]🧠 审核思考: {sr.reasoning}[/dim]")
        try:
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("level", 3), result.get("description", ""), result.get("risk", "")
        except Exception:
            pass
        return 3, content.strip()[:100], ""

    def _execute_command(self, command: str, timeout: int) -> str:
        try:
            workdir = os.path.expanduser("~")
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=workdir,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"[stderr]\n{result.stderr}"
            output += f"\n[退出码: {result.returncode}]"
            return output.strip()
        except subprocess.TimeoutExpired as e:
            partial = ""
            if e.stdout:
                partial += e.stdout
            if e.stderr:
                partial += f"[stderr]\n{e.stderr}"
            partial = partial.strip()
            if partial:
                return f"{partial}\n[命令执行超时（{timeout}秒），以上为已产生的部分输出]"
            return f"命令执行超时（{timeout}秒），无任何输出"
        except Exception as e:
            return f"命令执行失败: {e}"


# ---------- CLI 入口 ----------
def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[red]错误: 未设置 DEEPSEEK_API_KEY[/red]")
        raise typer.Exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


@app.command()
def chat(
    model: str = typer.Option("flash", "--model", "-m", help="模型: flash 或 pro"),
    temperature: float = typer.Option(1.0, "--temp", "-temp-opt", help="温度参数"),
    thinking: bool = typer.Option(False, "--thinking", "-r", help="开启思考模式（默认 high）"),
    effort: str = typer.Option("high", "--effort", help="推理强度: high 或 max"),
) -> None:
    """启动树状对话模式。"""
    selected_model = MODEL_V4_PRO if model.lower() == "pro" else MODEL_V4_FLASH

    if thinking:
        console.print(f"[cyan]🧠 开启思考模式 (effort: {effort})[/cyan]")
    else:
        console.print(f"[dim]🧠 思考模式关闭[/dim]")

    client = get_client()
    session = InteractiveSession(
        client=client,
        default_system="你是一个有用的人工智能助手",
        default_temperature=temperature,
        default_model=selected_model,
        thinking_enabled=thinking,
        reasoning_effort=effort,
    )
    session.run()


@app.command()
def info() -> None:
    """显示当前配置信息。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    table = Table(title="mincli 配置")
    table.add_column("项目", style="cyan")
    table.add_column("状态", style="green")
    table.add_row("API Key", "已配置 ✓" if api_key else "未配置 ✗")
    table.add_row("模型", f"{MODEL_V4_FLASH}\n{MODEL_V4_PRO}")
    table.add_row("保存路径", SAVE_BASE_DIR)
    table.add_row("模式", "树状对话")
    table.add_row("输出方式", "流式实时刷新 + Markdown 渲染")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()