#!/usr/bin/env python3
"""
DeepSeek CLI - 交互式 AI 助手（支持树状对话 & 推理过程）

特性：
    - 流式输出，实时刷新 Markdown 渲染。
    - 空输入时按 ↑/↓ 浏览历史对话，按 ESC 返回最新对话。
    - 支持 /set 命令动态修改系统提示词、温度、模型及思考开关。
    - 支持 /save 保存单条对话，/save_group 批量保存到组文件夹。
    - 树状对话模式（-t）：全局唯一节点ID，分支自由切换。
    - 交互过程中可随时切换思考/非思考模型，并完整保留推理内容。

使用方法：
    python main.py chat -i          # 进入普通交互模式
    python main.py chat -t          # 进入树状对话模式
    python main.py info             # 显示配置信息
"""

import os
import re
import sys
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Union, Any
import json

import tiktoken
import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree as RichTree
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.keys import Keys

# ---------- 初始化 ----------
load_dotenv()
console = Console(stderr=True, highlight=False)
app = typer.Typer(help="DeepSeek CLI - 交互式 AI 助手（支持树状对话）")

# 模型常量（V4）
MODEL_V4_FLASH = "deepseek-v4-flash"   # 轻量快速
MODEL_V4_PRO = "deepseek-v4-pro"       # 旗舰性能
DEFAULT_MODEL = MODEL_V4_FLASH

# 保存路径配置
SAVE_BASE_DIR = os.path.expanduser(
    os.getenv("DEEPSEEK_SAVE_PATH", "~/Documents/DeepSeek_Conversations")
)


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
            max_tokens=30,
            extra_body={"thinking": {"type": "disabled"}}   # 显式关闭思考
        )
        title = resp.choices[0].message.content.strip()
        # 移除文件系统不安全的字符
        title = re.sub(r'[\\/*?:"<>|]', '', title)
        title = title.replace(' ', '_')
        if len(title) > 30:
            title = title[:30]
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
def stream_response(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    user_question: str,
    thinking_enabled: bool = False,
    reasoning_effort: str = "high",
) -> Tuple[Optional[str], Optional[str], int, int]:
    """
    流式获取 AI 回复，并使用 Rich Live 实时渲染 Markdown。
    
    Returns:
        (完整回答内容, 推理内容, 输入 token 数, 输出 token 数)
        若调用失败，回答内容和推理内容为 None，token 统计为 0。
    """
    estimated_input = estimate_tokens(messages)
    full_content = ""
    reasoning_text = ""
    usage_input = 0
    usage_output = 0

    try:
        # 将自定义参数通过 extra_body 传递
        extra_body = {}
        if thinking_enabled:
            extra_body["thinking"] = {"type": "enabled"}
            extra_body["reasoning_effort"] = reasoning_effort
        else:
            extra_body["thinking"] = {"type": "disabled"}

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            extra_body=extra_body,      # 通过 extra_body 传入
        )

        with Live(auto_refresh=False, console=console, screen=True) as live:
            header = f"**你:**\n{user_question}\n\n"
            initial_display = header + "**DeepSeek:** "
            live.update(Markdown(initial_display), refresh=True)

            for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_input = chunk.usage.prompt_tokens
                    usage_output = chunk.usage.completion_tokens

                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_text += delta.reasoning_content
                if delta.content:
                    full_content += delta.content

                # 构建动态显示
                display = header
                if reasoning_text:
                    display += "[dim]**DeepSeek 思考过程:**\n "
                    display += reasoning_text + "[/dim]\n\n"
                display += f"**DeepSeek:** {full_content}"
                live.update(Markdown(display), refresh=True)

            # 最终渲染
            final_display = header
            if reasoning_text:
                final_display += "[dim]**DeepSeek 思考过程:**\n "
                final_display += reasoning_text + "[/dim]\n\n"
            final_display += f"**DeepSeek:** {full_content}"
            live.update(Markdown(final_display), refresh=True)

        if usage_input == 0 and usage_output == 0:
            input_tokens = estimated_input
            output_tokens = estimate_tokens([{"role": "assistant", "content": full_content}])
        else:
            input_tokens = usage_input
            output_tokens = usage_output

        return full_content, reasoning_text, input_tokens, output_tokens

    except Exception as e:
        console.print(f"[red]API 调用失败: {e}[/red]")
        return None, None, 0, 0


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
    SAVE_FILE_TREE = os.path.expanduser("~/.deepseek_cli_tree_session.json")
    SAVE_FILE_LINEAR = os.path.expanduser("~/.deepseek_cli_linear_session.json")

    def __init__(
        self,
        client: OpenAI,
        default_system: str,
        default_temperature: float,
        default_model: str = DEFAULT_MODEL,
        start_tree_mode: bool = False,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
    ):
        self.client = client
        self.current_system = default_system
        self.current_temperature = default_temperature
        self.current_model = default_model
        self.tree_mode = start_tree_mode
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort

        # 线性模式状态
        self.linear_conversations: List[Dict] = []
        self.linear_messages = [{"role": "system", "content": default_system}]

        # 树状模式状态
        self.tree: Optional[ConversationTree] = None
        if start_tree_mode:
            self.tree = ConversationTree(default_system)

        # 历史浏览状态（线性模式）
        self.browse_mode = False
        self.browse_index = -1

        # Prompt Toolkit 会话
        self.history_file = os.path.expanduser("~/.deepseek_cli_history")
        self.session = PromptSession(history=FileHistory(self.history_file))
        self.bindings = self._create_key_bindings()

        self._load_session()

    def _get_save_file_path(self) -> str:
        return self.SAVE_FILE_TREE if self.tree_mode else self.SAVE_FILE_LINEAR

    def _save_session(self) -> None:
        """保存会话状态，包含异常提示。"""
        filepath = self._get_save_file_path()
        try:
            if self.tree_mode and self.tree:
                data = {
                    "mode": "tree",
                    "system_prompt": self.current_system,
                    "temperature": self.current_temperature,
                    "model": self.current_model,
                    "thinking_enabled": self.thinking_enabled,
                    "reasoning_effort": self.reasoning_effort,
                    "tree": self.tree.to_dict(),
                }
            else:
                data = {
                    "mode": "linear",
                    "system_prompt": self.current_system,
                    "temperature": self.current_temperature,
                    "model": self.current_model,
                    "thinking_enabled": self.thinking_enabled,
                    "reasoning_effort": self.reasoning_effort,
                    "linear_conversations": self.linear_conversations,
                    "linear_messages": self.linear_messages,
                }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[red]⚠️ 会话保存失败: {e}[/red]")

    def _load_session(self) -> bool:
        # 检查两个可能的会话文件
        files = [
            (self.SAVE_FILE_LINEAR, "linear"),
            (self.SAVE_FILE_TREE, "tree"),
        ]
        valid_files = []
        for path, mode in files:
            if os.path.exists(path):
                try:
                    mtime = os.path.getmtime(path)
                    valid_files.append((mtime, path, mode))
                except OSError:
                    continue

        if not valid_files:
            return False

        valid_files.sort(reverse=True)  # 按修改时间最新优先
        _, filepath, file_mode = valid_files[0]

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]⚠️ 会话文件损坏，已忽略: {e}[/red]")
            try:
                os.remove(filepath)
            except:
                pass
            return False

        was_tree_mode = self.tree_mode
        self.tree_mode = (file_mode == "tree")

        self.current_system = data.get("system_prompt", self.current_system)
        self.current_temperature = data.get("temperature", self.current_temperature)
        self.current_model = data.get("model", self.current_model)
        self.thinking_enabled = data.get("thinking_enabled", False)
        self.reasoning_effort = data.get("reasoning_effort", "high")

        if self.tree_mode:
            tree_data = data.get("tree")
            if tree_data:
                self.tree = ConversationTree.from_dict(tree_data)
            else:
                self.tree = ConversationTree(self.current_system)
            self.linear_conversations = []
            self.linear_messages = [{"role": "system", "content": self.current_system}]
        else:
            self.linear_conversations = data.get("linear_conversations", [])
            self.linear_messages = data.get("linear_messages", [{"role": "system", "content": self.current_system}])
            if self.linear_messages and self.linear_messages[0].get("role") == "system":
                self.linear_messages[0]["content"] = self.current_system
            else:
                self.linear_messages.insert(0, {"role": "system", "content": self.current_system})
            self.tree = None

        if was_tree_mode != self.tree_mode:
            console.print(f"[dim]📂 已从文件恢复为{'树状' if self.tree_mode else '线性'}模式[/dim]")
        else:
            console.print("[dim]📂 已加载上次会话记录[/dim]")
        return True

    def _delete_session_file(self) -> None:
        filepath = self._get_save_file_path()
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
    
    def _create_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add(Keys.Up, filter=Condition(lambda: not self.browse_mode and self.session.default_buffer.text == ""))
        def up_browse(event):
            if self.tree_mode or not self.linear_conversations:
                return
            self.browse_mode = True
            self.browse_index = len(self.linear_conversations) - 1
            self._display_linear_conversation(self.browse_index)

        @bindings.add(Keys.Down, filter=Condition(lambda: not self.browse_mode and self.session.default_buffer.text == ""))
        def down_browse(event):
            if self.tree_mode or not self.linear_conversations:
                return
            self.browse_mode = True
            self.browse_index = 0
            self._display_linear_conversation(self.browse_index)

        @bindings.add(Keys.Up, filter=Condition(lambda: self.browse_mode))
        def up_nav(event):
            if not self.linear_conversations:
                return
            self.browse_index = max(0, self.browse_index - 1)
            self._display_linear_conversation(self.browse_index)

        @bindings.add(Keys.Down, filter=Condition(lambda: self.browse_mode))
        def down_nav(event):
            if not self.linear_conversations:
                return
            self.browse_index = min(len(self.linear_conversations) - 1, self.browse_index + 1)
            self._display_linear_conversation(self.browse_index)

        @bindings.add(Keys.Escape, filter=Condition(lambda: self.browse_mode))
        def exit_browse_esc(event):
            self.browse_mode = False
            if not self.tree_mode:
                self._display_latest_linear()
            event.app.invalidate()

        return bindings

    def _display_linear_conversation(self, index: int) -> None:
        clear_screen()
        conv = self.linear_conversations[index]
        self._render_conversation(conv['user'], conv['assistant'], conv.get('reasoning', ''),
                                  f"对话 {index + 1}", conv['input_tokens'], conv['output_tokens'])

    def _display_latest_linear(self) -> None:
        clear_screen()
        if not self.linear_conversations:
            console.print("[dim]暂无对话历史，请输入您的问题。[/dim]")
            return
        self._display_linear_conversation(len(self.linear_conversations) - 1)

    def _render_conversation(self, user_msg: str, assistant_msg: str, reasoning: str,
                             title: str, input_tokens: int, output_tokens: int) -> None:
        console.print(Panel(title, style="bold cyan"))
        if reasoning:
            console.print("[dim]DeepSeek 思考过程:[/dim]")
            console.print(f"[dim]> {reasoning.replace(chr(10), chr(10)+'> ')}[/dim]\n")
        console.print(Markdown(f"**你:** {user_msg}\n\n**DeepSeek:** {assistant_msg}"))
        console.print(
            f"[dim]📊 输入: {input_tokens} tokens | 输出: {output_tokens} tokens[/dim]"
        )

    def _display_tree_node(self, node: ConversationNode) -> None:
        clear_screen()
        self._render_conversation(node.user_msg, node.assistant_msg, node.reasoning,
                                  f"节点 {node.id}: {node.title}",
                                  node.input_tokens, node.output_tokens)
        console.print("[bold]对话树：[/bold]")
        console.print(self.tree.render_tree(node.id))
        console.print(f"[dim]当前节点: {node.id} ({node.title})[/dim]")

    def _save_linear_conversation(self, index: int) -> None:
        if index < 1 or index > len(self.linear_conversations):
            console.print(f"[red]序号无效，有效范围 1-{len(self.linear_conversations)}[/red]")
            return
        
        conv = self.linear_conversations[index - 1]
        title = conv.get('title') or generate_conversation_title(
            self.client, conv['user'], conv['assistant']
        )
        conv['title'] = title
        
        content = (
            f"# 对话 #{index}\n\n"
            f"**时间：** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## 用户问题\n\n{conv['user']}\n\n"
        )
        if conv.get('reasoning'):
            content += f"## DeepSeek 思考过程\n\n {conv['reasoning'].replace(chr(10), chr(10)+'> ')}\n\n"
        content += f"## DeepSeek 回答\n\n{conv['assistant']}\n\n"
        token_stats = {
            'input_tokens': conv['input_tokens'],
            'output_tokens': conv['output_tokens']
        }
        filepath = save_conversation_to_file(content, title, "", token_stats)
        console.print(f"[green]✅ 已保存对话 {index} 到 {filepath}[/green]")

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
            content += f"## DeepSeek 思考过程\n\n {node.reasoning.replace(chr(10), chr(10)+'> ')}\n\n"
        content += f"## DeepSeek 回答\n\n{node.assistant_msg}\n\n"
        token_stats = {
            'input_tokens': node.input_tokens,
            'output_tokens': node.output_tokens
        }
        filepath = save_conversation_to_file(content, node.title, node.id, token_stats)
        console.print(f"[green]✅ 节点已保存到 {filepath}[/green]")

    def _convert_linear_to_tree(self) -> None:
        self.tree = ConversationTree(self.current_system)
        if not self.linear_conversations:
            self.tree_mode = True
            return
        
        prev_node = None
        for i, conv in enumerate(self.linear_conversations):
            title = conv.get('title') or generate_conversation_title(
                self.client, conv['user'], conv['assistant']
            )
            conv['title'] = title
            if i == 0:
                node = self.tree.create_root(
                    conv['user'], conv['assistant'], conv.get('reasoning', ''),
                    title, conv['input_tokens'], conv['output_tokens']
                )
            else:
                node = self.tree.add_child(
                    prev_node, conv['user'], conv['assistant'], conv.get('reasoning', ''),
                    title, conv['input_tokens'], conv['output_tokens']
                )
            prev_node = node
        
        self.tree.current_node = prev_node
        self.tree_mode = True
        console.print("[green]已切换到树状对话模式。[/green]")
        self._display_tree_node(self.tree.current_node)

    def handle_command(self, cmd: str) -> bool:
        cmd_stripped = cmd.strip()
        
        if cmd_stripped.startswith("/"):
            valid_commands = [
                "/exit", "/quit", "/q", "/e",
                "/clear", "/c",
                "/set",
                "/tree",
                "/cd", "/list", "/info", "/back", "/root", "/save_node",
                "/save", "/save_group",
            ]
            cmd_lower = cmd_stripped.lower()
            is_valid = any(cmd_lower.startswith(vc) for vc in valid_commands)
            if not is_valid:
                console.print(f"[yellow]未知命令: {cmd_stripped}。输入 /help 查看可用命令（暂不支持）。[/yellow]")
                return True
        
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ["/exit", "/quit", "/q", "/e"]:
            console.print("再见！👋")
            return True

        if cmd_lower in ["/clear", "/c"]:
            self._clear_history()
            return True

        if cmd.startswith("/set"):
            self._handle_set_command(cmd)
            return True

        if cmd_lower == "/tree":
            if not self.tree_mode:
                self._convert_linear_to_tree()
            else:
                console.print("[yellow]已在树状模式中[/yellow]")
            return True

        if self.tree_mode and self.tree:
            if self._handle_tree_command(cmd):
                return True

        if not self.tree_mode:
            if cmd.startswith("/save_group"):
                console.print("[yellow]批量保存功能简化，请使用 /save <序号>[/yellow]")
                return True
            if cmd.startswith("/save"):
                self._handle_linear_save(cmd)
                return True

        return False

    def _clear_history(self) -> None:
        if self.tree_mode:
            self.tree = ConversationTree(self.current_system)
        else:
            self.linear_messages = [{"role": "system", "content": self.current_system}]
            self.linear_conversations.clear()
        
        self._delete_session_file()
        clear_screen()
        console.print("[dim]对话历史已清除[/dim]")
        console.print("[dim]等待下一个问题...[/dim]\n")

    def _handle_set_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=2)
        if len(parts) < 2:
            console.print("[yellow]用法: /set system <提示词>  /set temp <值>  /set model <flash|pro>  /set thinking <on|off>  /set effort <high|max>  /set show[/yellow]")
            return
        
        sub = parts[1]
        if sub == "system" and len(parts) == 3:
            self.current_system = parts[2]
            if not self.tree_mode:
                self.linear_messages[0] = {"role": "system", "content": self.current_system}
            elif self.tree:
                self.tree.system_prompt = self.current_system
            console.print("[green]系统提示词已更新[/green]")
        
        elif sub == "temp" and len(parts) == 3:
            try:
                self.current_temperature = float(parts[2])
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
        console.print(f"[cyan]模式: {'树状' if self.tree_mode else '线性'}[/cyan]")
        if self.tree_mode and self.tree and self.tree.current_node:
            console.print(f"[cyan]当前节点: {self.tree.current_node.id} ({self.tree.current_node.title})[/cyan]")

    def _handle_tree_command(self, cmd: str) -> bool:
        parts = cmd.split()
        cmd_lower = parts[0].lower()
        
        if cmd_lower == "/cd" and len(parts) == 2:
            node_id = parts[1]
            if self.tree.switch_to_node(node_id):
                self._display_tree_node(self.tree.current_node)
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
                console.print(f"用户: {node.user_msg[:100]}...")
                console.print(f"助手: {node.assistant_msg[:200]}...")
                console.print(f"Tokens: 输入 {node.input_tokens} / 输出 {node.output_tokens}")
            else:
                console.print("[red]节点不存在[/red]")
            return True
        
        if cmd_lower == "/back":
            if self.tree.current_node and self.tree.current_node.parent_id:
                parent = self.tree.nodes.get(self.tree.current_node.parent_id)
                if parent:
                    self.tree.current_node = parent
                    self._display_tree_node(parent)
                    console.print("\n[bold green]--- 已返回父节点 ---[/bold green]\n")
            else:
                console.print("[yellow]已在根节点[/yellow]")
            return True
        
        if cmd_lower == "/root":
            if self.tree.root:
                self.tree.current_node = self.tree.root
                self._display_tree_node(self.tree.root)
                console.print("\n[bold green]--- 已跳转到根节点 ---[/bold green]\n")
            return True
        
        if cmd_lower.startswith("/save_node"):
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
                # 删除成功后，重新显示当前节点
                if self.tree.current_node:
                    self._display_tree_node(self.tree.current_node)
                console.print(f"[green]节点 {nid} 及其所有子节点已删除[/green]")
            else:
                console.print(f"[red]删除节点 {nid} 失败[/red]")
            return True
        
        return False

    def _handle_linear_save(self, cmd: str) -> None:
        parts = cmd.split()
        if len(parts) != 2:
            console.print("[yellow]用法: /save <序号>[/yellow]")
            return
        try:
            idx = int(parts[1])
            self._save_linear_conversation(idx)
        except ValueError:
            console.print("[red]序号必须是数字[/red]")

    def process_user_input(self, user_input: str) -> None:
        if self.tree_mode:
            self._process_tree_input(user_input)
        else:
            self._process_linear_input(user_input)
        
        self.browse_mode = False

    def _process_linear_input(self, user_input: str) -> None:
        self.linear_messages.append({"role": "user", "content": user_input})
        answer, reasoning, in_tok, out_tok = stream_response(
            self.client, self.linear_messages, self.current_model,
            self.current_temperature, user_input,
            thinking_enabled=self.thinking_enabled,
            reasoning_effort=self.reasoning_effort,
        )
        if answer is not None:
            conv = {
                "user": user_input,
                "assistant": answer,
                "reasoning": reasoning or "",
                "input_tokens": in_tok,
                "output_tokens": out_tok
            }
            self.linear_conversations.append(conv)
            assistant_msg = {"role": "assistant", "content": answer}
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            self.linear_messages.append(assistant_msg)
            self._display_latest_linear()
            console.print("\n[bold green]--- 请输入下一个问题 ---[/bold green]\n")
        else:
            self.linear_messages.pop()
            console.print("[red]回答生成失败，请重试[/red]")

    def _process_tree_input(self, user_input: str) -> None:
        if not self.tree:
            self.tree = ConversationTree(self.current_system)
            messages = [{"role": "system", "content": self.current_system},
                        {"role": "user", "content": user_input}]
        else:
            if self.tree.current_node is None:
                messages = [{"role": "system", "content": self.current_system},
                            {"role": "user", "content": user_input}]
            else:
                messages = self.tree.get_messages_for_node(self.tree.current_node)
                messages.append({"role": "user", "content": user_input})
        
        answer, reasoning, in_tok, out_tok = stream_response(
            self.client, messages, self.current_model,
            self.current_temperature, user_input,
            thinking_enabled=self.thinking_enabled,
            reasoning_effort=self.reasoning_effort,
        )
        if answer is not None:
            title = generate_conversation_title(self.client, user_input, answer)
            if not self.tree.root:
                node = self.tree.create_root(user_input, answer, reasoning or "", title, in_tok, out_tok)
            else:
                node = self.tree.add_child(
                    self.tree.current_node, user_input, answer, reasoning or "", title, in_tok, out_tok
                )
            self.tree.current_node = node
            self._display_tree_node(node)
            console.print("\n[bold green]--- 请输入下一个问题或命令 ---[/bold green]\n")
        else:
            console.print("[red]回答生成失败，请重试[/red]")

    def run(self) -> None:
        self._show_welcome()
        
        try:
            while True:
                try:
                    prompt_text = self._get_prompt_text()
                    user_input = self.session.prompt(prompt_text, key_bindings=self.bindings)
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
        if self.tree_mode and self.tree and self.tree.current_node:
            return f"[{self.tree.current_node.id}] 你: "
        return "你: "

    def _show_welcome(self) -> None:
        clear_screen()
        mode_str = "树状对话模式" if self.tree_mode else "交互模式"
        console.print(Panel.fit(f"DeepSeek CLI {mode_str}", style="bold green"))
        console.print(
            "命令: /set system <提示词>  /set temp <值>  /set model <flash|pro>  "
            "/set thinking <on|off>  /set effort <high|max>  /set show  /clear  /save <序号>  /tree  /exit"
        )
        if self.tree_mode:
            console.print("树状命令: /cd <ID>  /list  /info [ID]  /back  /root  /save_node [ID]")
        console.print(f"💡 当前模型: [bold]{self.current_model}[/bold] | 思考: [bold]{'开' if self.thinking_enabled else '关'}[/bold] (effort: {self.reasoning_effort})")
        if not self.tree_mode:
            console.print("💡 空输入时按 ↑/↓ 浏览历史对话，按 ESC 返回最新对话")
        console.print("[dim]等待第一个问题...[/dim]\n")

        if self.tree_mode and not self.tree:
            console.print("[dim]当前为空树，请输入第一个问题以创建根节点。[/dim]\n")


# ---------- CLI 入口 ----------
def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[red]错误: 未设置 DEEPSEEK_API_KEY[/red]")
        raise typer.Exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


@app.command()
def chat(
    interactive: bool = typer.Option(False, "--interactive", "-i", help="进入普通交互模式"),
    tree: bool = typer.Option(False, "--tree", "-t", help="进入树状对话模式"),
    model: str = typer.Option("flash", "--model", "-m", help="模型: flash 或 pro"),
    temperature: float = typer.Option(1.0, "--temp", "-temp-opt", help="温度参数"),
    thinking: bool = typer.Option(False, "--thinking", "-r", help="开启思考模式（默认 high）"),
    effort: str = typer.Option("high", "--effort", help="推理强度: high 或 max"),
) -> None:
    """启动交互式对话。"""
    if not (interactive or tree):
        console.print("[yellow]请指定模式：-i (普通交互) 或 -t (树状对话)[/yellow]")
        console.print("示例: python main.py chat -i")
        console.print("      python main.py chat -t")
        raise typer.Exit()

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
        start_tree_mode=tree,
        thinking_enabled=thinking,
        reasoning_effort=effort,
    )
    session.run()


@app.command()
def info() -> None:
    """显示当前配置信息。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    table = Table(title="DeepSeek CLI 配置")
    table.add_column("项目", style="cyan")
    table.add_column("状态", style="green")
    table.add_row("API Key", "已配置 ✓" if api_key else "未配置 ✗")
    table.add_row("模型", f"{MODEL_V4_FLASH}\n{MODEL_V4_PRO}")
    table.add_row("保存路径", SAVE_BASE_DIR)
    table.add_row("模式", "普通交互 (-i) / 树状对话 (-t)")
    table.add_row("输出方式", "流式实时刷新 + Markdown 渲染")
    table.add_row("历史浏览", "空输入时按 ↑/↓ 浏览，ESC 退出")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()