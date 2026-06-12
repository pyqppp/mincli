import os
import json
import datetime
import tempfile
import subprocess
from typing import Optional, List, Dict, Tuple

from openai import OpenAI
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from mincli.config import (
    MODEL_V4_FLASH, MODEL_V4_PRO,
    TEMPERATURE_MIN, TEMPERATURE_MAX,
    PREVIEW_USER_MSG_LEN, PREVIEW_ASSISTANT_MSG_LEN,
)
from mincli.models import ConversationTree
from mincli.helpers import (
    clear_screen, get_balance, format_balance,
    generate_conversation_title, save_conversation_to_file,
    convert_formulas,
)
from mincli.render import console
from mincli.streaming import stream_response
from mincli.tools.registry import TOOLS
from mincli.tools.web_fetch import fetch_webpage, web_search
from mincli.tools.file_ops import parse_file, list_directory
from mincli.tools.execute import execute_command, audit_command


class InteractiveSession:
    SAVE_FILE = os.path.expanduser("~/.mincli_session.json")

    def __init__(
        self,
        client: OpenAI,
        default_system: str,
        default_temperature: float,
        default_model: str = MODEL_V4_FLASH,
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
        except Exception:
            console.print(f"[red]⚠️ 会话文件损坏，已忽略[/red]")
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
        console.print(Markdown(f"**你:** {user_msg}"))
        if reasoning:
            console.print(Markdown("\n**DeepSeek 思考过程:**"))
            console.print(f"[dim]{reasoning}[/dim]")
        console.print(Markdown(f"**DeepSeek:** {assistant_msg}"))
        balance_infos = get_balance(self.client)
        balance_str = format_balance(balance_infos)
        console.print(
            f"[dim]📊 输入: {input_tokens} tokens | 输出: {output_tokens} tokens"
            f"{' | 💰 ' + balance_str if balance_str else ''}[/dim]"
        )

    def _display_tree_node(self, node, branch_total: Optional[Tuple[int, int]] = None) -> None:
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

        user_msg = convert_formulas(node.user_msg)
        assistant_msg = convert_formulas(node.assistant_msg)
        reasoning = convert_formulas(node.reasoning)

        content = f"# {node.title}\n\n"
        content += f"---\n\n**你：**\n\n{user_msg}\n\n"
        if node.reasoning:
            content += f"---\n\n**DeepSeek 思考过程：**\n\n{reasoning}\n\n"
        content += f"---\n\n**DeepSeek：**\n\n{assistant_msg}\n\n"
        token_stats = {
            'input_tokens': node.input_tokens,
            'output_tokens': node.output_tokens,
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
                result = fetch_webpage(parts[1].strip())
                if result:
                    self.imported_content = result
                    console.print("[green]✅ 网页内容已导入，将在下一次提问时自动附加。[/green]")
            return True

        if cmd_lower.startswith("/imp"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]用法: /imp <文件路径>[/yellow]")
            else:
                result = parse_file(parts[1].strip())
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
            if nid == "main" or nid == self.tree.root.id:
                console.print("[red]不能删除根节点[/red]")
                return True
            node_to_delete = self.tree.nodes[nid]
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
                        tool_result = parse_file(args.get("filepath", ""))
                    elif name == "fetch_webpage":
                        tool_result = fetch_webpage(args.get("url", ""))
                    elif name == "list_directory":
                        tool_result = list_directory(args.get("directory", ""), args.get("show_hidden", False))
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
                            tool_result = web_search(query, freshness, count)
                            console.print(f"[dim]剩余搜索配额: {self.search_quota}[/dim]")
                    elif name == "execute_command":
                        command = args.get("command", "")
                        timeout = args.get("timeout", 30)
                        level, desc, risk, audit_reasoning = audit_command(self.client, command)
                        if audit_reasoning:
                            console.print(f"[dim]🧠 审核思考: {audit_reasoning}[/dim]")
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
                            tool_result = execute_command(command, timeout)
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
