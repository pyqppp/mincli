import os
import re
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
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import (
    Layout, HSplit, VSplit, Float, FloatContainer,
    Window, FormattedTextControl, Dimension,
)
from prompt_toolkit.widgets import Button, Label
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from mincli.config import (
    MODEL_V4_FLASH, MODEL_V4_PRO,
    TEMPERATURE_MIN, TEMPERATURE_MAX,
    PREVIEW_USER_MSG_LEN, PREVIEW_ASSISTANT_MSG_LEN,
    load_mcp_servers, save_mcp_servers, get_mcp_config_path,
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
from mincli.tools.web_fetch import fetch_webpage
from mincli.tools.file_ops import parse_file
from mincli.tools.execute import audit_command, matches_dangerous
try:
    from mincli.mcp_client import McpToolClient
except ImportError:
    McpToolClient = None


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
        self.audit_level: int = 1

        self.tree = ConversationTree(default_system)

        self.history_file = os.path.expanduser("~/.mincli_history")
        kb = KeyBindings()
        @kb.add('enter')
        def _accept(event):
            event.current_buffer.validate_and_handle()
        @kb.add('escape', 'enter')
        def _newline_alt(event):
            event.current_buffer.insert_text('\n')
        @kb.add('c-j')
        def _newline_ctrlj(event):
            event.current_buffer.insert_text('\n')
        @kb.add('escape', '[', '1', '3', ';', '2', 'u')
        def _newline_shift(event):
            event.current_buffer.insert_text('\n')
        self.session = PromptSession(
            history=FileHistory(self.history_file),
            multiline=True,
            key_bindings=kb,
        )

        self.imported_content: Optional[str] = None
        self.temp_dir = tempfile.mkdtemp(prefix="mincli_")
        self.temp_files: Dict[str, str] = {}
        self._load_session()

        self._mcp = None
        self._mcp_tool_names: set = set()
        if McpToolClient is not None:
            self._mcp = McpToolClient()
            try:
                self._mcp.start()
                self._mcp_tool_names = self._mcp.tool_names()
            except Exception as e:
                console.print(f"[red]MCP 客户端初始化失败: {e}[/red]")
                self._mcp = None
        self.llm_tools = TOOLS + (self._mcp.tools() if self._mcp else [])

    def _save_session(self) -> None:
        filepath = self.SAVE_FILE
        try:
            data = {
                "system_prompt": self.current_system,
                "temperature": self.current_temperature,
                "model": self.current_model,
                "thinking_enabled": self.thinking_enabled,
                "reasoning_effort": self.reasoning_effort,
                "audit_level": self.audit_level,
                "tree": self.tree.to_dict(),
                "imported_content": self.imported_content,
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
        self.audit_level = data.get("audit_level", 1)

        tree_data = data.get("tree")
        if tree_data:
            self.tree = ConversationTree.from_dict(tree_data)
        else:
            self.tree = ConversationTree(self.current_system)

        self.imported_content = data.get("imported_content")
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
        console.print(f"[bold]你:[/bold]")
        console.print(user_msg)
        if reasoning:
            console.print(Markdown("\n**DeepSeek 思考过程:**"))
            console.print(reasoning)
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
        active_prefix = self.tree._get_subtree_root_prefix(node.id)
        console.print(self.tree.render_tree(node.id, active_subtree=active_prefix))
        console.print(f"[dim]当前节点: {node.id} ({node.title})[/dim]")

    def _auto_title_subtree(self, node) -> None:
        if not self.tree.root or node.id == "main":
            return
        prefix = self.tree._get_subtree_root_prefix(node.id)
        if not prefix or prefix in self.tree.subtree_titles:
            return
        count = self.tree.count_subtree_nodes(prefix)
        if count == 3:
            root_id = next((nid for nid in self.tree.nodes
                            if nid.startswith(prefix)
                            and self.tree.nodes[nid].parent_id == "main"), None)
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
                    console.print(f"[cyan]📌 已自动为对话树「{prefix}」命名: {title}[/cyan]")
            except Exception:
                pass

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

        if cmd_lower in ["/view"]:
            self._show_current_node()
            return True

        if cmd_lower in ["/help", "/h"]:
            self._show_help()
            return True

        if cmd_lower.startswith("/set"):
            self._handle_set_command(cmd)
            return True

        if cmd_lower.startswith("/mcp"):
            self._handle_mcp_command(cmd)
            return True

        if self.tree and self._handle_tree_command(cmd):
            return True

        if cmd_lower.startswith("/import"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]用法: /import <文件路径或URL>[/yellow]")
            else:
                target = parts[1].strip()
                if re.match(r'^https?://', target):
                    console.print(f"[dim]正在抓取 {target}…[/dim]")
                    result = fetch_webpage(target)
                else:
                    result = parse_file(target)
                if result:
                    self.imported_content = result
                    console.print("[green]✅ 内容已导入，将在下一次提问时自动附加。[/green]")
            return True

        if self.tree:
            m = re.match(r'^/([a-z]+\d+|main)$', cmd_stripped)
            if m and m.group(1) in self.tree.nodes:
                self._jump_to_node(m.group(1))
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
            console.print("[yellow]用法: /set system <提示词>  /set temp <值>  /set model <flash|pro>  /set thinking <on|off>  /set effort <low|high|max>  /set audit <1-4>  /set show[/yellow]")
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
            if arg in ["low", "high", "max"]:
                self.reasoning_effort = arg
                console.print(f"[green]推理强度已设置为: {arg}[/green]")
            else:
                console.print("[yellow]用法: /set effort <low|high|max>[/yellow]")

        elif sub == "audit" and len(parts) == 3:
            try:
                level = int(parts[2])
                if level in (1, 2, 3, 4):
                    self.audit_level = level
                    labels = {1: "最高（AI审核 + 用户确认）", 2: "中等（AI审核，低风险自动执行）",
                              3: "最低（文本匹配，高风险询问）", 4: "无（直接执行）"}
                    console.print(f"[green]审核层级已设置为: {labels[level]}[/green]")
                else:
                    console.print("[yellow]审核层级须为 1-4[/yellow]")
            except ValueError:
                console.print("[yellow]审核层级须为数字 1-4[/yellow]")

        elif sub == "show":
            self._show_config()
        else:
            console.print("[yellow]用法: /set system <提示词>  /set temp <值>  /set model <flash|pro>  /set thinking <on|off>  /set effort <low|high|max>  /set audit <1-4>  /set show[/yellow]")
            return

    def _show_config(self) -> None:
        console.print(f"[cyan]系统提示词: {self.current_system}[/cyan]")
        console.print(f"[cyan]温度: {self.current_temperature}[/cyan]")
        console.print(f"[cyan]模型: {self.current_model}[/cyan]")
        console.print(f"[cyan]思考模式: {'开' if self.thinking_enabled else '关'} | 推理强度: {self.reasoning_effort}[/cyan]")
        audit_labels = {1: "最高（AI审核 + 用户确认）", 2: "中等（AI审核，低风险自动执行）",
                        3: "最低（文本匹配，高风险询问）", 4: "无（直接执行）"}
        console.print(f"[cyan]审核层级: {self.audit_level} - {audit_labels[self.audit_level]}[/cyan]")
        if self.tree and self.tree.current_node:
            console.print(f"[cyan]当前节点: {self.tree.current_node.id} ({self.tree.current_node.title})[/cyan]")

    def _handle_mcp_command(self, cmd: str) -> None:
        parts = cmd.strip().split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        if sub in ("", "list", "ls", "status", "show"):
            self._show_mcp_config()
        elif sub == "add":
            self._mcp_add(rest)
        elif sub in ("remove", "rm", "del"):
            name = rest.split()[0] if rest.split() else ""
            if not name:
                console.print("[yellow]用法: /mcp remove <名称>[/yellow]")
            else:
                self._mcp_remove(name)
        elif sub == "reload":
            self._mcp_reload()
        else:
            console.print("[yellow]用法: /mcp list  /mcp add <名称> <命令|URL> [参数...]  /mcp remove <名称>  /mcp reload[/yellow]")

    def _show_mcp_config(self) -> None:
        from rich.table import Table
        console.print(f"[cyan]MCP 配置文件: {get_mcp_config_path()}[/cyan]")
        status = self._mcp.server_status() if self._mcp else {}
        servers = load_mcp_servers()
        if not status:
            console.print("[dim]（MCP 客户端未就绪）[/dim]")
            return
        table = Table(title="MCP Servers")
        table.add_column("名称", style="cyan")
        table.add_column("命令", style="green")
        table.add_column("工具数", style="yellow")
        table.add_column("状态", style="magenta")
        for name in sorted(status):
            st = status[name]
            if name == "mincli":
                cmd = "内置 server"
            else:
                cfg = servers.get(name, {})
                cmd = cfg.get("url") or cfg.get("command", "")
            table.add_row(name, cmd, str(st["tools"]), "✅ 已连接" if st["connected"] else "⚠ 未连接")
        console.print(table)
        if not servers:
            console.print("[dim]（未配置第三方 server，可用 /mcp add 添加）[/dim]")

    def _mcp_add(self, rest: str) -> None:
        servers = load_mcp_servers()
        is_url = lambda s: bool(re.match(r'^https?://', s))

        tokens = rest.split()
        if len(tokens) >= 2:
            name = tokens[0]
            target = tokens[1]
            command, args, env = None, [], {}
            if not is_url(target):
                command, args = target, tokens[2:]
        else:
            name = tokens[0] if tokens else ""
            if not name:
                console.print("[yellow]用法: /mcp add <名称> <命令> [参数...] 或 /mcp add <名称> <URL>[/yellow]")
                return
            try:
                target = self.session.prompt(f"MCP「{name}」命令或 URL> ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("[dim]已取消[/dim]")
                return
            except Exception:
                console.print("[yellow]无法进入交互输入，请直接使用: /mcp add <名称> <命令> [参数...] 或 /mcp add <名称> <URL>[/yellow]")
                return
            if is_url(target):
                command, args, env = None, [], {}
            else:
                command = target
                try:
                    args_raw = self.session.prompt("参数（空格分隔，回车跳过）> ").strip()
                except (KeyboardInterrupt, EOFError, Exception):
                    args_raw = ""
                args = args_raw.split() if args_raw else []
                env = {}
                while True:
                    try:
                        line = self.session.prompt("环境变量 KEY=VALUE（回车结束）> ").strip()
                    except (KeyboardInterrupt, EOFError, Exception):
                        break
                    if not line:
                        break
                    if "=" in line:
                        k, _, v = line.partition("=")
                        env[k.strip()] = v.strip()
                    else:
                        console.print("[yellow]格式须为 KEY=VALUE[/yellow]")

        if not name or (not command and not is_url(target)):
            console.print("[yellow]缺少命令或 URL[/yellow]")
            return

        if name in servers:
            console.print(f"[yellow]已存在同名 server「{name}」，将被覆盖[/yellow]")
        if is_url(target):
            servers[name] = {"url": target}
        else:
            entry = {"command": command}
            if args:
                entry["args"] = args
            if env:
                entry["env"] = env
            servers[name] = entry
        path = save_mcp_servers(servers)
        console.print(f"[green]✅ 已保存到 {path}，运行 /mcp reload 生效[/green]")

    def _mcp_remove(self, name: str) -> None:
        servers = load_mcp_servers()
        if name not in servers:
            console.print(f"[red]未找到 server「{name}」[/red]")
            return
        if not self._confirm("移除 MCP server", f"确定要移除「{name}」吗？"):
            console.print("[dim]取消移除[/dim]")
            return
        del servers[name]
        path = save_mcp_servers(servers)
        console.print(f"[green]✅ 已移除「{name}」，运行 /mcp reload 生效[/green]")

    def _mcp_reload(self) -> None:
        if not self._mcp:
            console.print("[red]MCP 客户端未就绪[/red]")
            return
        console.print("[dim]正在重新加载 MCP servers…[/dim]")
        try:
            self._mcp.reload()
            self._mcp_tool_names = self._mcp.tool_names()
            self.llm_tools = TOOLS + self._mcp.tools()
            console.print("[green]✅ MCP 已重新加载[/green]")
        except Exception as e:
            console.print(f"[red]MCP 重载失败: {e}[/red]")

    def _show_help(self) -> None:
        from rich.table import Table
        from rich.panel import Panel

        basic = Table(show_header=False, box=None, padding=(0, 2))
        basic.add_column("cmd", style="cyan")
        basic.add_column("desc")
        basic.add_row("/exit, /quit, /q, /e", "退出程序（自动保存会话）")
        basic.add_row("/clear, /c", "清空当前会话")
        basic.add_row("/help, /h", "显示此帮助")
        basic.add_row("/import <路径或URL>", "导入文件或抓取网页")
        basic.add_row("/mcp <list|add|remove|reload>", "管理第三方 MCP server")
        basic.add_row("/view", "用编辑器打开当前回答")

        set_cmds = Table(show_header=False, box=None, padding=(0, 2))
        set_cmds.add_column("cmd", style="cyan")
        set_cmds.add_column("desc")
        set_cmds.add_row("/set system <提示词>", "修改系统提示词")
        set_cmds.add_row("/set temp <值>", "设置温度（0.0~2.0）")
        set_cmds.add_row("/set model <flash|pro>", "切换模型")
        set_cmds.add_row("/set thinking <on|off>", "开关思考模式")
        set_cmds.add_row("/set effort <low|high|max>", "推理强度")
        set_cmds.add_row("/set audit <1-4>", "审核层级（默认 1）")
        set_cmds.add_row("/set show", "显示当前配置")

        tree_cmds = Table(show_header=False, box=None, padding=(0, 2))
        tree_cmds.add_column("cmd", style="cyan")
        tree_cmds.add_column("desc")
        tree_cmds.add_row("/<节点ID>（如 /a3）", "直接跳转到指定节点")
        tree_cmds.add_row("/tree", "列出所有节点")
        tree_cmds.add_row("/info [节点ID]", "查看节点详情")
        tree_cmds.add_row("/up", "返回父节点")
        tree_cmds.add_row("/home", "跳回根节点")
        tree_cmds.add_row("/save [节点ID]", "导出节点为 Markdown")
        tree_cmds.add_row("/delete <节点ID>", "删除节点及其子节点")

        keys = Table(show_header=False, box=None, padding=(0, 2))
        keys.add_column("key", style="cyan")
        keys.add_column("desc")
        keys.add_row("Enter", "发送消息")
        keys.add_row("Ctrl+J", "插入换行")
        keys.add_row("Alt+Enter", "插入换行")
        keys.add_row("Ctrl+C", "中断/退出")

        content = Table.grid(padding=(0, 1))
        content.add_column()
        content.add_row("[bold]基本命令[/bold]")
        content.add_row(basic)
        content.add_row("")
        content.add_row("[bold]配置命令[/bold]")
        content.add_row(set_cmds)
        content.add_row("")
        content.add_row("[bold]树状命令[/bold]")
        content.add_row(tree_cmds)
        content.add_row("")
        content.add_row("[bold]快捷键[/bold]")
        content.add_row(keys)

        console.print(Panel(content, title="📖 帮助", border_style="cyan"))

    def _jump_to_node(self, node_id: str) -> None:
        if self.tree.switch_to_node(node_id):
            bt = self.tree.get_branch_total_tokens(self.tree.current_node.id)
            self._display_tree_node(self.tree.current_node, bt)
            console.print("\n[bold green]--- 已切换节点 ---[/bold green]\n")
        else:
            console.print("[red]未找到该节点ID[/red]")

    def _handle_tree_command(self, cmd: str) -> bool:
        parts = cmd.split()
        cmd_lower = parts[0].lower()

        if cmd_lower == "/tree":
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

        if cmd_lower == "/up":
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

        if cmd_lower == "/home":
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

        if cmd_lower.startswith("/delete"):
            nid = parts[1] if len(parts) > 1 else None
            if nid is None:
                console.print("[yellow]用法: /delete <节点ID>[/yellow]")
                return True
            if nid not in self.tree.nodes:
                console.print(f"[red]未找到节点 {nid}[/red]")
                return True
            if nid == "main" or nid == self.tree.root.id:
                console.print("[red]不能删除根节点[/red]")
                return True
            node_to_delete = self.tree.nodes[nid]
            if not self._confirm("删除节点", f"确定要删除节点 {nid} 及其所有子节点吗？"):
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
            if tool_messages:
                clear_screen()
            sr = stream_response(
                self.client, messages, self.current_model,
                self.current_temperature, user_input,
                thinking_enabled=self.thinking_enabled,
                reasoning_effort=self.reasoning_effort,
                tools=self.llm_tools,
            )

            content, reasoning, in_tok, out_tok, tool_calls = (
                sr.content, sr.reasoning, sr.input_tokens, sr.output_tokens, sr.tool_calls
            )
            if reasoning:
                accumulated_reasoning += ("\n[dim]" if accumulated_reasoning else "[dim]") + reasoning + "[/dim]"
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

                    console.print(f"[bright_black]▸ {name}[/bright_black]")

                    if name == "query_conversation_tree":
                        tool_result = self._query_conversation_tree(args.get("root", ""))
                    elif name == "read_conversation_nodes":
                        tool_result = self._read_conversation_nodes(args.get("node_ids", ""))
                    elif name == "write_file":
                        tool_result = self._write_file(args.get("filepath", ""), args.get("content", ""))
                    elif name == "edit_file":
                        tool_result = self._edit_file(args.get("filepath", ""), args.get("old_string", ""), args.get("new_string", ""))
                    elif name == "execute_command":
                        tool_result = self._execute_command_tool(args)
                    elif name in self._mcp_tool_names:
                        tool_result = self._mcp_call(name, args)
                    else:
                        tool_result = f"未知工具: {name}"

                    summary = (tool_result or "").strip()[:100].replace("\n", " ")
                    console.print(f"[bright_black]  └─ {summary}[/bright_black]")

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

                    args_repr = json.dumps(args, ensure_ascii=False)
                    accumulated_reasoning += f"\n[cyan]▸ {name}({args_repr})[/cyan]"
                    if tool_result:
                        summary = tool_result.strip()[:200].replace("\n", " ")
                        accumulated_reasoning += f"\n[cyan]  └─ {summary}{'…' if len(tool_result.strip()) > 200 else ''}[/cyan]"

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

        title = generate_conversation_title(self.client, user_input)
        if not self.tree.root:
            node = self.tree.create_root(user_input, final_answer, final_reasoning, title, accumulated_in_tok, accumulated_out_tok)
        else:
            node = self.tree.add_child(
                self.tree.current_node, user_input, final_answer, final_reasoning, title, accumulated_in_tok, accumulated_out_tok
            )
        if tool_messages:
            node.tool_messages = tool_messages
        self.tree.current_node = node
        self._auto_title_subtree(node)
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
            if self._mcp:
                self._mcp.close()

    def _get_prompt_text(self) -> str:
        if self.tree and self.tree.current_node:
            return f"[{self.tree.current_node.id}] 你: "
        return "你: "

    def _show_welcome(self) -> None:
        from rich.table import Table

        clear_screen()
        cmd_table = Table.grid(padding=(0, 3))
        cmd_table.add_column()
        cmd_table.add_row("[bold]基本[/bold]  /import  /mcp  /clear  /exit")
        cmd_table.add_row("[bold]配置[/bold]  /set system /temp /model /thinking /effort /audit")
        cmd_table.add_row("[bold]树状[/bold]  /<ID>跳转 /tree /info /up /home /save /delete")
        cmd_table.add_row("")
        cmd_table.add_row("[dim]输入 /help 查看完整命令说明[/dim]")

        thinking_status = f"思考{'开' if self.thinking_enabled else '关'}"
        audit_labels = {1: "最高", 2: "中等", 3: "最低", 4: "无"}
        status_line = f"[bold]{self.current_model}[/bold] \u00b7 {thinking_status} \u00b7 审核{self.audit_level}({audit_labels[self.audit_level]})"

        content = Table.grid(padding=(0, 1))
        content.add_column()
        content.add_row(f"[bold]mincli 树状对话模式[/bold]")
        content.add_row("")
        content.add_row(cmd_table)
        content.add_row("")
        content.add_row(f"\u2007{status_line}")
        content.add_row("  Enter \u2192 发送  Ctrl+J \u2192 换行")
        content.add_row("")
        content.add_row("[dim]等待第一个问题...[/dim]")

        console.print(Panel(content, border_style="bright_cyan"))

    def _confirm(self, title: str = "确认执行", text: str = "是否执行？") -> bool:
        result = [False]
        app_ref = [None]

        def done(val: bool):
            result[0] = val
            if app_ref[0]:
                app_ref[0].exit()

        TL, TR, H, V, BL, BR = '╭', '╮', '─', '│', '╰', '╯'
        bs = 'class:dialog.border'
        vs = Window(width=1, char=V, style=bs)

        yes_btn = Button(" 是 ", handler=lambda: done(True), left_symbol="", right_symbol="")
        no_btn = Button(" 否 ", handler=lambda: done(False), left_symbol="", right_symbol="")

        def body_row(content, height=None):
            return VSplit([vs, Window(width=1), content, Window(width=1), vs], height=height)

        body = HSplit([
            VSplit([
                Window(width=1, height=1, char=TL, style=bs),
                Window(char=H, style=bs),
                Window(FormattedTextControl(HTML(f"<b> {title} </b>")),
                       style='class:dialog.title', dont_extend_width=True),
                Window(char=H, style=bs),
                Window(width=1, height=1, char=TR, style=bs),
            ], height=1),
            body_row(Window(height=1)),
            body_row(Label(text=text, style='class:dialog.text')),
            body_row(Window(height=1)),
            body_row(VSplit([
                Window(),
                yes_btn,
                Window(width=3),
                no_btn,
                Window(),
            ]), height=1),
            body_row(Window(height=1)),
            VSplit([
                Window(width=1, height=1, char=BL, style=bs),
                Window(char=H, style=bs),
                Window(width=1, height=1, char=BR, style=bs),
            ], height=1),
        ])

        root = FloatContainer(
            content=Window(FormattedTextControl('')),
            floats=[Float(content=body, allow_cover_cursor=True)],
        )

        dlg_kb = KeyBindings()
        @dlg_kb.add('tab')
        def _(event):
            event.app.layout.focus_next()
        @dlg_kb.add('s-tab')
        def _(event):
            event.app.layout.focus_previous()
        @dlg_kb.add('right')
        def _(event):
            event.app.layout.focus_next()
        @dlg_kb.add('left')
        def _(event):
            event.app.layout.focus_previous()

        dlg_style = Style([
            ("dialog.title", "bg:default fg:ansicyan bold"),
            ("dialog.text", "bg:default fg:white"),
            ("dialog.border", "bg:default fg:ansicyan"),
            ("button", "bg:default fg:ansicyan"),
            ("button.focused", "bg:ansicyan fg:white bold"),
        ])

        try:
            app = Application(
                layout=Layout(root),
                key_bindings=dlg_kb,
                style=dlg_style,
                full_screen=True,
            )
            app_ref[0] = app
            app.run()
        except Exception:
            return False
        return result[0]

    def _mcp_call(self, name: str, args: dict) -> str:
        if not self._mcp:
            return f"工具不可用（MCP 未就绪）: {name}"
        return self._mcp.call(name, args)

    def _execute_command_tool(self, args: dict) -> str:
        command = args.get("command", "")
        timeout = args.get("timeout", 30)
        call_args = {"command": command, "timeout": timeout}

        if self.audit_level == 4:
            console.print(f"[bright_black]▸ execute_command（无审核）[/bright_black]")
            return self._mcp_call("execute_command", call_args)

        if self.audit_level == 3:
            if matches_dangerous(command):
                if self._confirm("高危命令", f"命令: {command}\n\n⚠️ 匹配到高危命令模式，确认执行？"):
                    return self._mcp_call("execute_command", call_args)
                return "用户未确认执行此命令"
            console.print(f"[bright_black]▸ execute_command（文本审核通过）[/bright_black]")
            return self._mcp_call("execute_command", call_args)

        if self.audit_level == 2:
            level, desc, risk, audit_reasoning = audit_command(self.client, command)
            if level <= 2:
                console.print(f"[bright_black]▸ {desc}（等级{level}/5，自动执行）[/bright_black]")
                return self._mcp_call("execute_command", call_args)
            risk_text = f"\n⚠️ {risk}" if risk else ""
            if self._confirm("执行确认", f"命令: {command}\n\n审核: 等级 {level}/5 | {desc}{risk_text}"):
                return self._mcp_call("execute_command", call_args)
            return "用户未确认执行此命令"

        level, desc, risk, audit_reasoning = audit_command(self.client, command)
        if audit_reasoning:
            console.print(f"[dim]🧠 审核思考: {audit_reasoning}[/dim]")
        risk_text = f"\n⚠️ {risk}" if risk else ""
        if self._confirm("执行确认", f"命令: {command}\n\n审核: 等级 {level}/5 | {desc}{risk_text}"):
            return self._mcp_call("execute_command", call_args)
        return "用户未确认执行此命令"

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
        title_text = f"即将{'覆盖' if exists else '写入'}文件"
        if not self._confirm(title_text, details):
            return "用户已取消操作"
        return self._mcp_call("write_file", {"filepath": filepath, "content": content})

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

        if not self._confirm("即将修改文件", details):
            return "用户已取消操作"
        return self._mcp_call("edit_file", {
            "filepath": filepath,
            "old_string": old_string,
            "new_string": new_string,
        })

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
            root_id = next((nid for nid in self.tree.nodes
                            if nid.startswith(root)
                            and self.tree.nodes[nid].parent_id == "main"), None)
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
