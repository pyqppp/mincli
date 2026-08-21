"""mincli ChatApp 主应用（阶段 3：斜杠命令迁移到 TUI）。

运行：`venv/bin/python -m mincli.tui.app`（需要真实终端 + DEEPSEEK_API_KEY）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time

# 必须在 Textual Markdown 组件创建解析器之前执行：
# 防御 markdown-it-py 解析极端输入（引用块内表格被流式截断等）时的越界崩溃
from mincli.markdown_safe import _patch_markdown_it

_patch_markdown_it()

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Markdown, Static, Tree
from textual.widgets._markdown import MarkdownBlockQuote

# 防御 Textual 选区提取越界：流式渲染会重建 Markdown 块，拖选跨越重建
# 瞬间时，锚点行号可能等于/超过新内容行数 → Selection.extract 直接索引
# 越界崩溃（这也是历史版本用 ALLOW_SELECT=False 禁用整个选区的根本原因；
# 现在从源头修掉，选区可正常使用）。
from textual.selection import Selection as _TextualSelection


def _patch_textual_selection() -> None:
    """把 Selection.extract 包一层安全钳制（幂等）。"""
    if getattr(_TextualSelection, "_mincli_safe_extract", False):
        return
    _orig_extract = _TextualSelection.extract

    def _safe_extract(selection, text: str) -> str:
        try:
            return _orig_extract(selection, text)
        except IndexError:
            # 内容在选中期间被重建/缩短：退化为返回当前全部文本，而不是崩溃
            try:
                return "\n".join(text.splitlines())
            except Exception:
                return ""

    _TextualSelection.extract = _safe_extract
    _TextualSelection._mincli_safe_extract = True


_patch_textual_selection()

from mincli.config import (
    DEFAULT_SYSTEM_PROMPT,
    MODEL_V4_FLASH,
    COMPACT_DEFAULT_KEEP,
    BALANCE_REFRESH_SECONDS,
    PREVIEW_ASSISTANT_MSG_LEN,
    PREVIEW_USER_MSG_LEN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    MODELS_AVAILABLE,
    API_PROVIDERS,
    load_models,
    register_model,
    get_mcp_config_path,
    load_mcp_servers,
    save_mcp_servers,
)
from mincli.controller import AUDIT_LABELS, ChatController, ControllerEvent
from mincli.tui.confirm import ConfirmScreen
from mincli.tui.widgets import ChatInput

WELCOME = """# mincli

DeepSeek 树状对话 TUI

- **左侧**：对话树（点击节点切换，点小三角收起/展开）
- **中间**：消息流（Markdown 流式渲染）
- **底部**：多行输入框（**Enter** 发送，**Ctrl+J** 换行）
- **Ctrl+C**：退出（自动保存会话）

直接输入问题开始对话，输入 `/help` 查看命令。
"""

# 命令补全/提示用元数据：命令 → 帮助文本（首行为简要说明）
COMMAND_HELP: dict[str, str] = {
    "/exit": "退出程序（自动保存会话）",
    "/clear": "清空当前会话",
    "/compact": f"用法: /compact [保留轮数] | /compact off\n把当前分支早期对话压缩成详细摘要（默认保留最近 {COMPACT_DEFAULT_KEEP} 轮原文，0=全部压缩；off 清除压缩恢复原文）",
    "/help": "显示此帮助",
    "/import": "用法: /import <文件路径或URL>\n导入文件或抓取网页，下次提问自动附加到上下文",
    "/view": "用编辑器打开当前回答",
    "/mcp": "用法: /mcp list | /mcp add <名称> <命令|URL> [参数...] [--header 'K: V'] | /mcp remove <名称> | /mcp reload\n管理第三方 MCP server（--header 仅对远程 server 生效，可重复使用）",
    "/model": "用法: /model list | /model register <模型名> <URL> [-p provider] [-k key_var]\n列出/注册模型配置（注册后可用 /set model <模型名> 切换）",
    "/set": "用法: /set system <提示词> | /set temp <值> | /set model <flash|pro|模型名> | /set thinking <on|off> | /set effort <low|high|max> | /set audit <1-4> | /set workspace <路径> | /set show\n修改运行配置",
    "/tree": "显示完整对话树",
    "/info": "用法: /info [节点ID]\n查看节点详情（默认当前节点）",
    "/up": "返回父节点",
    "/home": "跳回根节点",
    "/full": "切换全览模式：节点树全宽显示（切换节点自动退出）\n隐藏右侧回答区、输入框保留；再按一次 /full 恢复分栏",
    "/reasoning": "展开/折叠当前消息的思考过程\n正文开始后思考过程会自动折叠成一行，点击灰色折叠块也可展开",
    "/save": "用法: /save [节点ID]\n导出节点为 Markdown 文件",
    "/delete": "用法: /delete <节点ID>\n删除节点及其所有子节点（需确认）",
}

# 思考过程折叠后的一行占位（块引用，灰色、可点击展开）
REASONING_COLLAPSED_MD = "> ▶ 思考过程（点击展开 · /reasoning）"

# 青色主题：整体围绕青色设计（对应原命令行版的青色风格），
# 回答区背景近黑灰色、整体色相略偏蓝。
# 默认主题的 $accent 是橙色（输入框/弹窗边框）、滚动条背景是纯黑（ansi_black），
# 这里统一改为：主色青蓝、强调色亮青色、滑条深青蓝。
MINCLI_THEME = Theme(
    name="mincli-cyan",
    primary="#00d0dc",       # 主色：青蓝（选中/链接/边框/按钮）
    secondary="#008394",     # 辅助：深青蓝
    accent="#00e5ff",        # 强调：亮青色（输入框聚焦边框 / 弹窗边框）
    warning="#ffb454",
    error="#ff6b81",
    success="#3dd68c",
    foreground="#ffffff",    # 正文文字：纯白（树/输入框/AI 回答不再偏青）
    background="#0a0d10",    # 全局背景（回答区）：近黑灰色、微带蓝
    surface="#101a20",       # 侧栏/输入框/弹窗面板：深蓝灰
    panel="#15222b",
    variables={
        # 正文纯白（部分控件走 $text）
        "text": "#ffffff",
        # 滚动条：纯黑背景 → 深青蓝；滑块 → 青蓝
        "scrollbar": "#1e6272",
        "scrollbar-hover": "#2b8494",
        "scrollbar-active": "#00e5ff",
        "scrollbar-background": "#07141a",
        "scrollbar-background-hover": "#0a1c24",
        "scrollbar-background-active": "#0a1c24",
        "scrollbar-corner-color": "#07141a",
        # 边框：未聚焦也从纯黑改为深青蓝
        "border": "#00dce6",
        "border-blurred": "#14323a",
        # 光标与选区
        "block-cursor-background": "#00e5ff",
        "block-cursor-foreground": "#041114",
        "block-cursor-text-style": "none",
        "input-selection-background": "#00e5ff 35%",
        "screen-selection-background": "#00e5ff 35%",
        # Markdown 标题（围绕青蓝）
        "markdown-h1-color": "#00e5ff",
        "markdown-h1-text-style": "bold",
        "markdown-h2-color": "#00d0dc",
        "markdown-h2-text-style": "underline",
        "markdown-h3-color": "#00d0dc",
        "markdown-h4-color": "#00d0dc",
        "markdown-h5-color": "#00d0dc",
        "markdown-h6-color": "#00d0dc",
        # 链接与 Footer
        "link-color": "#00e5ff",
        "link-color-hover": "#7ff3ff",
        "footer-key-foreground": "#00e5ff",
    },
)


class ChatApp(App):
    """mincli 主界面：会话树 + 消息流 + 多行输入。"""

    TITLE = "mincli"
    SUB_TITLE = "DeepSeek Chat"

    CSS_PATH = "chat.tcss"

    BINDINGS = [
        # Ctrl+C 在所有平台统一：有选中文字时先复制（Textual 屏幕级绑定
        # 优先，ChatInput/输入框选区、聊天区选区均可复制），无选中时退出
        Binding("ctrl+c", "quit", "退出"),
        Binding(
            "caps_lock,num_lock,scroll_lock",
            "ignore_lock",
            "忽略锁定键",
            show=False,
            priority=True,
        ),
    ]

    def __init__(self, controller: ChatController | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_theme(MINCLI_THEME)
        self._injected_controller = controller
        self.ctrl: ChatController | None = None
        self._stream_active = False
        self._reasoning_text = ""  # 当前消息的思考过程全文（折叠后用于展开）
        self._reasoning_md = ""  # 消息区中思考块的当前 Markdown 原文（用于折叠/展开替换）
        self._reasoning_collapsed = False  # 思考块是否已折叠
        self._answer_started = False  # 正文的 "**mincli：**" 头部是否已输出
        self._full_view = False  # 全览模式：节点树全宽（隐藏回答区）
        self._last_scroll_t = 0.0  # 上下键滚动：上次按键时间（用于双击加速）
        self._scroll_fast_until = 0.0  # 双击按住 → 2 倍速滚动截止时间
        self._chat_lock = asyncio.Lock()  # 串行化聊天区 update/append（点击折叠 vs 流式）
        self._completion_matches: list[str] = []
        self._completion_index = 0
        # 流式渲染节流：SSE 按 token 级产生事件，逐事件 append 会导致
        # Markdown 组件反复 mount/布局/重绘，主线程跟不上 → 渲染卡顿。
        # 这里把增量累积到缓冲，每 80ms 批量渲染一次，大幅降低渲染次数。
        self._stream_buf_content = ""  # 待渲染的正文增量缓冲
        self._stream_buf_reasoning = ""  # 待渲染的思考增量缓冲
        self._flush_interval = 0.08  # 批量渲染间隔（秒）
        self._flush_timer = None  # 批量渲染定时器（textual Timer）
        self._balance_txt: Optional[str] = None  # 最近一次拉取的账户余额（字符串）

    def action_ignore_lock(self) -> None:
        """忽略锁定键（Caps Lock / Num Lock / Scroll Lock）。"""
        pass

    def copy_to_clipboard(self, text: str) -> None:
        """复制文本到系统剪贴板。

        Textual 默认通过 OSC52 转义序列写剪贴板，macOS 的 Terminal.app
        不支持该序列（这也是历史版本"能选择但不能拷贝"的原因）；这里在
        macOS 上额外调用 pbcopy 写入系统剪贴板，其他平台走 Textual 默认。
        复制快捷键：所有平台统一按 Ctrl+C（有选中文本时复制，无选中时退出）。
        """
        super().copy_to_clipboard(text)
        if sys.platform == "darwin":
            try:
                subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
            except Exception:
                pass
        if text:
            self.notify(f"已复制 {len(text)} 字符")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                with Horizontal(id="sidebar-header"):
                    yield Static("会话", id="sidebar-title")
                    yield Button("⛶ 全览", id="fullview-btn", compact=True)
                yield Tree("全部", id="tree")
            yield Markdown(WELCOME, id="chat-log")
        with Vertical(id="cmd-popup"):
            yield Static("", id="cmd-popup-body")
        yield ChatInput(
            id="chat-input", placeholder="输入消息，Enter 发送，Ctrl+J 换行"
        )
        with Horizontal(id="usage-bar"):
            yield Static("", id="usage-left")
            yield Static("", id="usage-right")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "mincli-cyan"  # 应用青色主题
        if self._injected_controller is not None:
            self.ctrl = self._injected_controller
        else:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                self.notify("未设置 DEEPSEEK_API_KEY（请在 .env 或环境变量中配置）", severity="error", timeout=10)
                return
            from openai import OpenAI
            self.ctrl = ChatController(
                client=OpenAI(api_key=api_key, base_url="https://api.deepseek.com"),
                default_system=DEFAULT_SYSTEM_PROMPT,
                default_temperature=1.0,
                default_model=MODEL_V4_FLASH,
            )
        self.ctrl.confirm = self._confirm
        self.query_one("#tree", Tree).auto_expand = False  # 点击节点名只切换节点，不收起/展开
        self._rebuild_tree()
        self.query_one("#chat-input", ChatInput).focus()
        if self.ctrl.session_loaded:
            self.notify("已加载上次会话记录", timeout=4)
        self._start_balance_refresh()
        self._refresh_usage_bar()

    def on_unmount(self) -> None:
        self._cancel_flush()
        if self.ctrl is not None:
            self.ctrl.save_session()
            self.ctrl.close()

    # ---------------- 输入栏状态条（缓存命中率 / 余额 / 下次输入估算） ----------------

    def _start_balance_refresh(self) -> None:
        """启动账户余额定时刷新（首次立即拉取一次）。"""
        self._balance_txt = None
        self.set_interval(BALANCE_REFRESH_SECONDS, self._refresh_balance)
        self._refresh_balance()

    def _refresh_balance(self) -> None:
        """定时回调：后台线程拉取余额。"""
        if self.ctrl is None:
            return
        self.run_worker(
            self._fetch_balance_thread,
            thread=True,
            exit_on_error=False,
            name="balance-fetch",
        )

    def _fetch_balance_thread(self) -> None:
        try:
            info = self.ctrl.fetch_balance()
            total = info.get("total_balance") if info else None
        except Exception:
            total = None
        self.call_from_thread(self._set_balance, total)

    def _set_balance(self, total: Optional[str]) -> None:
        self._balance_txt = total
        self._refresh_usage_bar()

    def _refresh_usage_bar(self) -> None:
        """刷新输入栏下方状态条（左：缓存命中率+余额；右：下次输入估算）。"""
        if self.ctrl is None:
            return
        stats = self.ctrl.usage_stats()
        rate = stats["cache_hit_rate"]
        rate_txt = f"{rate * 100:.0f}%" if rate is not None else "--"
        if self._balance_txt is None:
            bal_txt = "…"
        else:
            bal_txt = f"¥{self._balance_txt}"
        self.query_one("#usage-left", Static).update(
            f"🎯 缓存命中 {rate_txt}   💵 余额 {bal_txt}"
        )
        tokens = stats["next_input_tokens"]
        price = stats["estimated_price"]
        price_txt = f"≈ ¥{price:.4f}" if price is not None else "--"
        peak_txt = "高峰" if stats["peak"] else "空闲"
        self.query_one("#usage-right", Static).update(
            f"⏭ 下次输入 {tokens:,} tok {price_txt}（{peak_txt}价）"
        )

    # ---------------- 对话树侧栏 ----------------

    def _rebuild_tree(self) -> None:
        tree_w = self.query_one("#tree", Tree)
        tree_w.clear()
        root = self.ctrl.tree.root if self.ctrl else None
        if root is None:
            tree_w.root.label = "（空）"
            return
        current_id = self.ctrl.tree.current_node.id if self.ctrl.tree.current_node else None
        tree_w.root.label = f"main: {root.title}"
        tree_w.root.data = "main"
        tree_w.root.expand()
        for child in root.children:
            self._add_tree_node(tree_w.root, child, current_id)

    def _add_tree_node(self, parent, node, current_id) -> None:
        if node.id == current_id:
            label = f"➤ {node.id}: {node.title}"
        else:
            label = f"{node.id}: {node.title}"
        n = parent.add(label, data=node.id)
        n.expand()
        for child in node.children:
            self._add_tree_node(n, child, current_id)

    def _find_tree_node(self, parent, node_id: str):
        if parent.data == node_id:
            return parent
        for child in parent.children:
            found = self._find_tree_node(child, node_id)
            if found is not None:
                return found
        return None

    def _select_tree_node(self, node_id: str) -> None:
        """树中选中并滚动到指定节点（用 move_cursor，避免触发 NodeSelected 打断流式输出）。"""
        tree_w = self.query_one("#tree", Tree)
        tn = self._find_tree_node(tree_w.root, node_id)
        if tn is not None:
            tree_w._tree_lines  # 强制重建行索引，确保 node._line 有效
            tree_w.move_cursor(tn, animate=False)
            tree_w.scroll_to_node(tn, animate=False)

    def _shrink_lists(self, chat: Markdown) -> None:
        """MarkdownBlock 默认 expand=True，列表块会被拉伸成整屏高度（项目间出现大间隔），
        每次渲染后关闭列表块的 expand。"""
        from textual.widgets._markdown import MarkdownList

        for w in chat.query("*"):
            if isinstance(w, MarkdownList) and w.expand:
                w.expand = False

    def _node_content(self, node) -> str:
        """把节点渲染为消息区 Markdown 内容（思考过程内联在提问与回答之间）。"""
        content = f"# {node.id}: {node.title}\n\n**你：**\n\n{node.user_msg}\n\n"
        if node.reasoning:
            content += "\n\n" + self._build_reasoning_md(node.reasoning) + "\n\n"
        content += (
            f"**mincli：**\n\n{node.assistant_msg}\n\n---\n\n"
            f"*📊 输入 {node.input_tokens} tokens | 输出 {node.output_tokens} tokens*"
        )
        return content

    # ---------------- 思考过程（灰色块引用、正文开始后自动折叠） ----------------

    @staticmethod
    def _build_reasoning_md(text: str) -> str:
        """把思考全文转成灰色块引用 Markdown（含折叠头部）。"""
        lines = ["> ▼ 思考过程（点击折叠）", ">"]
        for ln in text.splitlines():
            lines.append(f"> {ln}" if ln else ">")
        return "\n".join(lines)

    @staticmethod
    def _reasoning_chunk_md(chunk: str) -> str:
        """流式思考增量 → 块引用行增量：只按 chunk 内真实换行断行，
        跨 chunk 直接拼接（避免每 token 断行）。"""
        md = ""
        for i, ln in enumerate(chunk.split("\n")):
            if i == 0:
                md += ln
            else:
                md += ("\n> " + ln) if ln else "\n>"
        return md

    async def _collapse_reasoning(self) -> None:
        """正文开始：把灰色思考块替换为一行折叠占位。

        仅在 _handle_event_inner 内调用（此时已持有 _chat_lock）。
        """
        chat = self.query_one("#chat-log", Markdown)
        src = chat.source
        if self._reasoning_md and self._reasoning_md in src:
            new = "\n\n" + REASONING_COLLAPSED_MD
            src = src.replace(self._reasoning_md, new)
            await chat.update(src)
            self._reasoning_md = new
        self._reasoning_collapsed = True
        chat.scroll_end(animate=False)

    async def _toggle_reasoning_async(self) -> None:
        """折叠/展开思考块（点击折叠块或 /reasoning）；与流式渲染串行化。"""
        async with self._chat_lock:
            await self._toggle_reasoning_inner()

    async def _toggle_reasoning_inner(self) -> None:
        if not self._reasoning_md:
            self.notify("当前没有思考过程", severity="warning")
            return
        chat = self.query_one("#chat-log", Markdown)
        src = chat.source
        if self._reasoning_md not in src:
            return
        if self._reasoning_collapsed:
            new = "\n\n" + self._build_reasoning_md(self._reasoning_text)
        else:
            new = "\n\n" + REASONING_COLLAPSED_MD
        src = src.replace(self._reasoning_md, new)
        await chat.update(src)
        self._reasoning_md = new
        self._reasoning_collapsed = not self._reasoning_collapsed
        chat.scroll_end(animate=False)

    async def on_click(self, event: events.Click) -> None:
        """点击灰色思考块：展开/折叠思考过程。

        注意：点击命中的是块引用内部的最里层段落，需向上找到
        MarkdownBlockQuote 祖先，再从其中找含「思考过程」标记的文本。
        """
        w = event.widget
        if w is None:
            return
        quote = w
        while quote is not None and not isinstance(quote, MarkdownBlockQuote):
            quote = getattr(quote, "parent", None)
        if quote is None:
            return
        for sub in quote.walk_children(with_self=True):
            content = getattr(sub, "_content", None)
            if content is not None and "思考过程" in str(content):
                await self._toggle_reasoning_async()
                event.stop()
                return

    # ---------------- 工具调用块（代码块样式） ----------------

    @staticmethod
    def _format_tool_args(args_str: str) -> str:
        """把工具参数 JSON 字符串格式化为 键=值 列表。"""
        try:
            obj = json.loads(args_str)
        except Exception:
            return (args_str or "").strip()[:200]
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                s = json.dumps(v, ensure_ascii=False)
                if len(s) > 60:
                    s = s[:57] + "..."
                parts.append(f"{k}={s}")
            return "；".join(parts) if parts else "{}"
        return json.dumps(obj, ensure_ascii=False)[:200]

    def _set_full_view(self, on: bool) -> None:
        """全览模式：隐藏右侧回答区，节点树全宽显示（输入框保留）。"""
        if on == self._full_view:
            return
        self._full_view = on
        chat = self.query_one("#chat-log", Markdown)
        chat.set_class(on, "overview-hidden")
        self.query_one("#sidebar").set_class(on, "overview")
        self.query_one("#fullview-btn", Button).label = "⧉ 分栏" if on else "⛶ 全览"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """侧栏「全览/分栏」按钮：切换全览模式。"""
        if event.button.id == "fullview-btn":
            self._set_full_view(not self._full_view)

    # ---------------- 上下键滚动回答区（双击按住 = 2 倍速） ----------------

    def _scroll_chat(self, delta: int) -> None:
        """↑/↓ 滚动回答区；快速连按/按住（双击）为 2 倍速。"""
        now = time.monotonic()
        if now - self._last_scroll_t < 0.45:
            self._scroll_fast_until = now + 1.0
        self._last_scroll_t = now
        fast = now < self._scroll_fast_until
        chat = self.query_one("#chat-log", Markdown)
        chat.scroll_relative(y=delta * (2 if fast else 1), animate=False)

    def _switch_to(self, node_id: str) -> bool:
        """切换到节点：设当前节点 + 刷新树 + 光标跟随 + 消息区显示节点内容。"""
        if self._full_view:
            self._set_full_view(False)  # 全览模式下切换节点 → 自动退出全览
        self._cancel_flush()  # 丢弃未渲染的流式缓冲（视图将被整体重建）
        if not (self.ctrl and self.ctrl.tree.switch_to_node(node_id)):
            return False
        self._rebuild_tree()
        self._select_tree_node(node_id)
        node = self.ctrl.tree.current_node
        chat = self.query_one("#chat-log", Markdown)
        chat.update(self._node_content(node))
        self._shrink_lists(chat)
        # 思考过程状态：内联显示在节点视图里（提问与回答之间），可点击折叠
        self._reasoning_text = node.reasoning or ""
        self._reasoning_md = (
            "\n\n" + self._build_reasoning_md(node.reasoning) if node.reasoning else ""
        )
        self._reasoning_collapsed = False
        self._answer_started = True  # 节点视图已含 **mincli：** 头部
        self._refresh_usage_bar()
        return True

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """点击会话树节点：切换当前节点并显示其内容。"""
        node_id = event.node.data
        if node_id:
            self._switch_to(node_id)

    # ---------------- 鼠标滚轮（兜底：命中测试可能返回内部容器） ----------------

    def _pointer_over_chat_log(self, event) -> bool:
        try:
            region = self.screen.find_widget(self.query_one("#chat-log", Markdown)).region
        except Exception:
            return False
        return region.contains(int(event.screen_x), int(event.screen_y))

    def _on_mouse_scroll_down(self, event) -> None:
        if self._pointer_over_chat_log(event):
            self.query_one("#chat-log", Markdown).scroll_down(animate=False)
            event.stop()

    def _on_mouse_scroll_up(self, event) -> None:
        if self._pointer_over_chat_log(event):
            self.query_one("#chat-log", Markdown).scroll_up(animate=False)
            event.stop()

    # ---------------- 命令补全弹窗 ----------------

    def on_chat_input_text_changed(self, message: ChatInput.TextChanged) -> None:
        self._update_command_popup(message.text)

    def _update_command_popup(self, text: str) -> None:
        """根据输入内容更新输入框上方的命令补全/提示弹窗。"""
        popup = self.query_one("#cmd-popup")
        body = self.query_one("#cmd-popup-body", Static)
        low = text.strip().lower()

        if not low.startswith("/"):
            popup.remove_class("visible")
            self._completion_matches = []
            self._completion_index = 0
            return

        if low in COMMAND_HELP:
            # 命令已输入完整 → 提示框模式
            popup.add_class("visible")
            body.update(f"[b]{low}[/b]\n\n{COMMAND_HELP[low]}")
            self._completion_matches = []
            self._completion_index = 0
            return

        base = low.split()[0]
        if base in COMMAND_HELP and low != base:
            # 已带参数 → 仍显示该命令的提示
            popup.add_class("visible")
            body.update(f"[b]{base}[/b]\n\n{COMMAND_HELP[base]}")
            self._completion_matches = []
            self._completion_index = 0
            return

        matches = [c for c in COMMAND_HELP if c.startswith(low)]
        self._completion_matches = matches
        self._completion_index = 0
        if not matches:
            popup.remove_class("visible")
            return
        popup.add_class("visible")
        self._render_command_list()

    def _render_command_list(self) -> None:
        body = self.query_one("#cmd-popup-body", Static)
        lines = ["[b]命令补全[/b]（Tab 切换 · Enter 补全）", ""]
        for i, name in enumerate(self._completion_matches):
            brief = COMMAND_HELP[name].splitlines()[0]
            marker = "→" if i == self._completion_index else " "
            lines.append(f"{marker} [b]{name}[/b]  [dim]{brief}[/dim]")
        body.update("\n".join(lines))

    def _advance_or_complete(self) -> bool:
        """Tab 处理：多候选循环高亮，唯一候选直接补全。返回 True 表示已消费。"""
        matches = self._completion_matches
        if not matches:
            return False
        if len(matches) > 1:
            self._completion_index = (self._completion_index + 1) % len(matches)
            self._render_command_list()
            return True
        return self._complete_from_popup()

    def _complete_from_popup(self) -> bool:
        """把输入补全为当前高亮的命令。返回 True 表示已补全（未执行）。"""
        matches = self._completion_matches
        if not matches:
            return False
        if not (0 <= self._completion_index < len(matches)):
            self._completion_index = 0
        target = matches[self._completion_index]
        inp = self.query_one("#chat-input", ChatInput)
        if inp.text != target:
            # 光标移到行尾再插入补全后缀（直接替换 .text 会把光标重置到行首，
            # 导致后续输入插到错误位置）
            inp.cursor_location = (
                inp.document.line_count - 1,
                len(inp.document.lines[-1]),
            )
            suffix = target[len(inp.text):]
            inp.insert(suffix)
        self._update_command_popup(target)
        return True

    # ---------------- 斜杠命令 ----------------

    async def _handle_command(self, text: str) -> bool:
        """处理斜杠命令；返回 True 表示已处理（不再发送给 LLM）。"""
        cmd = text.strip()
        low = cmd.lower()
        ctrl = self.ctrl

        if low in ("/exit", "/quit", "/q", "/e"):
            self.exit()
            return True
        if low in ("/clear", "/c"):
            ctrl.reset()
            chat = self.query_one("#chat-log", Markdown)
            await chat.update(WELCOME)
            self._rebuild_tree()
            self._reasoning_text = ""
            self._reasoning_md = ""
            self._reasoning_collapsed = False
            self._answer_started = False
            self.notify("对话历史已清除")
            self._refresh_usage_bar()
            return True
        if low.startswith("/compact"):
            await self._cmd_compact(cmd)
            return True
        if low in ("/view",):
            self._cmd_view()
            return True
        if low in ("/help", "/h"):
            await self._cmd_help()
            return True
        if low.startswith("/set"):
            await self._cmd_set(cmd)
            return True
        if low.startswith("/model"):
            await self._cmd_model(cmd)
            return True
        if low.startswith("/mcp"):
            await self._cmd_mcp(cmd)
            return True
        if low in ("/full", "/f"):
            self._set_full_view(not self._full_view)
            self.notify("已进入全览模式（切换节点或发送消息自动退出）" if self._full_view else "已退出全览模式")
            return True
        if low in ("/reasoning", "/reason", "/think", "/r"):
            await self._toggle_reasoning_async()
            return True
        if await self._cmd_tree(cmd):
            return True
        if low.startswith("/import"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                self.notify("用法: /import <文件路径或URL>", severity="warning")
            else:
                self.notify("正在导入…")
                err = ctrl.import_target(parts[1].strip())
                if err is None:
                    self.notify("✅ 内容已导入，将在下一次提问时自动附加")
                else:
                    self.notify(err, severity="error")
            return True

        m = re.match(r"^/([A-Za-z]+\d+|main)$", cmd)
        if m and ctrl.tree and m.group(1) in ctrl.tree.nodes:
            self._switch_to(m.group(1))
            return True
        if cmd.startswith("/"):
            self.notify(f"未知命令: {cmd}。输入 /help 查看可用命令", severity="warning")
            return True
        return False

    async def _chat_append(self, markdown: str) -> None:
        """向消息区追加一段 Markdown（带分隔线并滚动到底）。"""
        chat = self.query_one("#chat-log", Markdown)
        await self._safe_append(chat, f"\n\n---\n\n{markdown}")
        self._shrink_lists(chat)
        chat.scroll_end(animate=False)

    async def _safe_append(self, chat: Markdown, md: str) -> None:
        """chat.append 的防御版本：markdown 解析异常时降级为代码块原样显示。"""
        try:
            await chat.append(md)
        except Exception:
            try:
                escaped = md.replace("```", "``")
                await chat.append(f"```\n{escaped}\n```")
            except Exception:
                pass

    def _cmd_view(self) -> None:
        node = self.ctrl.tree.current_node
        if not node:
            self.notify("当前没有节点", severity="warning")
            return
        filepath = self.ctrl.get_node_markdown_file(node.id)
        if filepath is None:
            self.notify("当前节点没有可打开的回答内容", severity="warning")
            return
        try:
            subprocess.Popen(["open", filepath])
            self.notify(f"已用编辑器打开节点 {node.id} 的回答")
        except Exception as e:
            self.notify(f"打开文件失败: {e}", severity="error")

    async def _cmd_help(self) -> None:
        await self._chat_append(
            """**📖 帮助**

**基本命令**
- `/exit`, `/quit`, `/q` — 退出程序（自动保存会话）
- `/clear`, `/c` — 清空当前会话
- `/compact [N]` — 压缩上下文：把早期对话压成详细摘要，保留最近 N 轮原文（默认 5，0=全部压缩；再次执行会重新压缩）
- `/compact off` — 清除压缩摘要，恢复发送完整原始消息
- `/help`, `/h` — 显示此帮助
- `/import <路径或URL>` — 导入文件或抓取网页
- `/mcp <list|add|remove|reload>` — 管理第三方 MCP server
- `/view` — 用编辑器打开当前回答

**配置命令**
- `/set system <提示词>` — 修改系统提示词
- `/set temp <值>` — 设置温度（0.0~2.0）
- `/set model <flash|pro|模型名>` — 切换模型
- `/set thinking <on|off>` — 开关思考模式
- `/set effort <low|high|max>` — 推理强度
- `/set audit <1-4>` — 审核层级
- `/set workspace <路径>` — 命令执行默认工作目录（默认 mincli 启动目录）
- `/set show` — 显示当前配置

**多模型**
- `/model list` — 查看内置与已注册模型
- `/model register <模型名> <URL> [-p provider] [-k key_var]` — 注册新模型（OpenAI 兼容 API）

**树状命令**
- `/<节点ID>`（如 /a3）— 直接跳转到指定节点
- `/tree` — 显示完整对话树
- `/info [节点ID]` — 查看节点详情
- `/up` — 返回父节点
- `/home` — 跳回根节点
- `/full` — 全览模式：隐藏回答区，节点树全宽（再按一次或切换节点自动退出）
- `/reasoning` — 展开/折叠当前消息的思考过程（正文开始后自动折叠，也可点击折叠块）
- `/save [节点ID]` — 导出节点为 Markdown
- `/delete <节点ID>` — 删除节点及其子节点

**快捷键**
- **Enter** 发送 · **Ctrl+J** 换行 · **Alt+Enter** 换行 · **Ctrl+C** 退出
"""
        )

    async def _cmd_set(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=2)
        ctrl = self.ctrl
        usage = "用法: /set system <提示词> | /set temp <值> | /set model <flash|pro|模型名> | /set thinking <on|off> | /set effort <low|high|max> | /set audit <1-4> | /set workspace <路径> | /set show"
        if len(parts) < 2:
            self.notify(usage, severity="warning")
            return
        sub = parts[1]
        if sub == "system" and len(parts) == 3:
            ctrl.set_system(parts[2])
            self.notify("系统提示词已更新")
        elif sub == "temp" and len(parts) == 3:
            try:
                temp = float(parts[2])
                if temp < TEMPERATURE_MIN or temp > TEMPERATURE_MAX:
                    self.notify(f"温度建议在 {TEMPERATURE_MIN}~{TEMPERATURE_MAX} 之间", severity="warning")
                ctrl.set_temperature(temp)
                self.notify(f"温度已设置为 {ctrl.current_temperature}")
            except ValueError:
                self.notify("温度须为数字", severity="error")
        elif sub == "model" and len(parts) == 3:
            if ctrl.set_model(parts[2]):
                self.notify(f"模型已切换为: {ctrl.current_model}")
            else:
                self.notify("未找到该模型。可用 /model list 查看，或 /model register <模型名> <URL> 注册", severity="warning")
        elif sub == "thinking" and len(parts) == 3:
            arg = parts[2].lower()
            if arg in ("on", "1", "true"):
                ctrl.set_thinking(True)
                self.notify(f"思考模式已开启（effort: {ctrl.reasoning_effort}）")
            elif arg in ("off", "0", "false"):
                ctrl.set_thinking(False)
                self.notify("思考模式已关闭")
            else:
                self.notify("用法: /set thinking <on|off>", severity="warning")
        elif sub == "effort" and len(parts) == 3:
            if ctrl.set_effort(parts[2]):
                self.notify(f"推理强度已设置为: {ctrl.reasoning_effort}")
            else:
                self.notify("用法: /set effort <low|high|max>", severity="warning")
        elif sub == "audit" and len(parts) == 3:
            try:
                level = int(parts[2])
                if ctrl.set_audit(level):
                    self.notify(f"审核层级已设置为: {AUDIT_LABELS[level]}")
                else:
                    self.notify("审核层级须为 1-4", severity="warning")
            except ValueError:
                self.notify("审核层级须为数字 1-4", severity="warning")
        elif sub == "workspace" and len(parts) == 3:
            path = os.path.expanduser(parts[2])
            if ctrl.set_workspace(path):
                self.notify(f"命令工作目录已设置为: {ctrl.workspace}")
            else:
                self.notify(f"无法创建/访问目录: {path}", severity="error")
        elif sub == "workspace" and len(parts) == 2:
            self.notify(
                f"当前命令工作目录: {ctrl.workspace or '（未设置，默认 mincli 启动目录）'}"
            )
        elif sub == "show":
            ctrl = self.ctrl
            lines = [
                "**当前配置**",
                "",
                f"- **系统提示词**: {ctrl.current_system}",
                f"- **温度**: {ctrl.current_temperature}",
                f"- **模型**: {ctrl.current_model}",
                f"- **思考模式**: {'开' if ctrl.thinking_enabled else '关'} | 推理强度: {ctrl.reasoning_effort}",
                f"- **审核层级**: {ctrl.audit_level} - {AUDIT_LABELS[ctrl.audit_level]}",
                f"- **命令工作目录**: {ctrl.workspace or '（未设置，默认 mincli 启动目录）'}",
            ]
            if ctrl.tree.current_node:
                lines.append(f"- **当前节点**: {ctrl.tree.current_node.id} ({ctrl.tree.current_node.title})")
            await self._chat_append("\n".join(lines))
        else:
            self.notify(usage, severity="warning")

    async def _cmd_tree(self, cmd: str) -> bool:
        parts = cmd.split()
        low = parts[0].lower()
        tree = self.ctrl.tree
        current_id = tree.current_node.id if tree.current_node else None

        if low == "/tree":
            await self._chat_append(f"```\n{tree.render_tree(current_id)}\n```")
            return True
        if low.startswith("/info"):
            nid = parts[1] if len(parts) > 1 else current_id
            node = tree.nodes.get(nid) if nid else None
            if node:
                await self._chat_append(
                    f"**节点 {node.id}: {node.title}**\n\n"
                    f"- **用户**: {node.user_msg[:PREVIEW_USER_MSG_LEN]}…\n"
                    f"- **助手**: {node.assistant_msg[:PREVIEW_ASSISTANT_MSG_LEN]}…\n"
                    f"- **Tokens**: 输入 {node.input_tokens} / 输出 {node.output_tokens}"
                )
            else:
                self.notify("节点不存在", severity="error")
            return True
        if low == "/up":
            if tree.current_node and tree.current_node.parent_id:
                parent = tree.nodes.get(tree.current_node.parent_id)
                if parent and not self._switch_to(parent.id):
                    self.notify("返回父节点失败", severity="error")
            else:
                self.notify("已在根节点", severity="warning")
            return True
        if low == "/home":
            if tree.root:
                self._switch_to(tree.root.id)
            return True
        if low.startswith("/save"):
            nid = parts[1] if len(parts) > 1 else current_id
            filepath = self.ctrl.save_node(nid) if nid else None
            if filepath:
                self.notify(f"✅ 节点已保存到 {filepath}")
            else:
                self.notify("节点不存在", severity="error")
            return True
        if low.startswith("/delete"):
            nid = parts[1] if len(parts) > 1 else None
            if nid is None:
                self.notify("用法: /delete <节点ID>", severity="warning")
                return True
            if nid not in tree.nodes:
                self.notify(f"未找到节点 {nid}", severity="error")
                return True
            if nid == "main" or (tree.root and nid == tree.root.id):
                self.notify("不能删除根节点", severity="warning")
                return True
            self._ask_confirm(
                "删除节点",
                f"确定要删除节点 {nid} 及其所有子节点吗？",
                lambda ok: self._on_delete_confirmed(nid, ok),
            )
            return True
        return False

    def _on_delete_confirmed(self, nid: str, ok: bool) -> None:
        """确认弹窗回调：ok=True 时执行删除（App 消息泵空闲时才被调用）。"""
        if not ok:
            self.notify("已取消删除")
            return
        tree = self.ctrl.tree
        if tree.delete_node(nid):
            self.ctrl._cleanup_temp_files(keep_ids=set(tree.nodes.keys()))
            self._rebuild_tree()
            if tree.current_node:
                self._select_tree_node(tree.current_node.id)
            self.notify(f"节点 {nid} 及其所有子节点已删除")
        else:
            self.notify(f"删除节点 {nid} 失败", severity="error")

    async def _cmd_model(self, cmd: str) -> None:
        """管理模型注册：/model list | /model register <模型名> <URL> [-p provider] [-k key_var]"""
        parts = cmd.strip().split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub in ("", "list", "ls"):
            await self._model_list()
            return

        if sub in ("register", "add"):
            self._model_register(parts[2] if len(parts) > 2 else "")
            return

        self.notify(
            "用法: /model list | /model register <模型名> <URL> [-p provider] [-k key_var]",
            severity="warning",
        )

    async def _model_list(self) -> None:
        """列出内置 + 已注册模型。"""
        registered = load_models()
        lines = ["**可用模型**", "", "| 模型 | API URL | Key 环境变量 | 来源 |", "|---|---|---|---|"]
        for name, url in MODELS_AVAILABLE.items():
            key = API_PROVIDERS.get("deepseek", "DEEPSEEK_API_KEY")
            lines.append(f"| {name} | {url} | {key} | 内置 |")
        for name, cfg in registered.items():
            lines.append(
                f"| {name} | {cfg.get('url', '—')} | {cfg.get('key_var', 'DEEPSEEK_API_KEY')} | 已注册 |"
            )
        lines.append("\n注册新模型: `/model register <模型名> <URL>`")
        lines.append("切换模型: `/set model <模型名>`")
        await self._chat_append("\n".join(lines))

    def _model_register(self, rest: str) -> None:
        """解析并注册模型：/model register <模型名> <URL> [-p provider] [-k key_var]"""
        tokens = rest.split()
        if len(tokens) < 2:
            self.notify("用法: /model register <模型名> <URL> [-p provider] [-k key_var]", severity="warning")
            return
        model_name = tokens[0]
        url = tokens[1]
        provider = "deepseek"
        key_var = None
        i = 2
        while i < len(tokens):
            if tokens[i] in ("-p", "--provider") and i + 1 < len(tokens):
                provider = tokens[i + 1]
                i += 2
            elif tokens[i] in ("-k", "--key-var") and i + 1 < len(tokens):
                key_var = tokens[i + 1]
                i += 2
            else:
                self.notify(f"无法识别的参数: {tokens[i]}", severity="warning")
                return

        if register_model(provider, model_name, url, key_var):
            self.notify(f"✅ 已注册模型「{model_name}」→ {url}")
            if self.ctrl is not None:
                # 注册后立即可用
                self.ctrl.set_model(model_name)
                self.notify(f"已切换当前模型为: {model_name}")
        else:
            self.notify("注册失败（请检查 ~/.mincli 目录写权限）", severity="error")

    async def _cmd_mcp(self, cmd: str) -> None:
        parts = cmd.strip().split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        if sub in ("", "list", "ls", "status", "show"):
            await self._mcp_list()
        elif sub == "add":
            self._mcp_add(rest)
        elif sub in ("remove", "rm", "del"):
            name = rest.split()[0] if rest.split() else ""
            if not name:
                self.notify("用法: /mcp remove <名称>", severity="warning")
                return
            servers = load_mcp_servers()
            if name not in servers:
                self.notify(f"未找到 server「{name}」", severity="error")
                return
            self._ask_confirm(
                "移除 MCP server",
                f"确定要移除「{name}」吗？",
                lambda ok: self._on_mcp_remove_confirmed(name, ok),
            )
        elif sub == "reload":
            self.notify("正在重新加载 MCP servers…")
            try:
                self.ctrl.mcp_reload()
                self.notify("✅ MCP 已重新加载")
            except Exception as e:
                self.notify(f"MCP 重载失败: {e}", severity="error")
        else:
            self.notify(
                "用法: /mcp list | add <名称> <命令|URL> [参数...] [--header 'K: V'] | remove <名称> | reload",
                severity="warning",
            )

    def _on_mcp_remove_confirmed(self, name: str, ok: bool) -> None:
        """移除 MCP server 的确认回调（App 消息泵空闲时才被调用）。"""
        if not ok:
            return
        servers = load_mcp_servers()
        if name not in servers:
            self.notify(f"未找到 server「{name}」", severity="error")
            return
        del servers[name]
        path = save_mcp_servers(servers)
        self.notify(f"✅ 已移除「{name}」，运行 /mcp reload 生效")

    async def _mcp_list(self) -> None:
        status = self.ctrl.mcp_status()
        servers = load_mcp_servers()
        lines = [f"**MCP Servers**（配置文件: {get_mcp_config_path()}）"]
        if not status:
            lines.append("\nMCP 客户端未就绪")
        else:
            lines += ["", "| 名称 | 命令 | 工具数 | 状态 |", "|---|---|---|---|"]
            for name in sorted(status):
                st = status[name]
                if name == "mincli":
                    cmd = "内置 server"
                else:
                    cfg = servers.get(name, {})
                    cmd = cfg.get("url") or cfg.get("command", "")
                    if cfg.get("headers"):
                        cmd += f"（带 {len(cfg['headers'])} 个请求头）"
                state = "✅ 已连接" if st["connected"] else "⚠ 未连接"
                lines.append(f"| {name} | {cmd} | {st['tools']} | {state} |")
        if not servers:
            lines.append("\n（未配置第三方 server，可用 /mcp add 添加）")
        await self._chat_append("\n".join(lines))

    def _mcp_add(self, rest: str) -> None:
        servers = load_mcp_servers()
        is_url = lambda s: bool(re.match(r"^https?://", s))
        try:
            tokens = shlex.split(rest)  # 支持带引号的参数与 --header 'K: V'
        except ValueError:
            self.notify("参数解析失败（引号不匹配）", severity="warning")
            return
        if len(tokens) < 2:
            self.notify(
                "用法: /mcp add <名称> <命令> [参数...] [--header 'K: V'] 或 /mcp add <名称> <URL> [--header 'K: V']",
                severity="warning",
            )
            return
        name, target = tokens[0], tokens[1]
        headers: dict = {}
        extra: list = []
        i = 2
        while i < len(tokens):
            t = tokens[i]
            if t in ("--header", "-H"):
                if i + 1 >= len(tokens):
                    self.notify(f"缺少 {t} 的值，用法: {t} 'Key: Value'", severity="warning")
                    return
                hv = tokens[i + 1]
                if ":" not in hv:
                    self.notify(f"无效的 header「{hv}」（应为 'Key: Value'）", severity="warning")
                    return
                k, v = hv.split(":", 1)
                headers[k.strip()] = v.strip()
                i += 2
            else:
                extra.append(t)
                i += 1
        if name in servers:
            self.notify(f"已存在同名 server「{name}」，将被覆盖", severity="warning")
        if is_url(target):
            entry: dict = {"url": target}
            if headers:
                entry["headers"] = headers
            servers[name] = entry
        else:
            entry = {"command": target}
            if extra:
                entry["args"] = extra
            if headers:
                self.notify("--header 仅对远程（http/https）server 生效，已忽略", severity="warning")
            servers[name] = entry
        path = save_mcp_servers(servers)
        self.notify(f"✅ 已保存到 {path}，运行 /mcp reload 生效")

    # ---------------- 上下文压缩 ----------------

    async def _cmd_compact(self, cmd: str) -> None:
        """/compact [保留轮数] | /compact off —— 压缩当前分支早期对话。"""
        parts = cmd.strip().split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        ctrl = self.ctrl
        if sub in ("off", "reset", "clear", "undo"):
            if ctrl.clear_compaction():
                self.notify("已清除上下文压缩摘要，将恢复发送完整原始消息")
            else:
                self.notify("当前没有压缩摘要", severity="warning")
            return
        keep = COMPACT_DEFAULT_KEEP
        if sub:
            try:
                keep = max(0, int(sub))
            except ValueError:
                self.notify("用法: /compact [保留轮数] | /compact off", severity="warning")
                return
        if not ctrl.tree or ctrl.tree.current_node is None:
            self.notify("当前没有对话可压缩", severity="warning")
            return
        self.notify("正在压缩上下文…")
        stats = await asyncio.to_thread(ctrl.compact_history, keep, emit=self._emit_from_thread)
        if stats is None:
            self.notify("无可压缩的对话（对话太短，或压缩失败）", severity="warning")
            return
        md = (
            f"**📦 上下文已压缩**\n\n"
            f"- 压缩 {stats['nodes_compressed']} 轮，保留最近 {stats['nodes_kept']} 轮原文\n"
            f"- Token：{stats['before_tokens']} → {stats['after_tokens']}（节省 {stats['saved_tokens']}）\n"
            f"- 摘要长度：{stats['summary_chars']} 字\n\n"
            f"---\n\n{stats['summary']}"
        )
        await self._chat_append(md)

    # ---------------- 消息发送与流式渲染 ----------------

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        if self.ctrl is None:
            self.notify("控制器未就绪（请检查 DEEPSEEK_API_KEY）", severity="error", timeout=5)
            return
        if self._complete_from_popup():
            return  # 命令未输完：先补全，等再次 Enter 执行
        if await self._handle_command(event.text):
            return
        self._cancel_flush()  # 新消息开始前丢弃上一轮残留的流式缓冲
        chat = self.query_one("#chat-log", Markdown)
        chat.append(f"\n\n---\n\n**你**\n\n{event.text}")
        chat.scroll_end(animate=False)
        self._stream_active = False
        self._reasoning_text = ""
        self._reasoning_md = ""
        self._reasoning_collapsed = False
        self._answer_started = False
        self.run_worker(
            lambda: self._run_message(event.text),
            name="chat-message",
            thread=True,
            exit_on_error=False,
        )

    def _run_message(self, text: str) -> None:
        """线程 worker 中运行 controller（阻塞 API 调用），事件转发到主线程。"""
        try:
            self.ctrl.send_message(text, self._emit_from_thread)
        except Exception as e:
            self.call_from_thread(self._append_error, str(e))

    def _emit_from_thread(self, ev: ControllerEvent) -> None:
        self.call_from_thread(self._handle_event, ev)

    async def _handle_event(self, ev: ControllerEvent) -> None:
        """主线程处理控制器事件。

        stream 事件（SSE 按 token 级产生，频率极高）先累积到缓冲，由定时器
        每 80ms 批量渲染一次，避免每个 token 都触发 Markdown 组件的
        mount/布局/重绘 → 渲染卡顿、主线程事件积压。
        低频事件（node_created/tool/status/done/error）先冲刷缓冲再即时处理，
        保证渲染顺序正确。
        """
        if ev.kind == "stream":
            if ev.content:
                self._stream_buf_content += ev.content
            if ev.reasoning:
                self._stream_buf_reasoning += ev.reasoning
            self._ensure_flush_timer()
            return
        if ev.kind == "node_created":
            # 新节点视图整体重建，丢弃任何残留缓冲（防御：正常情况下缓冲为空）
            self._cancel_flush()
            async with self._chat_lock:
                await self._handle_event_inner(ev)
            return
        await self._flush_stream_buffer()
        async with self._chat_lock:
            await self._handle_event_inner(ev)

    def _ensure_flush_timer(self) -> None:
        """确保批量渲染定时器已启动（未启动时才创建）。"""
        if self._flush_timer is None:
            self._flush_timer = self.set_timer(self._flush_interval, self._flush_stream_buffer)

    def _cancel_flush(self) -> None:
        """停止定时器并丢弃缓冲（切换节点/新消息/异常时调用）。"""
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._stream_buf_content = ""
        self._stream_buf_reasoning = ""

    async def _flush_stream_buffer(self) -> None:
        """把缓冲中的流式增量一次性渲染（节流合并的核心）。"""
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        content, reasoning = self._stream_buf_content, self._stream_buf_reasoning
        self._stream_buf_content = ""
        self._stream_buf_reasoning = ""
        if not content and not reasoning:
            return
        async with self._chat_lock:
            await self._render_stream_chunk(content, reasoning)

    async def _render_stream_chunk(self, content: str, reasoning: str) -> None:
        """渲染一批流式增量（正文 + 思考），逻辑与原逐 chunk 渲染等价但按批执行。"""
        chat = self.query_one("#chat-log", Markdown)
        if not self._stream_active:
            self._stream_active = True
            self._answer_started = True
            await chat.append("\n\n---\n\n**mincli**\n\n")
        if reasoning:
            # 思考过程：灰色块引用，位于提问之后、回答之前；
            # 增量跨批次直接拼接（不按 token 断行）
            if not self._reasoning_md:
                md = (
                    "\n\n> ▼ 思考过程（点击折叠）\n>\n> "
                    + self._reasoning_chunk_md(reasoning)
                )
                self._reasoning_md = md
                await self._safe_append(chat, md)
            else:
                md = self._reasoning_chunk_md(reasoning)
                self._reasoning_md += md
                await self._safe_append(chat, md)
            self._reasoning_text += reasoning
            self._reasoning_collapsed = False
        if content:
            if self._reasoning_md and not self._reasoning_collapsed:
                # 正文开始 → 自动折叠思考过程
                await self._collapse_reasoning()
            if not self._answer_started:
                self._answer_started = True
                await self._safe_append(chat, "\n\n**mincli：**\n\n")
            await self._safe_append(chat, content)
        self._shrink_lists(chat)
        chat.scroll_end(animate=False)

    async def _handle_event_inner(self, ev: ControllerEvent) -> None:
        chat = self.query_one("#chat-log", Markdown)
        if ev.kind == "node_created":
            node = ev.node
            if node is not None:
                if self._full_view:
                    self._set_full_view(False)  # 新消息进入新节点 → 退出全览，恢复流式视图
                # 直接进入新节点：树重建 + 光标跟随 + 消息区切换到该节点视图
                # （不输出 **mincli：** 头部——思考过程在提问之后、头部与正文之前）
                self._rebuild_tree()
                self._select_tree_node(node.id)
                await chat.update(
                    f"# {node.id}: {node.title}\n\n"
                    f"**你：**\n\n{node.user_msg}"
                )
                self._shrink_lists(chat)
                chat.scroll_end(animate=False)
                self._stream_active = True  # 视图已含节点头部，后续流式内容直接追加
                self._reasoning_text = ""
                self._reasoning_md = ""
                self._reasoning_collapsed = False
                self._answer_started = False
        elif ev.kind == "tool":
            if ev.tool_summary:
                await chat.append(f"结果：{ev.tool_summary}\n```\n")
            else:
                args = self._format_tool_args(ev.tool_args)
                await chat.append(f"\n```\n{ev.tool_name}\n参数：{args}\n")
            self._shrink_lists(chat)
            chat.scroll_end(animate=False)
        elif ev.kind == "status":
            await chat.append(f"\n> {ev.message}\n")
            self._shrink_lists(chat)
            chat.scroll_end(animate=False)
        elif ev.kind == "error":
            await chat.append(f"\n\n> ⚠️ {ev.message}\n")
            self._shrink_lists(chat)
            chat.scroll_end(animate=False)
            self._rebuild_tree()  # 出错时节点已被回滚，刷新树移除空节点
        elif ev.kind == "done":
            node = ev.node
            if node is not None:
                await chat.append(
                    f"\n\n---\n\n*📊 输入 {node.input_tokens} tokens | 输出 {node.output_tokens} tokens*\n"
                )
            self._shrink_lists(chat)
            chat.scroll_end(animate=False)
            self._rebuild_tree()
            if node is not None:
                self._select_tree_node(node.id)
            self._refresh_usage_bar()

    def _append_error(self, message: str) -> None:
        self._cancel_flush()  # 出错后不再渲染残留流式缓冲
        chat = self.query_one("#chat-log", Markdown)
        chat.append(f"\n\n> ⚠️ {message}\n")
        self._shrink_lists(chat)
        chat.scroll_end(animate=False)

    # ---------------- 确认对话框（供 controller 工具调用） ----------------

    def _ask_confirm(self, title: str, text: str, on_result) -> None:
        """非阻塞确认：推送确认弹窗，用户点击后以 on_result(bool) 回调。

        必须在消息处理器里避免阻塞等待用户输入：处理器不返回，App 消息泵就
        卡在 _dispatch_message 里，键盘/鼠标事件（包括点到弹窗按钮）全部无法
        分发——弹窗能显示但整个应用卡死。因此这里直接 push_screen + callback，
        处理器立即返回，用户点击后回调在泵空闲时执行。
        """
        self.push_screen(ConfirmScreen(title, text), callback=on_result)

    def _confirm(self, title: str, text: str) -> bool:
        """worker 线程调用：在主线程弹确认框并阻塞等待结果。

        与 _ask_confirm 不同：本方法经 call_from_thread 以独立 asyncio 任务
        运行（不占用 App 消息泵），所以可以阻塞等待 Future。
        """
        return self.call_from_thread(self._confirm_async, title, text)

    async def _confirm_async(self, title: str, text: str) -> bool:
        """主线程弹确认框并阻塞等待结果（仅限独立任务上下文，勿在消息处理器内 await）。

        ConfirmScreen 在它自己的消息泵上直接 set_result 解决 Future，不依赖
        push_screen 的 callback（那会经 call_next 投递到 App 泵）。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        screen = ConfirmScreen(title, text, result_future=future)
        await self.push_screen(screen)
        return await future


def main() -> None:
    ChatApp().run()


if __name__ == "__main__":
    main()
