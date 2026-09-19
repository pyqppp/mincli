"""mincli ChatApp 主应用（阶段 3：斜杠命令迁移到 TUI）。

运行：`venv/bin/python -m mincli.tui.app`（需要真实终端 + DEEPSEEK_API_KEY）
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import time

# 必须在 Textual Markdown 组件创建解析器之前执行：
# 防御 markdown-it-py 解析极端输入（引用块内表格被流式截断等）时的越界崩溃
from mincli.markdown_safe import _patch_markdown_it

_patch_markdown_it()

from rich.markup import escape as _markup_escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Markdown, Static, Tree
from textual.widgets._header import HeaderIcon, HeaderTitle

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

# 防御 Textual Screen 选区初始化崩溃：流式渲染重建 Markdown 块时，鼠标
# 按下可能命中刚被移除的块（parent 已变 None），Screen._forward_event 的
# MouseDown 分支构造 SelectStart 时 container = widget.parent = None，
# 访问 container.region.offset 抛 AttributeError——与上面 Selection.extract
# 越界是同一竞态的另一半。命中已分离 widget 时临时关闭选区重试一次：
# 选区逻辑跳过、鼠标事件正常转发，随后恢复 ALLOW_SELECT。
_ORIG_SCREEN_FORWARD_EVENT = None


def _patch_textual_screen_forward_event() -> None:
    """把 Screen._forward_event 包一层「崩溃重试」（幂等）。"""
    global _ORIG_SCREEN_FORWARD_EVENT
    from textual.screen import Screen

    if getattr(Screen, "_mincli_safe_forward_event", False):
        return
    _ORIG_SCREEN_FORWARD_EVENT = Screen._forward_event
    Screen._forward_event = _safe_screen_forward_event
    Screen._mincli_safe_forward_event = True


def _safe_screen_forward_event(self, event) -> None:
    """MouseDown 命中已分离 widget 导致选区初始化崩溃 → 临时关选区重试。"""
    if isinstance(event, events.MouseDown):
        try:
            return _ORIG_SCREEN_FORWARD_EVENT(self, event)
        except AttributeError:
            old = self.app.ALLOW_SELECT
            self.app.ALLOW_SELECT = False
            try:
                return _ORIG_SCREEN_FORWARD_EVENT(self, event)
            finally:
                self.app.ALLOW_SELECT = old
    return _ORIG_SCREEN_FORWARD_EVENT(self, event)


_patch_textual_screen_forward_event()

from mincli.config import (
    DEFAULT_SYSTEM_PROMPT,
    MODEL_FLASH,
    BALANCE_REFRESH_SECONDS,
    FILES_LIST_PAGE,
    FILES_LIST_PAGE_MAX,
    FILES_MAX_BYTES,
    FILES_MAX_COUNT,
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
from mincli.helpers import open_path_with_os, split_path_args
from mincli.tools.files import FilesAPIError
from mincli.tools.images import image_placeholder_text
from mincli.tui.commands import (
    Completion,
    command_block,
    complete as complete_command,
    help_markdown,
    runtime_providers,
    usage_line,
)
from mincli.tui.confirm import ConfirmScreen
from mincli.tui.theme import TREE_THEMES, color_label, theme_name
from mincli.tui.tree_wizard import TreeWizardScreen
from mincli.tui.widgets import ChatInput, ToolCard, TreeRow

WELCOME = """# mincli

DeepSeek 树状对话 TUI

- **左侧**：对话树（点击节点切换，点小三角收起/展开）
- **中间**：消息流（Markdown 流式渲染）
- **底部**：多行输入框（**Enter** 发送，**Ctrl+J** 换行）
- **Esc** / 生成中 **Ctrl+C**：打断当前生成或正在执行的命令
- **Ctrl+C**：退出（自动保存会话）

直接输入问题开始对话，输入 `/help` 查看命令。
"""

# 命令补全/提示、/help、各处的用法提示都来自 mincli/tui/commands.py 的同一份
# 命令规格（三级：/set → /set thinking → /set thinking on），这里不再维护文案表。
COMPLETION_HINT = "Tab 切换 · Shift+Tab 反向 · Enter 补全"

# /help 尾部：命令清单由命令规格生成，这里只补规格里没有的说明
HELP_TAIL = """
**补充说明**
- 输入 `/` 打开命令补全：继续输入过滤候选，**Tab** 循环（**Shift+Tab** 反向），**Enter** 补全；命令写完按 **Enter** 执行
- 二级命令（如 `/set thinking on|off`）与运行时候选（对话树编号、工作流名、MCP server 名）都能补全
- `/<节点ID>`（如 `/a3`）— 直接跳转到指定节点
- 图片理解仅 deepseek-flash 支持；其他模型会提示切换
- 配置类命令（`/set ...`）对当前对话树生效，工作目录与审核层级也是每棵树独立

**快捷键**
- **Enter** 发送 · **Ctrl+J** 换行 · **Alt+Enter** 换行
- **Tab** 命令补全/循环候选 · **Shift+Tab** 反向循环候选
- **Esc** / 生成中 **Ctrl+C** 打断当前生成或正在执行的命令（再按一次 **Ctrl+C** 强制退出） · **Ctrl+C** 退出
- **Ctrl/Alt+1~8** 切换对话树
"""

# 思考过程：灰色块引用（不再折叠/点击展开；多轮工具调用时每个思考段
# 各自独立成块，正文穿插其间正常显示）
REASONING_HEADER_MD = "> 思考过程"

# ---------------- 流式渲染性能（长输出卡顿的根因治理） ----------------
# Textual 的 Markdown.append 每次都会把「仍在增长的尾部块」整体重新解析并重建
# 组件（见 Markdown.append → _update_from_block：remove + 重新 mount）。流式
# 输出越长，这个尾部块越大，单次刷新越慢，累计就是 O(n²) —— 思考过程动辄上万
# 字，于是越写越卡，最终主线程被刷新压满、整个界面卡死。
#
# 治理办法：不让单个 Markdown 段无限增长。超过 STREAM_SEG_MAX_CHARS 后，在
# 「安全边界」（行尾、代码围栏配对、非列表/表格行）封存当前段，后续内容进入
# 新段。这样每次刷新只重建一个有界的尾块，整体回到 O(n)；封段处最多多出一个
# 段间距，Markdown 结构不会被截断。
STREAM_SEG_MAX_CHARS = 2000      # 达到该长度且边界安全 → 封段
STREAM_SEG_FORCE_CHARS = 4000    # 迟迟找不到安全边界时的兜底封段阈值（仍避开未闭合围栏）
FLUSH_INTERVAL_BASE = 0.08       # 刷新间隔基准（秒）
FLUSH_INTERVAL_MIN = 0.05        # 自适应刷新间隔下限（秒）
FLUSH_INTERVAL_MAX = 0.35        # 自适应刷新间隔上限（秒）

# 列表项行首（有序/无序），封段处若正好落在列表项上会把列表拆成两段、序号重排
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s")


def _strip_quote_prefix(line: str) -> str:
    """去掉思考过程块引用的 `> ` 前缀，便于按原始 Markdown 语义判断结构。"""
    s = line.lstrip()
    if s.startswith(">"):
        s = s[1:].lstrip()
    return s


def stream_segment_fences_balanced(raw: str) -> bool:
    """当前流式段的 ``` 围栏是否成对（不成对时封段会把代码块截成两半）。"""
    fences = 0
    for ln in raw.split("\n"):
        if _strip_quote_prefix(ln).startswith("```"):
            fences += 1
    return fences % 2 == 0


def stream_segment_sealable(raw: str) -> bool:
    """判断能否在“批次之间”安全封段（不破坏 Markdown 结构）。

    安全条件：以换行结尾（不切在行中间）、围栏成对、最后一个非空行不是列表项
    或表格行（否则会把列表/表格拆成两段）。
    """
    if not raw.endswith("\n"):
        return False
    if not stream_segment_fences_balanced(raw):
        return False
    last = ""
    for ln in reversed(raw.split("\n")):
        s = _strip_quote_prefix(ln).strip()
        if s and not s.startswith("```"):
            last = s
            break
    if last.startswith("|"):
        return False
    return not _LIST_ITEM_RE.match(last)


def escape_leading_quote_markers(line: str) -> str:
    """转义行首的 Markdown 引用标记（`>`），避免与我们自己加的 `> ` 前缀叠加。

    我们把思考过程整体渲染成灰色块引用（每行加 `> `）。模型思考里若自带引用
    行（`> 注意…`）或流式 chunk 边界正好落在 `>` 前，前缀就会叠加成 `>>` /
    `> >`，被 Markdown 解析成「引用块里再套引用块」——界面上多出一道竖线，
    看上去就是 `>>`。这里把行首的 `>` 逐个转义成 `\\>`（渲染仍是字面 `>`），
    使其留在同一层引用块内，不再嵌套。

    只处理行首：正文中间的 `>` 本来就是普通字符。缩进 4 空格以上的行是代码块
    （内部 `>` 不会开启引用）故不处理；围栏代码块（```）内的行首 `>` 会被多
    转义出一个反斜杠，属极少数情况，且只影响思考过程的显示。（增量流式无法
    可靠地跨 chunk 跟踪围栏状态，故不做该分支。）
    """
    n = len(line)
    i = 0
    while i < n and line[i] == " " and i < 3:  # CommonMark 允许最多 3 个前导空格
        i += 1
    chars = list(line)
    j = i
    while j < n and line[j] == ">":
        chars[j] = "\\>"
        j += 1
        while j < n and line[j] == " ":  # `> > x` 这类多层标记一并转义
            j += 1
    return "".join(chars) if j != i else line


def quote_block_text(text: str) -> str:
    """把整段文本转成块引用（每行 `> `，行首自带引用标记先转义）。"""
    out = []
    for ln in text.splitlines():
        safe = escape_leading_quote_markers(ln)
        out.append(f"> {safe}" if safe else ">")
    return "\n".join(out)


# 8 套对话树主题（编号 = 颜色，见 mincli/tui/theme.py）：
# 树 1 就是原有的青色主题，2-8 号依次紫、绿、橙、粉、蓝、黄、红，
# 第 9 棵起循环复用。背景/面板色 8 套一致，只轮换主色与强调色。


class ChatHeader(Header):
    """顶栏：左侧标题，右侧显示当前对话树编号与后台完成状态。

    Header 默认会用 HeaderClockSpace 占住右侧 10 列（本项目未启用时钟），
    这里把它换成自己的徽标——宽度自动让给标题区，不会重叠。
    """

    def compose(self) -> ComposeResult:
        yield HeaderIcon().data_bind(Header.icon)
        yield HeaderTitle()
        yield Static("", id="tree-badge", markup=False)


# Ctrl/Alt + 1..8 直接切换对话树。Alt 是兜底：不少终端（含 macOS
# Terminal.app 默认设置）不会把 Ctrl+数字上报成独立按键，能上报时两者都可用。
_TREE_SWITCH_BINDINGS = [
    Binding(f"ctrl+{i}", f"switch_tree({i})", f"切到对话树 {i}", show=False)
    for i in range(1, 9)
] + [
    Binding(f"alt+{i}", f"switch_tree({i})", f"切到对话树 {i}", show=False)
    for i in range(1, 9)
]


class ChatApp(App):
    """mincli 主界面：会话树 + 消息流 + 多行输入。"""

    TITLE = "mincli"
    SUB_TITLE = "DeepSeek Chat"

    CSS_PATH = "chat.tcss"

    BINDINGS = [
        # Ctrl+C 在所有平台统一：有选中文字时先复制（Textual 屏幕级绑定
        # 优先，ChatInput/输入框选区、聊天区选区均可复制），无选中时退出。
        # 生成/命令执行进行中则先打断当前轮（见 action_quit/action_interrupt）。
        Binding("ctrl+c", "quit", "退出/打断"),
        # Esc 打断当前生成（ChatInput 内因 TextArea 会吞 Esc，另加优先绑定）
        Binding("escape", "interrupt", "打断", show=False),
        Binding(
            "caps_lock,num_lock,scroll_lock",
            "ignore_lock",
            "忽略锁定键",
            show=False,
            priority=True,
        ),
    ] + _TREE_SWITCH_BINDINGS

    def __init__(self, controller: ChatController | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        for theme in TREE_THEMES:
            self.register_theme(theme)
        self._injected_controller = controller
        self.ctrl: ChatController | None = None
        self._stream_active = False
        # 是否有生成/命令执行进行中（Ctrl+C / Esc 据此决定「打断」还是「退出」）
        self._turn_active = False
        self._default_placeholder = ""
        self._reasoning_open = False  # 流式中当前思考块是否仍打开（正文出现则关闭）
        self._answer_started = False  # 正文的 "**mincli：**" 头部是否已输出
        self._full_view = False  # 全览模式：节点树全宽（隐藏回答区）
        self._last_scroll_t = 0.0  # 上下键滚动：上次按键时间（用于双击加速）
        self._scroll_fast_until = 0.0  # 双击按住 → 2 倍速滚动截止时间
        self._chat_lock = asyncio.Lock()  # 串行化聊天区 update/append（流式渲染 vs 节点切换）
        # 侧栏对话树列表（#tree-list 里的 TreeRow，最多显示 3 行，超出滚动）
        self._tree_rows: list[TreeRow] = []
        # 聊天区（#chat-log 容器）：多段正文 Markdown + 工具卡片，按时间穿插。
        self._chat_md: Markdown | None = None  # 当前正在流式的正文 Markdown 段
        self._chat_blocks: list = []  # 有序：Markdown 段 或 ToolCard
        self._active_tool_card = None  # 正在执行的工具卡片（开始→结束更新同一张）
        self._completion: Completion | None = None  # 当前补全状态（命令树 + 候选）
        self._completion_index = 0
        # 流式渲染节流：SSE 按 token 级产生事件，逐事件 append 会导致
        # Markdown 组件反复 mount/布局/重绘，主线程跟不上 → 渲染卡顿。
        # 这里把增量累积到缓冲批量渲染；间隔按单次刷新的真实 CPU 耗时自适应
        # （见 _adapt_flush_interval），再叠加按段封段限制单块增长。
        self._stream_buf_content = ""  # 待渲染的正文增量缓冲
        self._stream_buf_reasoning = ""  # 待渲染的思考增量缓冲
        self._flush_interval = FLUSH_INTERVAL_BASE  # 批量渲染间隔（秒，按刷新耗时自适应）
        self._flush_cpu_ema = 0.0  # 单次刷新 CPU 耗时的指数滑动平均（用于自适应间隔）
        self._flush_timer = None  # 批量渲染定时器（textual Timer）
        # 当前流式 Markdown 段的规模（用于封段，见 STREAM_SEG_MAX_CHARS 注释）
        self._stream_seg_chars = 0  # 已 append 到当前段的字符数
        self._stream_seg_raw = ""  # 当前段的原始文本（未加 `> ` 前缀），用于结构判断
        self._balance_txt: Optional[str] = None  # 最近一次拉取的账户余额（字符串）
        # 工作流（/wf）：已挂载到“下一次输入”的工作流名（发送后自动解除）
        self._pending_wf: Optional[str] = None
        # /files：最近一次列表结果（供 /files info|delete 用序号引用，免手抄 file_id）
        self._files_listing: list = []
        # /wf edit：系统编辑器打开的临时文档轮询（检测改动后自动回写）
        self._wf_edit_timer = None  # textual Interval
        self._wf_edit_path: Optional[str] = None
        self._wf_edit_name: Optional[str] = None
        self._wf_edit_mtime: Optional[float] = None
        self._wf_edit_deadline: Optional[float] = None

    def action_ignore_lock(self) -> None:
        """忽略锁定键（Caps Lock / Num Lock / Scroll Lock）。"""
        pass

    async def action_quit(self) -> None:
        """Ctrl+C：有进行中的生成/命令时先打断；已请求打断仍未结束时强制退出。"""
        if self._turn_active and self.ctrl is not None:
            if not self.ctrl.interrupt_pending:
                self.action_interrupt()
                return
            # 已请求过打断（例如流式卡住不返回）：再按一次直接退出
        self.exit()

    def action_interrupt(self) -> None:
        """打断当前轮的流式生成 / 正在执行的命令（Esc 或忙碌时 Ctrl+C）。

        空闲时（没有进行中的轮次）不做任何事，交给其他 Esc 处理逻辑。
        """
        if self.ctrl is None or not self._turn_active:
            return
        self.ctrl.interrupt()
        self.notify("⏹ 正在打断当前生成/命令…", timeout=3)

    def _set_turn_active(self, active: bool) -> None:
        """切换「生成中」状态：占用期间输入框占位提示如何打断。"""
        self._turn_active = active
        self._refresh_input_placeholder()
        self._refresh_tree_badge()

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

    def notify(self, message, **kwargs) -> None:
        """统一关闭 markup 解析。

        通知里常有路径/模型名/工作流名/工具参数这类用户或模型可控的文本，
        形如 `x="y z"` 的片段会被 Textual 当样式标签解析，
        直接抛 MarkupError: Expected markup value（显示层报错）。
        通知内容不需要富文本，全部按纯文本渲染。
        """
        kwargs.setdefault("markup", False)
        super().notify(message, **kwargs)

    def compose(self) -> ComposeResult:
        yield ChatHeader()
        with Horizontal():
            with Vertical(id="sidebar"):
                with Horizontal(id="sidebar-header"):
                    yield Static("会话", id="sidebar-title")
                    yield Button("全览", id="fullview-btn", compact=True)
                # 对话树列表：最多 3 行，超出这栏内部滚动（见 chat.tcss）
                yield VerticalScroll(id="tree-list")
                yield Tree("全部", id="tree")
            yield VerticalScroll(id="chat-log")
        with Vertical(id="cmd-popup"):
            yield Static("", id="cmd-popup-body")
        yield ChatInput(
            id="chat-input", placeholder="输入消息，Enter 发送，Ctrl+J 换行"
        )
        # 悬停弹窗放独立布局层（见 chat.tcss #import-popup），固定显示在状态条上方
        # markup=False：这里显示文件名/工作流名，方括号或 `x="y"` 会被当标签解析而报错
        yield Static("", id="import-popup", markup=False)
        # 状态条合并三分栏：左=缓存/余额，中=已导入文件提示（悬停显示完整列表），右=下次输入
        with Horizontal(id="usage-bar"):
            yield Static("", id="usage-left", markup=False)
            yield Static("", id="usage-center", markup=False)
            yield Static("", id="usage-right", markup=False)
        yield Footer()

    async def on_mount(self) -> None:
        self.theme = theme_name(1)  # 默认树 1 的青色；_refresh_tree_ui 会按当前树覆盖
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
                default_model=MODEL_FLASH,
                # MCP 由 _start_mcp 在首屏之后后台启动（日志出口也得先接好）
                auto_start_mcp=False,
            )
        self.ctrl.confirm = self._confirm
        self._start_mcp()
        self.query_one("#tree", Tree).auto_expand = False  # 点击节点名只切换节点，不收起/展开
        # 状态条中段导入提示 + 其上方的悬停弹窗（缓存引用，on_mouse_move 高频使用）
        self._import_center_w = self.query_one("#usage-center", Static)
        self._import_popup_w = self.query_one("#import-popup", Static)
        self._rebuild_tree()
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_tree_ui()
        self._start_balance_refresh()
        self._refresh_usage_bar()
        self._refresh_import_status()
        # 启动即进入上次会话的当前节点：渲染放在首帧之后（恢复长对话的
        # Markdown 要 0.5s 左右，没必要挡在界面出现之前），界面先出来，
        # 内容紧接着补上——见 _restore_current_view。
        self.call_after_refresh(self._restore_current_view)
        # 一棵树都没有：界面先出来，再弹建树向导（取消则退出程序）
        if not self.ctrl.has_trees:
            self.call_after_refresh(lambda: self._open_tree_wizard(initial=True))

    async def _restore_current_view(self) -> None:
        """首帧渲染完成后显示当前节点内容（无节点时显示欢迎页）。"""
        if self.ctrl is None or self.ctrl.tree is None:
            return
        # 首帧到这里的间隙里用户已经发了消息：别用旧节点内容盖掉当前视图
        if self._turn_active or self._stream_active:
            return
        if not self.ctrl.has_trees:
            await self._chat_reset(WELCOME)
            return
        node = self.ctrl.tree.current_node
        if node is not None:
            await self._switch_to(node.id)
        else:
            await self._chat_reset(WELCOME)
        self._load_draft()

    def on_unmount(self) -> None:
        self._cancel_flush()
        self._wf_stop_edit()
        if self.ctrl is not None:
            self._save_draft()
            self.ctrl.save_session()
            self.ctrl.close()

    # ---------------- 对话树：侧栏列表 / 顶部徽标 / 配色 / 草稿 ----------------

    def _apply_tree_theme(self) -> None:
        """按当前树的颜色切换整机主题（背景与面板色不变，只换主色系）。"""
        if self.ctrl is None:
            return
        try:
            self.theme = theme_name(self.ctrl.tree_color())
        except Exception:
            self.theme = theme_name(1)

    def _refresh_tree_ui(self) -> None:
        """刷新与「当前是哪棵树」相关的界面：主题、侧栏列表、顶部徽标、输入提示。"""
        if self.ctrl is None:
            return
        self._apply_tree_theme()
        self._refresh_tree_badge()
        self._refresh_input_placeholder()
        try:
            self.run_worker(
                self._rebuild_tree_list(),
                name="tree-list",
                group="tree-list",   # 独立分组：exclusive 只取消上一次列表重建，
                exclusive=True,      # 不会误伤生成/余额等默认分组的 worker
                exit_on_error=False,
            )
        except Exception:
            pass

    async def _rebuild_tree_list(self) -> None:
        """重建侧栏对话树列表（行数不定，只能异步挂载/卸载）。"""
        if self.ctrl is None:
            return
        try:
            container = self.query_one("#tree-list", VerticalScroll)
        except Exception:
            return
        await container.remove_children()
        numbers = self.ctrl.tree_numbers()
        active = self.ctrl.active_number
        if not numbers:
            # 一棵树都没有：这一栏不占行（只剩上面那行「会话/全览」）。
            # 此时界面由建树向导 + 徽标/输入框提示引导，不需要占位文字。
            self._tree_rows = []
            try:
                container.styles.height = 0
            except Exception:
                pass
            return
        rows = []
        for number in numbers:
            rows.append(
                TreeRow(
                    number,
                    self.ctrl.tree_color(number),
                    f"{number} 对话",
                    active=(number == active),
                    mark=self.ctrl.tree_marks.get(number, ""),
                )
            )
        self._tree_rows = rows
        await container.mount_all(rows)
        # 高度：树少时收缩（1 棵 = 1 行），超过 3 棵固定 3 行改为内部滚动。
        # 注意不能用「height: auto + max-height」代替：那样超出的行会被挤成
        # 0 高度而不是产生滚动区（Textual 会把子控件压进可视高度里）。
        try:
            container.styles.height = 3 if len(rows) > 3 else "auto"
        except Exception:
            pass
        # 当前树滚进视野（列表最多显示 3 行）
        if active in numbers:
            index = numbers.index(active)
            try:
                container.scroll_to(y=max(0, index - 2), animate=False)
            except Exception:
                pass

    def _refresh_tree_badge(self) -> None:
        """顶部标题右侧的当前对话树徽标（编号 + 生成/完成/出错状态）。"""
        if self.ctrl is None:
            return
        try:
            badge = self.query_one("#tree-badge", Static)
        except Exception:
            return
        if not self.ctrl.has_trees:
            badge.update("未创建对话树")
            return
        number = self.ctrl.active_number
        parts = [f"树 {number}"]
        if self.ctrl.generating_number == number:
            parts.append("生成中")
        for mark_number, mark in sorted(self.ctrl.tree_marks.items())[:2]:
            text = "完成" if mark == "done" else "出错"
            if mark_number != number:
                text = f"树 {mark_number} {text}"
            parts.append(text)
        badge.update(" · ".join(parts))

    def _refresh_input_placeholder(self) -> None:
        inp = self.query_one("#chat-input", ChatInput)
        if not self._default_placeholder:
            self._default_placeholder = str(inp.placeholder or "")
        if self.ctrl is not None and not self.ctrl.has_trees:
            inp.placeholder = "请先新建对话树（/tree new）"
        elif self._turn_active:
            inp.placeholder = "生成中… 按 Esc 打断（Ctrl+C 亦可）"
        else:
            inp.placeholder = self._default_placeholder

    def _save_draft(self) -> None:
        """把输入框当前内容记到「当前树」的草稿（切树/退出前调用）。"""
        if self.ctrl is None:
            return
        try:
            self.ctrl.draft = self.query_one("#chat-input", ChatInput).text
        except Exception:
            pass

    def _load_draft(self) -> None:
        """切换到某棵树后恢复它自己的草稿。"""
        if self.ctrl is None:
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        text = self.ctrl.draft or ""
        if inp.text != text:
            inp.load_text(text)
            try:
                inp.move_cursor((inp.document.line_count - 1, len(inp.document.lines[-1])))
            except Exception:
                pass
        self._update_command_popup(inp.text)

    def _reset_view_state(self) -> None:
        """切树前清掉与上一棵树绑定的视图状态（流式缓冲/工具卡片等）。"""
        self._cancel_flush()
        self._stream_active = False
        self._reasoning_open = False
        self._answer_started = False
        self._active_tool_card = None

    async def _render_active_tree(self) -> None:
        """把界面切到当前树的当前节点（无节点则显示欢迎页）。"""
        if self.ctrl is None:
            return
        self._reset_view_state()
        tree = self.ctrl.tree
        node = tree.current_node if tree is not None else None
        if node is not None:
            await self._chat_reset(self._node_content(node))
            await self._chat_shrink_lists(scroll=False)
            self._answer_started = True
        else:
            await self._chat_reset(WELCOME)
        self._rebuild_tree()
        if node is not None:
            self._select_tree_node(node.id)
        self._load_draft()
        self._refresh_usage_bar()
        self._refresh_import_status()

    async def _switch_tree(self, number: int, notify: bool = True) -> bool:
        """切换到指定对话树（正在生成的那棵树会继续在后台跑）。"""
        if self.ctrl is None:
            return False
        if number == self.ctrl.active_number:
            return True
        if not self.ctrl.switch_tree(number):
            self.notify(f"对话树 {number} 不存在", severity="warning")
            return False
        self._refresh_tree_ui()
        await self._render_active_tree()
        if notify:
            self.notify(
                f"已进入对话树 {number}（{color_label(self.ctrl.tree_color())}）"
            )
        return True

    def action_switch_tree(self, number="1") -> None:
        """Ctrl/Alt + 1..8：直接切换对话树。"""
        try:
            target = int(number)
        except (TypeError, ValueError):
            return
        asyncio.ensure_future(self._switch_tree(target))

    async def on_tree_row_selected(self, event: TreeRow.Selected) -> None:
        """点击侧栏某一行对话树 → 切换过去。"""
        await self._switch_tree(int(event.number))

    # ---------------- 建树 / 改能力向导 ----------------

    def _open_tree_wizard(
        self, initial: bool = False, edit_number: int | None = None
    ) -> None:
        """弹出建树向导（initial=True 表示启动时没有树，取消则退出程序）。"""
        if self.ctrl is None:
            return
        if edit_number is None:
            heading = "新建对话树"
            note = (
                "对话能力必选；系统工具与外置 MCP 工具按需勾选。"
                "建好后可用 /tree N tools 修改。"
            )
            caps = {"system_tools": True, "mcp_tools": []}
        else:
            heading = f"对话树 {edit_number} 的能力"
            note = "改动对下一次请求生效。"
            caps = self.ctrl.tree_caps(edit_number)
        screen = TreeWizardScreen(
            heading=heading,
            note=note,
            groups=self.ctrl.available_tool_groups(),
            servers=self.ctrl.external_servers(),
            initial=caps,
            mcp_ready=self.ctrl.mcp_ready,
        )

        def on_done(result) -> None:
            if result is None:
                if initial:
                    self.notify("未创建对话树，退出程序")
                    self.exit()
                return
            if edit_number is None:
                number = self.ctrl.create_tree(
                    result["system_tools"], result["mcp_tools"]
                )
                asyncio.ensure_future(self._after_tree_created(number))
            else:
                self.ctrl.set_tree_caps(
                    edit_number, result["system_tools"], result["mcp_tools"]
                )
                self._refresh_tree_ui()
                self.notify(f"对话树 {edit_number} 的能力已更新")

        self.push_screen(screen, callback=on_done)

    async def _after_tree_created(self, number: int) -> None:
        self._refresh_tree_ui()
        await self._render_active_tree()
        self.notify(
            f"已创建对话树 {number}（{color_label(self.ctrl.tree_color())}）"
        )

    def _refresh_wizard_tools(self) -> None:
        """MCP 连完后把工具列表补进正开着的建树向导。"""
        if self.ctrl is None:
            return
        try:
            screen = self.screen
        except Exception:
            return
        if isinstance(screen, TreeWizardScreen):
            screen.refresh_tools(
                self.ctrl.available_tool_groups(),
                self.ctrl.external_servers(),
                self.ctrl.mcp_ready,
            )

    # ---------------- MCP 后台连接 ----------------

    def _start_mcp(self) -> None:
        """接上 MCP 日志出口并确保后台连接已启动。

        MCP 连接不再挡在首屏前面（原本要等 4 秒左右），改为后台进行：界面先
        出来，连接完成后再补上 MCP 工具；期间发送消息会短暂等待（见
        ChatController.wait_mcp_ready）。
        """
        if self.ctrl is None:
            return
        self.ctrl.mcp_logger = self._on_mcp_log
        if not self.ctrl.mcp_started:
            self.ctrl.start_mcp()
        if self.ctrl.mcp_started and self.ctrl.mcp_connecting:
            self._watch_mcp_ready()

    def _watch_mcp_ready(self) -> None:
        """后台等待 MCP 连接结束，完成后刷新状态条（不阻塞 UI）。"""
        self.run_worker(
            self._wait_mcp_worker, name="mcp-ready", thread=True, exit_on_error=False
        )

    def _wait_mcp_worker(self) -> None:
        if self.ctrl is None:
            return
        try:
            self.ctrl.wait_mcp_ready()
        finally:
            try:
                self.call_from_thread(self._on_mcp_ready_ui)
            except Exception:
                pass

    def _on_mcp_ready_ui(self) -> None:
        """MCP 连接结束：刷新状态条，并把刚取到的工具列表补进建树向导。"""
        self._refresh_usage_bar()
        self._refresh_wizard_tools()

    def _on_mcp_log(self, message: str) -> None:
        """MCP 客户端日志（后台线程）→ 主线程通知。

        这些消息以前是 print 在 TUI 启动之前输出的；现在连接在后台进行，
        直接写 stdout 会把界面冲乱，所以统一转成通知。
        """
        try:
            self.call_from_thread(self._show_mcp_log, message)
        except Exception:
            pass

    def _show_mcp_log(self, message: str) -> None:
        self.notify(message, timeout=5)
        self._refresh_usage_bar()

    # ---------------- 输入栏状态条（缓存命中率 / 余额 / 下次输入） ----------------

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
        """刷新输入栏下方状态条（左：缓存命中率+余额；右：下次输入 token 与预计价格）。"""
        if self.ctrl is None:
            return
        stats = self.ctrl.usage_stats()
        rate = stats["cache_hit_rate"]
        rate_txt = f"{rate * 100:.0f}%" if rate is not None else "--"
        if self._balance_txt is None:
            bal_txt = "…"
        else:
            bal_txt = f"¥{self._balance_txt}"
        left = f"缓存命中 {rate_txt}   余额 {bal_txt}"
        if self.ctrl.mcp_connecting:
            left += "   MCP 连接中…"
        self.query_one("#usage-left", Static).update(left)
        tokens = stats["next_input_tokens"]
        price = stats["estimated_price"]
        price_txt = f"≈ ¥{price:.4f}" if price is not None else "--"
        peak_txt = "高峰" if stats["peak"] else "空闲"
        self.query_one("#usage-right", Static).update(
            f"⏭ 下次输入 {tokens:,} tok {price_txt}（{peak_txt}价）"
        )

    def _refresh_import_status(self) -> None:
        """刷新状态条中段的提示：工作流挂载提示优先，其次「已导入文件」提示。

        无任何提示时清空隐藏（弹窗随之隐藏）。
        """
        if self.ctrl is None:
            return
        hint = self._import_center_w
        if self._pending_wf:
            text = f"工作流已挂载：{self._pending_wf}（发送即执行 · /wf stop 取消）"
            hint.update(text)
            hint.add_class("visible")
            self._import_popup_w.remove_class("visible")
            return
        text = self.ctrl.import_summary()
        if text:
            hint.update(text)
            hint.add_class("visible")
        else:
            hint.update("")
            hint.remove_class("visible")
            self._import_popup_w.remove_class("visible")

    def _pending_images_md(self) -> str:
        """待发送图片在聊天区里的 Markdown 占位行（提交回显用）。"""
        if self.ctrl is None or not self.ctrl.pending_images:
            return ""
        marks = "\n".join(
            f"- {image_placeholder_text(a)}" for a in self.ctrl.pending_images
        )
        return f"\n\n{marks}"

    # ---------------- 状态条上方固定悬停弹窗（完整文件名列表） ----------------
    # 弹窗位于独立布局层（CSS layer），显隐不影响主内容布局；显示位置固定贴住
    # 状态条上方（不跟随光标），设定最小宽度保证可读性。

    _IMPORT_KIND_LABELS = {"image": "图片", "text": "文本", "web": "网页"}

    def _show_import_popup(self) -> None:
        """在状态条上方固定位置显示完整文件名列表弹窗（悬停中段时）。

        已挂载工作流时中段显示的是工作流提示，不弹导入列表。
        """
        if self.ctrl is None:
            return
        if self._pending_wf:
            self._import_popup_w.remove_class("visible")
            return
        items = self.ctrl.import_file_list()
        if not items:
            return
        lines = []
        for item in items:
            name = item.get("name", "")
            if len(name) > 100:
                name = name[:97] + "…"
            label = self._IMPORT_KIND_LABELS.get(item.get("kind", ""))
            toks = item.get("tokens") or 0
            suffix = f" · 约 {toks} tokens" if toks else ""
            lines.append(
                f"[{label}] {name}{suffix}" if label else f"· {name}{suffix}"
            )
        popup = self._import_popup_w
        popup.update("\n".join(lines))
        bar = self.query_one("#usage-bar", Horizontal).region
        if bar.width <= 0 or bar.height <= 0:
            return
        # 绝对定位（offset 相对屏幕左上角）：弹窗左缘与状态条对齐、紧贴其上方。
        # 高度按内容行数估算（内容 + 圆角边框 1 行，封顶屏幕 40%）。
        max_h = int(self.screen.size.height * 0.4)
        h = min(len(lines), max_h) + 1
        top = max(0, bar.y - h - 1)
        popup.styles.position = "absolute"
        popup.styles.offset = (bar.x, top)
        popup.add_class("visible")

    def on_mouse_move(self, event) -> None:
        """鼠标在状态条中段（导入提示）上悬停 → 显示完整文件列表；移出 → 自动消失。"""
        center = getattr(self, "_import_center_w", None)
        if center is None or event.screen_x is None or event.screen_y is None:
            return
        region = center.region
        if region.width <= 0 or region.height <= 0:
            return
        if region.contains(int(event.screen_x), int(event.screen_y)):
            if not self._import_popup_w.has_class("visible"):
                self._show_import_popup()
        else:
            if self._import_popup_w.has_class("visible"):
                self._import_popup_w.remove_class("visible")

    # ---------------- 拖入文件直接导入（终端路径粘贴） ----------------

    def _notify_import_result(self, res: dict) -> None:
        """统一展示 import_targets 的导入结果通知（图片附带 token 估算）。"""
        bits = []
        if res["images_added"]:
            bits.append(f"{res['images_added']} 张图片")
        if res["text_added"]:
            bits.append(f"{res['text_added']} 个文本/网页")
        if bits:
            self.notify(
                f"已导入 {'、'.join(bits)}{self.ctrl.images_tokens_hint()}，发送时自动附带"
            )
        for err in res["errors"][:2]:
            self.notify(err, severity="warning")
        if len(res["errors"]) > 2:
            self.notify(f"…共 {len(res['errors'])} 个失败", severity="warning")

    async def on_chat_input_files_dropped(self, message: ChatInput.FilesDropped) -> None:
        """输入框内拖入文件：终端把路径粘贴进输入框 → 直接导入。"""
        if self.ctrl is None:
            return
        res = self.ctrl.import_targets(message.paths)
        self._refresh_import_status()
        self._notify_import_result(res)
        self.query_one("#chat-input", ChatInput).focus()

    async def on_paste(self, event: events.Paste) -> None:
        """兜底：焦点不在输入框时，粘贴内容若为文件路径也直接导入。"""
        if self.ctrl is None:
            return
        paths = ChatInput._paths_from_paste(event.text)
        if paths is None:
            return
        res = self.ctrl.import_targets(paths)
        self._refresh_import_status()
        self._notify_import_result(res)

    # ---------------- 对话树侧栏 ----------------

    def _rebuild_tree(self) -> None:
        tree_w = self.query_one("#tree", Tree)
        tree_w.clear()
        root = self.ctrl.tree.root if self.ctrl else None
        if root is None:
            tree_w.root.label = "（空）"
            return
        current_id = self.ctrl.tree.current_node.id if self.ctrl.tree.current_node else None
        root_label = f"main: {root.title}"
        tree_w.root.label = _markup_escape(root_label)
        tree_w.root.data = "main"
        tree_w.root.expand()
        for child in root.children:
            self._add_tree_node(tree_w.root, child, current_id)

    def _add_tree_node(self, parent, node, current_id) -> None:
        # 标题由模型生成/用户输入：Tree 标签会按 markup 解析，不转义的话
        # 形如 [b]…[/b]、[xxx="yyy"] 的标题会被当成样式标签吃掉，显示成截断文本
        if node.id == current_id:
            label = f"➤ {node.id}: {node.title}"
        else:
            label = f"{node.id}: {node.title}"
        n = parent.add(_markup_escape(label), data=node.id)
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

    # ---------------- 聊天区（#chat-log 容器：多段正文 Markdown + 工具卡片） ----------------
    # 正文一直 append 到「当前段」(_chat_md)；工具调用时固化当前段、把 ToolCard 插到
    # 其后、再为后续正文新建一段，从而在 Markdown 流中实现真实穿插的控件卡片。

    def _chat_container(self) -> VerticalScroll:
        return self.query_one("#chat-log", VerticalScroll)

    async def _chat_ensure_md(self) -> Markdown:
        """保证存在当前正文段（无则创建并挂到容器末尾）。"""
        if self._chat_md is None:
            md = Markdown("")
            md.styles.height = "auto"
            await self._chat_container().mount(md)
            self._chat_md = md
            self._chat_blocks.append(md)
        return self._chat_md

    async def _chat_stream_append(self, text: str) -> None:
        """正文增量 append 到当前段（防御 markdown 解析异常）。"""
        md = await self._chat_ensure_md()
        await self._safe_append(md, text)

    async def _chat_fix_segment(self) -> None:
        """固化当前正文段（置 None，后续 _chat_append 会新建段）。

        新段里没有任何已打开的引用块，因此思考块状态一并复位：否则下一轮
        思考会以为「块还开着」，丢掉「思考过程」标题和 `> ` 前缀。
        封段同时复位流式段计数（新段从头累计，见 _should_seal_segment）。
        """
        self._chat_md = None
        self._reasoning_open = False
        self._stream_seg_chars = 0
        self._stream_seg_raw = ""

    async def _chat_add_toolcard(self, card: ToolCard) -> None:
        """工具卡片插到当前段之后，并固化当前段（后续正文进新段）。"""
        await self._chat_fix_segment()
        await self._chat_container().mount(card)
        self._chat_blocks.append(card)

    async def _chat_reset(self, text: str) -> None:
        """重建聊天区：清空所有段/卡片，新建首段显示 text（切换节点/清空）。"""
        container = self._chat_container()
        await container.remove_children()
        self._chat_blocks = []
        self._chat_md = None
        self._active_tool_card = None
        self._reasoning_open = False  # 新段没有打开的引用块
        self._stream_seg_chars = 0
        self._stream_seg_raw = ""
        if text:
            md = Markdown(text)
            md.styles.height = "auto"
            await container.mount(md)
            self._chat_md = md
            self._chat_blocks.append(md)

    def _chat_source(self) -> str:
        """聚合所有段的 source（兼容现有 .source 读取）。"""
        parts = []
        for b in self._chat_blocks:
            if isinstance(b, Markdown):
                parts.append(b.source)
            elif isinstance(b, ToolCard):
                parts.append(b.card_summary())
        return "\n".join(parts)

    def _chat_scroll_end(self) -> None:
        # scroll_end 为异步调度，同步触发即可（头部诊断确认滚动生效），
        # 在 _chat_lock 内 await 会等待布局刷新进而挂起。
        self._chat_container().scroll_end(animate=False)

    async def _chat_shrink_lists(self, scroll: bool = True) -> None:
        for b in self._chat_blocks:
            if isinstance(b, Markdown):
                self._shrink_lists(b)
        if scroll:
            self._chat_scroll_end()

    def _node_content(self, node) -> str:
        """把节点渲染为消息区 Markdown 内容（思考过程内联在提问与回答之间）。"""
        comp = (
            self.ctrl.tree.compaction
            if self.ctrl is not None and self.ctrl.tree is not None
            else None
        )
        if comp and comp.get("boundary_id") == node.id:
            # 摘要节点：直接显示压缩后的信息
            return (
                f"# {node.id}: {node.title}\n\n"
                f"**📦 已压缩上下文**\n\n"
                f"在此节点输入将基于以下摘要继续对话；切换到其他节点仍使用完整历史。\n\n"
                f"---\n\n{node.user_msg}"
            )
        content = f"# {node.id}: {node.title}\n\n**你：**\n\n{node.user_msg}\n\n"
        if node.reasoning:
            content += "\n\n" + self._build_reasoning_md(node.reasoning) + "\n\n"
        content += f"**mincli：**\n\n{node.assistant_msg}\n\n"
        content += (
            f"---\n\n"
            f"*📊 输入 {node.input_tokens} tokens | 输出 {node.output_tokens} tokens*"
        )
        return content

    # ---------------- 思考过程（灰色块引用；多轮工具调用时每段各自成块） ----------------

    @staticmethod
    def _build_reasoning_md(text: str) -> str:
        """把思考全文转成灰色块引用 Markdown（节点视图/摘要用）。"""
        return REASONING_HEADER_MD + "\n>\n" + quote_block_text(text)

    @staticmethod
    def _reasoning_chunk_md(chunk: str) -> str:
        """流式思考增量 → 块引用行增量：只按 chunk 内真实换行断行，
        跨 chunk 直接拼接（避免每 token 断行）。

        行首自带 `>` 的思考文本会先转义，防止与 `> ` 前缀叠加成嵌套引用。
        """
        md = ""
        for i, ln in enumerate(chunk.split("\n")):
            safe = escape_leading_quote_markers(ln)
            if i == 0:
                md += safe
            else:
                md += ("\n> " + safe) if safe else "\n>"
        return md

    # ---------------- 工具调用块（代码块样式） ----------------

    @staticmethod
    def _format_tool_args(args_str: str) -> str:
        """把工具参数 JSON 格式化为逐行易读的多行文本（每行宽度受限）。

        目标：避免紧凑 JSON 变成一行超宽文本（横向溢出、视觉上像行内），
        每个键值独占一行，复杂值拍平并按宽度截断。
        """
        try:
            obj = json.loads(args_str)
        except Exception:
            return (args_str or "").strip()[:200]
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                s = json.dumps(v, ensure_ascii=False, indent=2)
                # 嵌套 JSON 拍平成单行，避免多行缩进后仍失控；再按宽度断行
                s = " ".join(s.split())
                if len(s) > 100:
                    s = s[:97] + "…"
                lines.append(f"{k}={s}")
            return "\n".join(lines) if lines else "{}"
        return json.dumps(obj, ensure_ascii=False)[:200]

    def _set_full_view(self, on: bool) -> None:
        """全览模式：隐藏右侧回答区，节点树全宽显示（输入框保留）。"""
        if on == self._full_view:
            return
        self._full_view = on
        self.query_one("#chat-log", VerticalScroll).set_class(on, "overview-hidden")
        self.query_one("#sidebar").set_class(on, "overview")
        self.query_one("#fullview-btn", Button).label = "分栏" if on else "全览"

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
        self.query_one("#chat-log", VerticalScroll).scroll_relative(
            y=delta * (2 if fast else 1), animate=False
        )

    async def _switch_to(self, node_id: str) -> bool:
        """切换到节点：设当前节点 + 刷新树 + 光标跟随 + 消息区显示节点内容。"""
        if self._full_view:
            self._set_full_view(False)  # 全览模式下切换节点 → 自动退出全览
        self._cancel_flush()  # 丢弃未渲染的流式缓冲（视图将被整体重建）
        if not (self.ctrl and self.ctrl.tree.switch_to_node(node_id)):
            return False
        self._rebuild_tree()
        self._select_tree_node(node_id)
        node = self.ctrl.tree.current_node
        await self._chat_reset(self._node_content(node))
        await self._chat_shrink_lists(scroll=False)  # 切换节点：显示节点开头，不滚到底
        self._answer_started = True  # 节点视图已含 **mincli：** 头部
        self._refresh_usage_bar()
        return True

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """点击会话树节点：切换当前节点并显示其内容。"""
        node_id = event.node.data
        if node_id:
            await self._switch_to(node_id)

    # ---------------- 鼠标滚轮（兜底：命中测试可能返回内部容器） ----------------

    def _pointer_over_chat_log(self, event) -> bool:
        try:
            region = self.screen.find_widget(self.query_one("#chat-log", VerticalScroll)).region
        except Exception:
            return False
        return region.contains(int(event.screen_x), int(event.screen_y))

    def _on_mouse_scroll_down(self, event) -> None:
        if self._pointer_over_chat_log(event):
            self.query_one("#chat-log", VerticalScroll).scroll_down(animate=False)
            event.stop()

    def _on_mouse_scroll_up(self, event) -> None:
        if self._pointer_over_chat_log(event):
            self.query_one("#chat-log", VerticalScroll).scroll_up(animate=False)
            event.stop()

    # ---------------- 命令补全弹窗 ----------------

    def on_chat_input_text_changed(self, message: ChatInput.TextChanged) -> None:
        self._update_command_popup(message.text)

    def _completion_providers(self) -> dict:
        """运行时候选来源（对话树编号、工作流名、MCP server 名）。"""
        if self.ctrl is None:
            return {}
        return runtime_providers(self.ctrl)

    def _update_command_popup(self, text: str) -> None:
        """根据输入内容更新输入框上方的命令补全/提示弹窗。

        候选来自 commands.py 的命令树，支持三级（/set → thinking → on）；
        命令写全但没有下一级时，弹窗转为「用法 + 说明」提示。
        """
        popup = self.query_one("#cmd-popup")
        completion = complete_command(text, self._completion_providers())
        if completion is None or (not completion.candidates and not completion.title):
            popup.remove_class("visible")
            self._completion = None
            self._completion_index = 0
            return
        prev_title = self._completion.title if self._completion else ""
        self._completion = completion
        # 同一层级内换候选不重置高亮；换了层级（如 /set → /set thinking）则回到第一项
        if not completion.candidates or completion.title != prev_title:
            self._completion_index = 0
        if self._completion_index >= len(completion.candidates):
            self._completion_index = 0
        popup.add_class("visible")
        self._render_command_list()

    def _render_command_list(self) -> None:
        """渲染弹窗：每个候选一行（用法 + 灰色说明），高亮行滚入可视区。"""
        completion = self._completion
        body = self.query_one("#cmd-popup-body", Static)
        if completion is None:
            return
        # 用 rich.Text 直接拼样式：候选里含 [参数...] / <值> 这类字符，
        # 走 markup 会被当标签解析（这也是旧实现里命令文案不敢带方括号的原因）
        text = Text(completion.title, style="bold")
        if completion.candidates:
            text.append("   ", style="dim")
            text.append(COMPLETION_HINT, style="dim")
            if len(completion.candidates) > 1:
                # 弹窗不显示滚动条，用序号告诉用户还有多少候选没露出来
                text.append(
                    f" · 第 {self._completion_index + 1}/{len(completion.candidates)} 项",
                    style="dim",
                )
        if completion.desc:
            text.append("  " + completion.desc, style="dim")
        for i, cand in enumerate(completion.candidates):
            text.append("\n")
            active = i == self._completion_index
            text.append("→ " if active else "  ", style="bold" if active else "")
            text.append(cand.usage, style="bold" if active else "")
            if cand.desc:
                text.append("  " + cand.desc, style="dim")
        body.update(text)
        self._scroll_completion_into_view()

    def _scroll_completion_into_view(self) -> None:
        """把高亮候选滚进弹窗可视区（否则 Tab 循环到后面的候选就看不见了）。"""
        popup = self.query_one("#cmd-popup")
        row = 1 + self._completion_index  # 第 0 行是标题
        height = popup.content_size.height
        if height <= 0:
            return
        top = int(popup.scroll_y)
        if row < top:
            popup.scroll_y = float(row)
        elif row >= top + height:
            popup.scroll_y = float(row - height + 1)

    def _advance_or_complete(self, reverse: bool = False) -> bool:
        """Tab/Shift+Tab：多候选循环高亮，唯一候选直接补全。返回 True 表示已消费。"""
        completion = self._completion
        candidates = completion.candidates if completion else []
        if not candidates:
            return False
        if len(candidates) > 1:
            step = -1 if reverse else 1
            self._completion_index = (self._completion_index + step) % len(candidates)
            self._render_command_list()
            return True
        return self._complete_from_popup()

    def _complete_from_popup(self) -> bool:
        """把输入补全为当前高亮的候选。返回 True 表示已补全（未执行）。"""
        completion = self._completion
        candidates = completion.candidates if completion else []
        if not candidates:
            return False
        if not (0 <= self._completion_index < len(candidates)):
            self._completion_index = 0
        target = candidates[self._completion_index].insert
        inp = self.query_one("#chat-input", ChatInput)
        if inp.text == target:
            # 已经是补全结果（再按一次应执行命令，而不是卡在原地）
            return False
        # 光标移到行尾再插入补全后缀（直接替换 .text 会把光标重置到行首，
        # 导致后续输入插到错误位置）；候选替换的是最后一个词，可能是改写而非追加
        inp.cursor_location = (
            inp.document.line_count - 1,
            len(inp.document.lines[-1]),
        )
        if target.startswith(inp.text):
            inp.insert(target[len(inp.text):])
        else:
            inp.text = target
            inp.cursor_location = (
                inp.document.line_count - 1,
                len(inp.document.lines[-1]),
            )
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
            await self._chat_reset(WELCOME)
            self._rebuild_tree()
            self._answer_started = False
            self.notify("对话历史已清除")
            self._refresh_usage_bar()
            return True
        if low.startswith("/compact"):
            await self._cmd_compact(cmd)
            return True
        # /wf（及 /workflow 别名）：仅精确前缀 + 空格，避免误吞 /wf1 这类节点跳转
        if (
            low == "/wf"
            or low.startswith("/wf ")
            or low == "/workflow"
            or low.startswith("/workflow ")
        ):
            await self._cmd_wf(cmd)
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
        if await self._cmd_tree(cmd):
            return True
        if low.startswith("/import"):
            # 参数解析走跨平台 split_path_args：兼容 Windows 反斜杠路径（如 C:\a b\x.txt）
            cmd_parts = cmd.split(maxsplit=1)
            targets = split_path_args(cmd_parts[1]) if len(cmd_parts) > 1 else []
            if not targets:
                self.notify(usage_line("/import"), severity="warning")
                return True
            if targets[0].lower() in ("clear", "c"):
                n = ctrl.clear_imports()
                self._refresh_import_status()
                self.notify(f"已清除 {n} 个待导入文件")
                return True
            self.notify("正在导入…")
            res = ctrl.import_targets(targets)
            self._refresh_import_status()
            self._notify_import_result(res)
            return True

        if low.startswith("/files"):
            await self._cmd_files(cmd)
            return True

        m = re.match(r"^/([A-Za-z]+\d+|main)$", cmd)
        if m and ctrl.tree and m.group(1) in ctrl.tree.nodes:
            await self._switch_to(m.group(1))
            return True
        if cmd.startswith("/"):
            self.notify(f"未知命令: {cmd}。输入 /help 查看可用命令", severity="warning")
            return True
        return False

    async def _chat_append(self, markdown: str) -> None:
        """向消息区追加一段 Markdown（带分隔线并滚动到底）。"""
        await self._chat_stream_append(f"\n\n---\n\n{markdown}")
        await self._chat_shrink_lists()

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
        err = open_path_with_os(filepath)
        if err:
            self.notify(
                f"打开文件失败（{err}）；文件路径: {filepath}",
                severity="warning",
                timeout=8,
            )
        else:
            self.notify(f"已用系统默认程序打开节点 {node.id} 的回答")

    async def _cmd_help(self) -> None:
        # 命令清单由 commands.py 的命令规格生成（与补全弹窗、用法提示同源）
        await self._chat_append(help_markdown() + HELP_TAIL)

    async def _cmd_set(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=2)
        ctrl = self.ctrl
        usage = usage_line("/set")
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
                self.notify(usage_line("/set thinking"), severity="warning")
        elif sub == "effort" and len(parts) == 3:
            if ctrl.set_effort(parts[2]):
                self.notify(f"推理强度已设置为: {ctrl.reasoning_effort}")
            else:
                self.notify(usage_line("/set effort"), severity="warning")
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
        elif sub == "detail" and len(parts) == 3:
            if ctrl.set_detail(parts[2].lower()):
                if ctrl.image_detail == "low":
                    self.notify(
                        "图片 detail 已设置为: low（本地图片将以内联 base64 发送，"
                        "detail 才真正生效；file_id 方式会忽略 detail）"
                    )
                else:
                    self.notify(
                        f"图片 detail 已设置为: {ctrl.image_detail}"
                        "（本地图片优先上传 Files API 复用；该档与上传后的处理一致）"
                    )
            else:
                self.notify(usage_line("/set detail"), severity="warning")
        elif sub == "file_confirm" and len(parts) == 3:
            arg = parts[2].lower()
            if arg in ("on", "1", "true"):
                ctrl.set_file_confirm(True)
                self.notify("写文件/编辑文件确认已开启")
            elif arg in ("off", "0", "false"):
                ctrl.set_file_confirm(False)
                self.notify("写文件/编辑文件确认已关闭（AI 可直接写入/修改文件）", severity="warning")
            else:
                self.notify(usage_line("/set file_confirm"), severity="warning")
        elif sub == "show":
            ctrl = self.ctrl
            lines = [
                "**当前配置**",
                "",
                f"- **对话树**: 树 {ctrl.active_number}（{color_label(ctrl.tree_color())}，共 {len(ctrl.tree_numbers())} 棵）",
                f"- **系统提示词**: {ctrl.current_system}",
                f"- **温度**: {ctrl.current_temperature}",
                f"- **模型**: {ctrl.current_model}",
                f"- **思考模式**: {'开' if ctrl.thinking_enabled else '关'} | 推理强度: {ctrl.reasoning_effort}",
                f"- **审核层级**: {ctrl.audit_level} - {AUDIT_LABELS[ctrl.audit_level]}",
                f"- **文件写入确认**: {'开' if ctrl.file_confirm else '关（AI 可直接写入/修改文件）'}",
                f"- **命令工作目录**: {ctrl.workspace or '（未设置，默认 mincli 启动目录）'}",
                f"- **图片 detail**: {ctrl.image_detail}"
                + ("（本地图片内联发送，真正生效）" if ctrl.image_detail == "low"
                   else "（本地图片上传复用 file_id；该档与上传后处理一致）"),
            ]
            if ctrl.tree.current_node:
                lines.append(f"- **当前节点**: {ctrl.tree.current_node.id} ({ctrl.tree.current_node.title})")
            await self._chat_append("\n".join(lines))
        else:
            self.notify(usage, severity="warning")

    # ---------------- /files：Files API 已上传文件管理 ----------------

    @staticmethod
    def _fmt_mib(num_bytes: int) -> str:
        """字节 → MiB 文本（小于 100KiB 时用 KiB，避免全显示成 0.00 MiB）。"""
        num_bytes = int(num_bytes or 0)
        if num_bytes < 100 * 1024:
            return f"{num_bytes / 1024:.0f} KiB"
        return f"{num_bytes / 1024 / 1024:.2f} MiB"

    @staticmethod
    def _fmt_ts(ts) -> str:
        if not ts:
            return "—"
        try:
            return datetime.datetime.fromtimestamp(int(ts)).strftime("%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return "—"

    def _resolve_file_ref(self, ref: str) -> Optional[str]:
        """把 /files 参数解析为 file_id：纯数字按最近列表序号，否则原样当 ID。

        序号优先用最近一次 /files list 的结果（省一次网络请求）；没列过或序号
        超出范围时按默认页拉一次列表再解析，让 /files delete 2 可以独立使用。
        """
        ref = (ref or "").strip()
        if not ref.isdigit() or self.ctrl is None:
            return ref or None
        index = int(ref)
        listing = list(self._files_listing)
        if not listing:
            try:
                page = self.ctrl.files_list(FILES_LIST_PAGE)
            except FilesAPIError:
                return None
            listing = page["items"]
        if 1 <= index <= len(listing):
            return listing[index - 1]["id"]
        return None

    def _file_ref_error(self, ref: str) -> str:
        """序号解析失败时的提示（带上当前列表范围，便于改用正确的序号或先列表）。"""
        total = len(self._files_listing)
        if total:
            return f"找不到序号 {ref}（当前列表只有 {total} 个）；/files list <N> 可看更多"
        return f"找不到序号 {ref}；请先 /files list 查看列表"

    def _render_files_table(self, page: dict) -> str:
        """把一页文件列表渲染成 Markdown 表格（带序号，便于按序号删除）。"""
        items = page["items"]
        if not items:
            return "**已上传图片文件（Files API）**\n\n（空）"
        total_bytes = sum(int(f.get("bytes", 0) or 0) for f in items)
        head = (
            f"**已上传图片文件（Files API）**\n\n"
            f"共 {len(items)} 个（最新在前）· 合计 {self._fmt_mib(total_bytes)}"
            f" · 配额 {FILES_MAX_COUNT} 个 / {FILES_MAX_BYTES // 1024 // 1024 // 1024} GiB"
        )
        if page.get("has_more"):
            head += f" · 还有更早的文件未列出（`/files list <N>`，单页上限 {FILES_LIST_PAGE_MAX}）"
        lines = [
            head,
            "",
            "| # | ID | 文件名 | 大小 | 创建时间 | 过期 |",
            "|---|---|---|---|---|---|",
        ]
        for i, f in enumerate(items, start=1):
            expires = self._fmt_ts(f.get("expires_at")) if f.get("expires_at") else "永久"
            # 文件名来自 API：转义 `|` 并截断，避免超长名字（有些工具用 ID 当文件名）
            # 把 Markdown 表格撑到换行；完整名字用 `/files info <序号>` 查看
            name = str(f.get("name", "")).replace("|", "\\|")
            if len(name) > 36:
                name = name[:35] + "…"
            lines.append(
                f"| {i} | `{f.get('id', '')}` | {name} "
                f"| {self._fmt_mib(f.get('bytes', 0))} | {self._fmt_ts(f.get('created_at'))} "
                f"| {expires} |"
            )
        lines.append(
            "\n查询: `/files info <ID|序号>` · 删除: `/files delete <ID|序号>`"
            " · 清理未引用: `/files clean`"
        )
        return "\n".join(lines)

    async def _cmd_files_list(self, args: list) -> None:
        """`/files list [N]`：列出最近上传的文件（默认 20 条）。"""
        limit = FILES_LIST_PAGE
        if args:
            if not args[0].isdigit():
                self.notify(usage_line("/files list"), severity="warning")
                return
            limit = max(1, min(int(args[0]), FILES_LIST_PAGE_MAX))
        try:
            page = self.ctrl.files_list(limit)
        except FilesAPIError as e:
            self.notify(str(e), severity="error")
            return
        self._files_listing = page["items"]
        await self._chat_append(self._render_files_table(page))

    async def _cmd_files_info(self, args: list) -> None:
        """`/files info <ID|序号>`：查询单个文件（GET /files/{id}）。"""
        if not args:
            self.notify(usage_line("/files info"), severity="warning")
            return
        ref = args[0]
        file_id = self._resolve_file_ref(ref)
        if file_id is None:
            self.notify(self._file_ref_error(ref), severity="warning")
            return
        try:
            info = self.ctrl.files_retrieve(file_id)
        except FilesAPIError as e:
            self.notify(str(e), severity="error")
            return
        expires = self._fmt_ts(info["expires_at"]) if info.get("expires_at") else "永久有效"
        await self._chat_append(
            "**文件信息（Files API）**\n\n"
            f"- **ID**: `{info['id']}`\n"
            f"- **文件名**: {info['name'] or '（无）'}\n"
            f"- **大小**: {self._fmt_mib(info['bytes'])}\n"
            f"- **创建时间**: {self._fmt_ts(info['created_at'])}\n"
            f"- **过期**: {expires}"
        )

    async def _cmd_files_delete(self, args: list) -> None:
        """`/files delete <ID|序号>`：删除文件，并同步清掉对话树里的失效引用。"""
        if not args:
            self.notify(usage_line("/files delete"), severity="warning")
            return
        ref = args[0]
        file_id = self._resolve_file_ref(ref)
        if file_id is None:
            self.notify(self._file_ref_error(ref), severity="warning")
            return
        name = next(
            (f.get("name", "") for f in self._files_listing if f.get("id") == file_id),
            "",
        )
        try:
            self.ctrl.files_delete(file_id)
        except FilesAPIError as e:
            self.notify(str(e), severity="error")
            return
        self._files_listing = [f for f in self._files_listing if f.get("id") != file_id]
        self.notify(f"已删除文件 {file_id}" + (f"（{name}）" if name else ""))

    def _on_files_clean_confirmed(self, ids: list, ok: bool) -> None:
        """确认后逐个删除未被引用的远端文件（失败只提示，不中断其余删除）。"""
        if not ok or not ids:
            if not ok:
                self.notify("已取消清理")
            return
        done, failed = 0, []
        for fid in ids:
            try:
                self.ctrl.files_delete(fid)
                done += 1
            except FilesAPIError as e:
                failed.append(f"{fid}: {e}")
        self._files_listing = [f for f in self._files_listing if f.get("id") not in set(ids)]
        self.notify(f"已清理 {done} 个未被引用的远端文件" + (f"，{len(failed)} 个失败" if failed else ""))
        for msg in failed[:2]:
            self.notify(msg, severity="warning")

    async def _cmd_files_clean(self) -> None:
        """`/files clean`：列出所有对话树都没引用的远端文件，确认后删除。"""
        try:
            res = self.ctrl.files_clean()
        except FilesAPIError as e:
            self.notify(str(e), severity="error")
            return
        if not res["total"]:
            self.notify("Files API 上没有任何文件")
            return
        if not res["unused"]:
            self.notify(f"没有可清理的文件（共 {res['total']} 个，全部仍被对话树引用）")
            return
        unused = res["unused"]
        preview = "\n".join(
            f"{f.get('name', '') or f.get('id', '')}（{self._fmt_mib(f.get('bytes', 0))}）"
            for f in unused[:10]
        )
        if len(unused) > 10:
            preview += f"\n…共 {len(unused)} 个"
        self._ask_confirm(
            f"清理 {len(unused)} 个未被引用的远端文件？",
            f"{preview}\n\n"
            f"这些文件不再被任何对话树引用（共 {res['total']} 个，"
            f"仍被引用 {res['used']} 个）。删除后无法恢复；"
            "若同一个 API Key 也被其它工具用来上传文件，那些文件同样会被删掉。",
            lambda ok, ids=[f["id"] for f in unused]: self._on_files_clean_confirmed(ids, ok),
        )

    async def _cmd_files(self, cmd: str) -> None:
        """`/files` 命令族：list / info / delete / clean。"""
        parts = cmd.split()
        sub = parts[1].lower() if len(parts) > 1 else "list"
        args = parts[2:]
        if sub in ("list", "ls"):
            await self._cmd_files_list(args)
        elif sub == "info":
            await self._cmd_files_info(args)
        elif sub in ("delete", "rm", "del"):
            await self._cmd_files_delete(args)
        elif sub == "clean":
            await self._cmd_files_clean()
        else:
            self.notify(usage_line("/files"), severity="warning")

    async def _cmd_tree(self, cmd: str) -> bool:
        parts = cmd.split()
        low = parts[0].lower()
        tree = self.ctrl.tree
        current_id = tree.current_node.id if tree.current_node else None

        if low == "/tree":
            await self._cmd_tree_list(parts[1:])
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
                if parent and not (await self._switch_to(parent.id)):
                    self.notify("返回父节点失败", severity="error")
            else:
                self.notify("已在根节点", severity="warning")
            return True
        if low == "/home":
            if tree.root:
                await self._switch_to(tree.root.id)
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
            nids = parts[1:]
            if not nids:
                self.notify(usage_line("/delete"), severity="warning")
                return True
            missing = [n for n in nids if n not in tree.nodes]
            if missing:
                self.notify(f"未找到节点: {'、'.join(missing)}", severity="error")
                return True
            roots = [n for n in nids if n == "main" or (tree.root and n == tree.root.id)]
            if roots:
                if len(roots) == len(nids):
                    self.notify("不能删除根节点", severity="warning")
                    return True
                self.notify(f"根节点不可删除，已忽略: {'、'.join(roots)}", severity="warning")
                nids = [n for n in nids if n not in roots]
            self._ask_confirm(
                "删除节点",
                f"确定要删除节点 {'、'.join(nids)} 及其所有子节点吗？",
                lambda ok, ids=nids: self._on_delete_confirmed(ids, ok),
            )
            return True
        return False

    async def _cmd_tree_list(self, args: list) -> None:
        """对话树命令族：/tree | /tree N | /tree new | /tree delete N | /tree N tools。"""
        ctrl = self.ctrl
        usage = usage_line("/tree")
        if not args:
            numbers = ctrl.tree_numbers()
            if not numbers:
                await self._chat_append(
                    f"**对话树**\n\n（尚未创建）\n\n{usage}"
                )
                return
            lines = ["**对话树**", ""]
            for number in numbers:
                summary = ctrl.tree_summary(number)
                mark = ctrl.tree_marks.get(number)
                state = ""
                if mark == "done":
                    state = "（后台完成）"
                elif mark == "error":
                    state = "（后台出错）"
                here = "（当前）" if number == ctrl.active_number else ""
                caps = ctrl.tree_caps(number)
                tools = "系统工具" if caps["system_tools"] else "无系统工具"
                if caps["mcp_tools"]:
                    tools += f" + {len(caps['mcp_tools'])} 个外置工具"
                lines.append(
                    f"- **树 {number}** · {color_label(ctrl.tree_color(number))} · "
                    f"{summary['nodes']} 个节点 · {tools}{state}{here}"
                )
            lines.append("")
            lines.append(command_block("/tree"))
            await self._chat_append("\n".join(lines))
            return

        sub = args[0].lower()
        if sub == "new":
            self._open_tree_wizard()
            return
        if sub in ("delete", "rm", "del"):
            if len(args) < 2 or not args[1].isdigit():
                self.notify(usage_line("/tree delete"), severity="warning")
                return
            number = int(args[1])
            if not ctrl.has_trees or number not in ctrl.tree_numbers():
                self.notify(f"对话树 {number} 不存在", severity="warning")
                return
            summary = ctrl.tree_summary(number)
            self._ask_confirm(
                "删除对话树",
                f"确定删除对话树 {number}（{summary['nodes']} 个节点）吗？\n"
                "该树的数据文件会一并删除，编号不再复用。",
                lambda ok, n=number: self._on_tree_delete_confirmed(n, ok),
            )
            return
        if sub.isdigit():
            number = int(sub)
            if len(args) > 1 and args[1].lower() in ("tools", "tool", "cap", "caps"):
                self._open_tree_wizard(edit_number=number)
                return
            await self._switch_tree(number)
            return
        self.notify(usage, severity="warning")

    async def _on_tree_delete_confirmed(self, number: int, ok: bool) -> None:
        if not ok:
            self.notify("已取消删除")
            return
        if self._turn_active and self.ctrl.generating_number == number:
            self.notify("这棵树正在生成，先按 Esc 打断再删除", severity="warning")
            return
        result = self.ctrl.delete_tree(number)
        if not result.get("ok"):
            self.notify(f"对话树 {number} 不存在", severity="error")
            return
        self._refresh_tree_ui()
        await self._render_active_tree()
        self.notify(f"已删除对话树 {number}")
        if not self.ctrl.has_trees:
            self._open_tree_wizard()

    async def _on_delete_confirmed(self, nids: list, ok: bool) -> None:
        """确认弹窗回调：ok=True 时批量删除（App 消息泵空闲时才被调用）。"""
        if not ok:
            self.notify("已取消删除")
            return
        tree = self.ctrl.tree
        prev_current_id = tree.current_node.id if tree.current_node else None
        result = self.ctrl.delete_nodes(nids)
        self.ctrl._cleanup_temp_files(keep_ids=set(tree.nodes.keys()))
        if prev_current_id is not None and prev_current_id not in tree.nodes:
            # 当前节点（或其祖先）被删：模型已把当前移到父节点/根 —— 自动跳转过去，
            # 让聊天区/树光标/用量状态跟随新的当前节点（_switch_to 内部重建树并选中）
            if tree.current_node is not None:
                await self._switch_to(tree.current_node.id)
            else:  # 理论不会发生（根节点不可删），防御：退回欢迎页
                self._rebuild_tree()
                await self._chat_reset(WELCOME)
        else:
            self._rebuild_tree()
            if tree.current_node:
                self._select_tree_node(tree.current_node.id)
        if result["deleted"]:
            self.notify(f"已删除 {len(result['deleted'])} 个节点（含其子节点及关联图片文件）")
        else:
            self.notify("没有可删除的节点", severity="warning")

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

        self.notify(usage_line("/model"), severity="warning")

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
            self.notify(usage_line("/model register"), severity="warning")
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
                self.notify(usage_line("/mcp remove"), severity="warning")
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
            try:
                self.ctrl.mcp_reload()
            except Exception as e:
                self.notify(f"MCP 重载失败: {e}", severity="error")
                return
            # 重连在后台进行：完成后 MCP 客户端会通过日志出口报「MCP 就绪」
            self.notify("正在重新连接 MCP servers…")
            self._refresh_usage_bar()
            self._watch_mcp_ready()
        else:
            self.notify(usage_line("/mcp"), severity="warning")

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
            connecting = self.ctrl.mcp_connecting
            for name in sorted(status):
                st = status[name]
                if name == "mincli":
                    cmd = "内置 server"
                else:
                    cfg = servers.get(name, {})
                    cmd = cfg.get("url") or cfg.get("command", "")
                    if cfg.get("headers"):
                        cmd += f"（带 {len(cfg['headers'])} 个请求头）"
                if st["connected"]:
                    state = "已连接"
                elif connecting:
                    state = "连接中…"
                else:
                    state = "未连接"
                lines.append(f"| {name} | {cmd} | {st['tools']} | {state} |")
        if not servers:
            lines.append("\n（未配置第三方 server，可用 /mcp add 添加）")
        await self._chat_append("\n".join(lines))

    def _mcp_add(self, rest: str) -> None:
        servers = load_mcp_servers()
        is_url = lambda s: bool(re.match(r"^https?://", s))
        # 跨平台解析：支持带引号的参数、--header 'K: V' 与 Windows 反斜杠命令路径
        tokens = split_path_args(rest)
        if len(tokens) < 2:
            self.notify(
                usage_line("/mcp add") + "（本地命令或远程 URL；--header 'K: V' 可重复）",
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
        """/compact —— 压缩当前分支全部对话，新建摘要节点。"""
        parts = cmd.strip().split()
        if len(parts) > 1:
            self.notify(usage_line("/compact") + "（不支持参数）", severity="warning")
            return
        ctrl = self.ctrl
        if not ctrl.tree or ctrl.tree.current_node is None:
            self.notify("当前没有对话可压缩", severity="warning")
            return
        if (
            ctrl.tree.compaction
            and ctrl.tree.compaction.get("boundary_id") == ctrl.tree.current_node.id
        ):
            self.notify(
                "当前节点已是压缩摘要节点（切换回其他节点仍使用完整历史）",
                severity="warning",
            )
            return
        self.notify("正在压缩上下文…")
        stats = await asyncio.to_thread(ctrl.compact_history, emit=self._emit_from_thread)
        if stats is None:
            self.notify("无可压缩的对话（或压缩失败）", severity="warning")
            return
        if stats.get("blocked"):
            self.notify("当前节点已是压缩摘要节点", severity="warning")
            return
        # 切换到新建的摘要节点：聊天区直接显示压缩后的信息
        await self._switch_to(stats["node_id"])
        self.notify(
            f"✅ 已压缩 {stats['nodes_compressed']} 轮 → 新节点 {stats['node_id']}（摘要）；"
            f"Token {stats['before_tokens']:,} → {stats['after_tokens']:,}"
            f"（节省 {stats['saved_tokens']:,}）"
        )

    # ---------------- 工作流（/wf） ----------------

    async def _cmd_wf(self, cmd: str) -> None:
        """工作流命令入口（/wf 与 /workflow 等价）。"""
        ctrl = self.ctrl
        if ctrl is None:
            self.notify("控制器未就绪", severity="error")
            return
        parts = cmd.strip().split(maxsplit=1)
        rest = parts[1].strip() if len(parts) > 1 else ""
        # 跨平台解析：工作流的键=值里可能含 Windows 反斜杠路径
        tokens = split_path_args(rest)
        sub = tokens[0].lower() if tokens else ""
        name = tokens[1] if len(tokens) > 1 else ""

        if sub in ("", "help", "-h", "--help"):
            await self._chat_append(
                "**工作流命令**\n\n"
                f"{command_block('/wf')}\n\n"
                "- 不带 `<名>` 的用法见上；工作流长期保存于 `~/.mincli/workflows.json`，重启后仍在\n"
                "- `/wf save` 会把每次变化的数据自动抽成 {变量}，`/wf run` 未提供的变量由模型结合当前情况推断"
            )
            return
        if sub in ("list", "ls"):
            await self._cmd_wf_list()
            return
        if sub == "show":
            if not name:
                self.notify(usage_line("/wf show"), severity="warning")
                return
            wf = ctrl.wf_get(name)
            if wf is None:
                self.notify(f"工作流「{name}」不存在（/wf list 查看）", severity="warning")
                return
            await self._chat_append(
                f"**工作流 {wf.name}**（运行 {wf.run_count} 次）\n\n{wf.doc}"
            )
            return
        if sub == "save":
            if not name:
                self.notify(usage_line("/wf save"), severity="warning")
                return
            start_id = tokens[2] if len(tokens) > 2 else None
            if ctrl.wf_get(name) is not None:
                self._ask_confirm(
                    "覆盖工作流",
                    f"工作流「{name}」已存在，重新提炼会覆盖旧版本，是否继续？",
                    lambda ok, n=name, s=start_id: self._on_wf_save_confirmed(n, s, ok),
                )
                return
            await self._wf_save_exec(name, start_id)
            return
        if sub == "use":
            if not name:
                self.notify(usage_line("/wf use") + "（/wf stop 取消挂载）", severity="warning")
                return
            if ctrl.wf_get(name) is None:
                self.notify(f"工作流「{name}」不存在（/wf list 查看）", severity="warning")
                return
            self._pending_wf = name
            self._refresh_import_status()
            self.notify(
                f"已挂载工作流「{name}」：发送下一条消息即按工作流执行（/wf stop 取消）"
            )
            return
        if sub in ("stop", "unuse"):
            if self._pending_wf:
                self.notify(f"已解除工作流「{self._pending_wf}」的挂载")
            else:
                self.notify("当前没有已挂载的工作流")
            self._pending_wf = None
            self._refresh_import_status()
            return
        if sub == "run":
            await self._cmd_wf_run(tokens)
            return
        if sub == "delete":
            if not name:
                self.notify(usage_line("/wf delete"), severity="warning")
                return
            if ctrl.wf_get(name) is None:
                self.notify(f"工作流「{name}」不存在", severity="warning")
                return
            self._ask_confirm(
                "删除工作流",
                f"确定删除工作流「{name}」吗？此操作不可恢复。",
                lambda ok, n=name: self._on_wf_delete_confirmed(n, ok),
            )
            return
        if sub == "rename":
            new_name = tokens[2] if len(tokens) > 2 else ""
            if not name or not new_name:
                self.notify(usage_line("/wf rename"), severity="warning")
                return
            err = ctrl.wf_rename(name, new_name)
            if err:
                self.notify(err, severity="warning")
            else:
                if self._pending_wf == name:
                    self._pending_wf = new_name
                    self._refresh_import_status()
                self.notify(f"已重命名：{name} → {new_name}")
            return
        if sub == "edit":
            if not name:
                self.notify(usage_line("/wf edit"), severity="warning")
                return
            if ctrl.wf_get(name) is None:
                self.notify(f"工作流「{name}」不存在", severity="warning")
                return
            if len(tokens) > 2:
                request = " ".join(tokens[2:])
                self.notify(f"正在按你的要求修订工作流「{name}」…")
                res = await asyncio.to_thread(ctrl.wf_revise, name, request)
                if res.get("status") == "error":
                    self.notify(res.get("message", "修订失败"), severity="error")
                else:
                    self.notify(f"✅ 工作流「{name}」已按你的要求更新")
                return
            self._wf_start_editor(name)
            return
        self.notify(usage_line("/wf"), severity="warning")

    async def _cmd_wf_list(self) -> None:
        data = self.ctrl.wf_list() if self.ctrl else []
        if not data:
            await self._chat_append(
                "**工作流**（暂无）\n\n把当前操作保存为可复用工作流：`/wf save <名>`；"
                "已有工作流用 `/wf use <名>` 挂载到下次输入、`/wf run <名>` 立即执行。"
            )
            return
        lines = [
            "**工作流列表**",
            "",
            "| 名称 | 目标 | 步骤 | 变量 | 运行 | 更新于 |",
            "|---|---|---|---|---|---|",
        ]
        for item in data:
            upd = (item.get("updated_at") or "")[:16].replace("T", " ")
            goal = item.get("goal") or "—"
            if len(goal) > 40:
                goal = goal[:40] + "…"
            lines.append(
                f"| `{item['name']}` | {goal} | {item.get('steps', 0)} "
                f"| {len(item.get('vars') or [])} | {item.get('run_count', 0)} | {upd} |"
            )
        lines += [
            "",
            "使用: `/wf use <名>` 挂载下次输入 · `/wf run <名>` 立即执行 · "
            "`/wf show <名>` 查看 · `/wf edit <名> <修改要求>` 修改",
        ]
        await self._chat_append("\n".join(lines))

    async def _wf_save_exec(self, name: str, start_id: Optional[str]) -> None:
        """后台提炼并保存工作流（force=True，覆盖同名）。"""
        ctrl = self.ctrl
        if ctrl is None:
            return
        self.notify(f"正在从对话提炼工作流「{name}」…")
        res = await asyncio.to_thread(ctrl.wf_save, name, start_id, True)
        if res.get("status") == "error":
            self.notify(res.get("message", "保存失败"), severity="error")
            return
        if res.get("from") == "fallback":
            self.notify(
                f"已保存工作流「{name}」（未能自动提炼，原始记录已存档，"
                f"可用 /wf edit {name} 修改）",
                severity="warning",
                timeout=8,
            )
        else:
            n = res.get("nodes", 0)
            v = len(res.get("placeholders") or [])
            self.notify(
                f"✅ 已提炼保存工作流「{name}」（{n} 轮 / {v} 个变量）；"
                f"/wf use {name} 挂载到下次输入",
                timeout=8,
            )

    async def _on_wf_save_confirmed(
        self, name: str, start_id: Optional[str], ok: bool
    ) -> None:
        if not ok:
            self.notify("已取消")
            return
        await self._wf_save_exec(name, start_id)

    async def _cmd_wf_run(self, tokens: list) -> None:
        name = tokens[1] if len(tokens) > 1 else ""
        if not name:
            self.notify(usage_line("/wf run") + "（位置参数按变量顺序填充）", severity="warning")
            return
        ctrl = self.ctrl
        if ctrl is None:
            return
        wf = ctrl.wf_get(name)
        if wf is None:
            self.notify(f"工作流「{name}」不存在（/wf list 查看）", severity="warning")
            return
        values: dict = {}
        positionals: list = []
        for tok in tokens[2:]:
            if "=" in tok:
                k, _, v = tok.partition("=")
                values[k.strip()] = v
            else:
                positionals.append(tok)
        if positionals:
            phs = wf.placeholders()
            for i, pv in enumerate(positionals):
                if i >= len(phs):
                    break
                key = phs[i]
                if key not in values:
                    values[key] = pv
            if len(positionals) > len(phs):
                ph_txt = "、".join("{" + p + "}" for p in phs) or "无"
                self.notify(f"多余的位置参数已忽略（该工作流变量：{ph_txt}）", severity="warning")
        composed = ctrl.wf_compose(name, values=values)
        if composed is None:
            self.notify(f"工作流「{name}」不存在", severity="warning")
            return
        self.notify(f"已开始执行工作流「{name}」")
        await self._send_user_text(composed)

    async def _on_wf_delete_confirmed(self, name: str, ok: bool) -> None:
        if not ok:
            self.notify("已取消")
            return
        if self.ctrl and self.ctrl.wf_delete(name):
            if self._pending_wf == name:
                self._pending_wf = None
                self._refresh_import_status()
            self.notify(f"已删除工作流「{name}」")
        else:
            self.notify(f"删除失败：工作流「{name}」不存在", severity="warning")

    # ---------------- 工作流编辑器回写（跨平台） ----------------

    def _wf_start_editor(self, name: str) -> None:
        """用系统默认编辑器打开工作流临时文件，保存改动后自动回写（跨平台）。

        macOS `open -e`（强制文本编辑器）；Windows `os.startfile`；
        其他 Unix `xdg-open`。失败时提示改用模型修订方式。
        """
        ctrl = self.ctrl
        if ctrl is None:
            return
        path = ctrl.wf_export_temp(name)
        if path is None:
            self.notify(f"工作流「{name}」不存在", severity="warning")
            return
        err = open_path_with_os(path, prefer_text_editor=True)
        if err:
            self.notify(
                f"打开编辑器失败（{err}）；可改用 /wf edit {name} <修改要求> "
                f"由模型修订，或手动编辑 {path}",
                severity="warning",
                timeout=8,
            )
            return
        self._wf_stop_edit()
        self._wf_edit_path = path
        self._wf_edit_name = name
        try:
            self._wf_edit_mtime = os.path.getmtime(path)
        except OSError:
            self._wf_edit_mtime = None
        self._wf_edit_deadline = time.time() + 60
        self._wf_edit_timer = self.set_interval(1.0, self._wf_editor_poll)
        self.notify(
            f"已在编辑器中打开工作流「{name}」：保存改动后自动更新（60 秒内）"
        )

    def _wf_editor_poll(self) -> None:
        """轮询编辑器临时文件：内容变化 → 回写工作流文档。"""
        if not self._wf_edit_path or not self._wf_edit_name:
            self._wf_stop_edit()
            return
        try:
            mtime = os.path.getmtime(self._wf_edit_path)
        except OSError:
            self._wf_stop_edit()
            return
        if mtime != self._wf_edit_mtime:
            try:
                with open(self._wf_edit_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                return
            self._wf_edit_mtime = mtime
            name = self._wf_edit_name
            if self.ctrl and self.ctrl.wf_import_text(name, text):
                self.notify(f"工作流「{name}」已从编辑器更新")
        if self._wf_edit_deadline is not None and time.time() > self._wf_edit_deadline:
            self._wf_stop_edit()

    def _wf_stop_edit(self) -> None:
        """停止编辑器轮询并清理状态（保留临时文件供用户继续编辑/另存）。"""
        if self._wf_edit_timer is not None:
            try:
                self._wf_edit_timer.stop()
            except Exception:
                pass
            self._wf_edit_timer = None
        self._wf_edit_path = None
        self._wf_edit_name = None
        self._wf_edit_mtime = None
        self._wf_edit_deadline = None

    # ---------------- 消息发送与流式渲染 ----------------

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        if self.ctrl is None:
            self.notify("控制器未就绪（请检查 DEEPSEEK_API_KEY）", severity="error", timeout=5)
            return
        if self._complete_from_popup():
            return  # 命令未输完：先补全，等再次 Enter 执行
        if await self._handle_command(event.text):
            return
        await self._send_user_text(event.text)

    async def _send_user_text(self, text: str) -> None:
        """发送一条用户消息（普通输入或 /wf run）。

        若已挂载工作流（/wf use），先把它合成为“按工作流执行”的消息再发送，
        挂载随之解除（一次性）。

        多对话树：同一时刻只允许一棵树在生成。别的树在后台生成时切过来可以看，
        但发送会被拒绝并提示是哪个编号的树在占用。
        """
        if self.ctrl is None:
            self.notify("控制器未就绪（请检查 DEEPSEEK_API_KEY）", severity="error", timeout=5)
            return
        if not self.ctrl.has_trees:
            self.notify("请先新建对话树（/tree new）", severity="warning")
            return
        if self._turn_active:
            busy = self.ctrl.generating_number
            where = f"对话树 {busy}" if busy else "当前轮"
            self.notify(
                f"{where} 正在生成，等它结束或按 Esc 打断后再发送", severity="warning"
            )
            return
        if self._pending_wf:
            wf_name = self._pending_wf
            self._pending_wf = None
            composed = self.ctrl.wf_compose(wf_name, typed=text)
            self._refresh_import_status()
            if composed is None:
                self.notify(f"工作流「{wf_name}」不存在，已解除挂载", severity="warning")
                return
            text = composed
            self.notify(f"已按工作流「{wf_name}」执行")
        self._cancel_flush()  # 新消息开始前丢弃上一轮残留的流式缓冲
        img_md = self._pending_images_md()
        await self._chat_stream_append(f"\n\n---\n\n**你**\n\n{text}{img_md}")
        await self._chat_shrink_lists()
        self._refresh_import_status()  # 待发送图片随消息进入发送流程，先隐藏提示行
        self._stream_active = False
        self._reasoning_open = False
        self._answer_started = False
        self._set_turn_active(True)
        self._refresh_tree_badge()
        self.run_worker(
            lambda: self._run_message(text),
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
        finally:
            self.call_from_thread(self._set_turn_active, False)

    def _emit_from_thread(self, ev: ControllerEvent) -> None:
        self.call_from_thread(self._handle_event, ev)

    async def _handle_event(self, ev: ControllerEvent) -> None:
        """主线程处理控制器事件。

        多对话树：事件带发起生成的那棵树编号；用户切走之后，属于别的树的事件
        一律不渲染到消息区（只在侧栏留「完成/出错」标记），否则流式内容会串台。

        stream 事件（SSE 按 token 级产生，频率极高）先累积到缓冲，由定时器
        每 80ms 批量渲染一次，避免每个 token 都触发 Markdown 组件的
        mount/布局/重绘 → 渲染卡顿、主线程事件积压。
        低频事件（node_created/tool/status/done/error）先冲刷缓冲再即时处理，
        保证渲染顺序正确。
        """
        if self.ctrl is not None and ev.tree and ev.tree != self.ctrl.active_number:
            self._handle_background_event(ev)
            return
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
        self._stream_seg_chars = 0
        self._stream_seg_raw = ""

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
            t0 = time.process_time()
            try:
                await self._render_stream_chunk(content, reasoning)
            finally:
                self._adapt_flush_interval(time.process_time() - t0)

    def _adapt_flush_interval(self, cpu: float) -> None:
        """按本次刷新的真实 CPU 耗时自适应调整刷新间隔。

        刷新越贵 → 间隔越长，保证渲染占用不超过约 1/4 的 CPU，长输出时也不会
        把消息泵压满；刷新变便宜后自动回落。用 process_time（纯 CPU）而非
        墙上时间，避免把等待/空闲算进去。
        """
        if cpu <= 0:
            return
        if self._flush_cpu_ema <= 0:
            self._flush_cpu_ema = cpu
        else:
            self._flush_cpu_ema = self._flush_cpu_ema * 0.7 + cpu * 0.3
        target = self._flush_cpu_ema * 4
        self._flush_interval = min(FLUSH_INTERVAL_MAX, max(FLUSH_INTERVAL_MIN, target))

    def _should_seal_segment(self) -> bool:
        """当前流式段是否该封段（限制单块增长，见 STREAM_SEG_MAX_CHARS 注释）。"""
        if self._stream_seg_chars < STREAM_SEG_MAX_CHARS:
            return False
        raw = self._stream_seg_raw
        if stream_segment_sealable(raw):
            return True
        # 兜底：超长单行/长代码块迟迟等不到安全边界时也封段；但未闭合围栏
        # 绝不在中间截断（宁可慢一点，也不能把代码块拆坏）。
        return (
            self._stream_seg_chars >= STREAM_SEG_FORCE_CHARS
            and stream_segment_fences_balanced(raw)
        )

    async def _seal_stream_segment(self) -> None:
        """封存当前流式 Markdown 段，后续内容写入新段。

        _chat_fix_segment 会复位段计数与 _reasoning_open：若封段发生在思考
        中途，下一批思考会重新带「思考过程」标题与 `> ` 前缀，自成一块。
        """
        await self._chat_fix_segment()

    def _track_stream_segment(self, text: str) -> None:
        """累计当前流式段的规模与原始文本（用于封段判断）。"""
        self._stream_seg_chars += len(text)
        self._stream_seg_raw += text

    async def _render_stream_chunk(self, content: str, reasoning: str) -> None:
        """渲染一批流式增量（正文 + 思考），逻辑与原逐 chunk 渲染等价但按批执行。

        多轮工具调用：一轮「思考→正文→工具调用」结束后会再来一轮
        「思考→正文」；每轮思考各自开启一个带「思考过程」标题的灰色块引用，
        正文穿插其间按普通文本显示。

        单段超过 STREAM_SEG_* 阈值时先封段再写，避免尾块无限增长导致 O(n²)
        重渲染卡顿（见文件顶部常量注释）。
        """
        chat = self._chat_container()
        if not self._stream_active:
            self._stream_active = True
            self._answer_started = True
            await self._chat_stream_append("\n\n---\n\n**mincli**\n\n")
        if reasoning:
            # 思考过程：灰色块引用；一轮思考期间跨批次直接拼接不重复标题，
            # 上一轮思考已被正文关闭（_reasoning_open=False）时开启新块
            if self._reasoning_open and self._should_seal_segment():
                await self._seal_stream_segment()
            if not self._reasoning_open:
                self._reasoning_open = True
                await self._chat_stream_append(
                    "\n\n" + REASONING_HEADER_MD + "\n>\n> "
                    + self._reasoning_chunk_md(reasoning),
                )
            else:
                await self._chat_stream_append(self._reasoning_chunk_md(reasoning))
            self._track_stream_segment(reasoning)
        if content:
            if self._should_seal_segment():
                await self._seal_stream_segment()
            self._reasoning_open = False  # 正文出现 → 当前思考块结束
            if not self._answer_started:
                self._answer_started = True
                await self._chat_stream_append("\n\n**mincli：**\n\n")
            await self._chat_stream_append(content)
            self._track_stream_segment(content)
        await self._chat_shrink_lists()

    async def _handle_event_inner(self, ev: ControllerEvent) -> None:
        if ev.kind == "node_created":
            node = ev.node
            if node is not None:
                if self._full_view:
                    self._set_full_view(False)  # 新消息进入新节点 → 退出全览，恢复流式视图
                # 直接进入新节点：树重建 + 光标跟随 + 消息区切换到该节点视图
                # （不输出 **mincli：** 头部——思考过程在提问之后、头部与正文之前）
                self._rebuild_tree()
                self._select_tree_node(node.id)
                view = f"# {node.id}: {node.title}\n\n" f"**你：**\n\n{node.user_msg}"
                if node.user_images:
                    marks = "\n".join(
                        f"- {image_placeholder_text(a)}" for a in node.user_images
                    )
                    view += f"\n\n{marks}"
                await self._chat_reset(view)
                await self._chat_shrink_lists(scroll=False)  # 新节点先显示开头，流式再滚到底
                self._refresh_import_status()  # 图片已绑定到节点，隐藏提示行
                self._stream_active = True  # 视图已含节点头部，后续流式内容直接追加
                self._reasoning_open = False
                self._answer_started = False
        elif ev.kind == "tool":
            # 工具调用渲染成结构化 ToolCard（真控件，穿插在正文 Markdown 流之间）：
            # 开始事件创建卡片并固化当前正文段，结束事件更新同一张卡片为「完成」。
            if ev.tool_summary:
                if self._active_tool_card is not None:
                    self._active_tool_card.set_result(ev.tool_summary)
            else:
                args_lines = self._format_tool_args(ev.tool_args).split("\n")
                card = ToolCard(ev.tool_name, args_lines)
                await self._chat_add_toolcard(card)
                self._active_tool_card = card
            await self._chat_shrink_lists()
        elif ev.kind == "status":
            # 状态也可能是多行模型文本（如审核思考）：逐行加引用前缀并转义行首
            # 自带标记，避免第二行起跑出引用块、或叠加成 `>>` 嵌套引用。
            await self._chat_stream_append("\n" + quote_block_text(ev.message) + "\n")
            await self._chat_shrink_lists()
        elif ev.kind == "error":
            await self._chat_stream_append(f"\n\n{quote_block_text(ev.message)}\n")
            # 出错不再回滚节点：本轮已保存（含已生成的部分内容）
            await self._chat_stream_append(">\n> 本轮已保存到当前节点。\n")
            await self._chat_shrink_lists()
            self._rebuild_tree()  # 节点保留，刷新树
            self._refresh_import_status()
            self._set_turn_active(False)
        elif ev.kind == "done":
            node = ev.node
            if node is not None:
                await self._chat_stream_append(
                    f"\n\n---\n\n*📊 输入 {node.input_tokens} tokens | 输出 {node.output_tokens} tokens*\n"
                )
            await self._chat_shrink_lists()
            self._rebuild_tree()
            if node is not None:
                self._select_tree_node(node.id)
            self._refresh_usage_bar()
            self._set_turn_active(False)

    def _handle_background_event(self, ev: ControllerEvent) -> None:
        """后台那棵树的事件：不渲染到消息区，只在侧栏/顶部留状态。

        生成在后台照常进行（节点内容与磁盘都会更新），切回去时按节点内容
        重新渲染，所以这里丢弃流式增量是安全的。
        """
        if ev.kind in ("done", "error"):
            self._refresh_tree_ui()
            label = "已完成" if ev.kind == "done" else "出错"
            self.notify(
                f"对话树 {ev.tree} {label}（点击侧栏可切过去查看）", timeout=6
            )

    def _append_error(self, message: str) -> None:
        self._cancel_flush()  # 出错后不再渲染残留流式缓冲
        asyncio.ensure_future(self._chat_append(
            f"\n\n{quote_block_text(message)}\n"
            ">\n> 本轮已保存到当前节点。\n"
        ))

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

        多对话树：确认来自后台生成的那棵树时（用户可能正看着别的树），标题里
        标明树编号，避免不知道这个弹窗是谁触发的。
        """
        title = self._scoped_confirm_title(title)
        return self.call_from_thread(self._confirm_async, title, text)

    def _scoped_confirm_title(self, title: str) -> str:
        if self.ctrl is None:
            return title
        number = self.ctrl.generating_number
        if number and number != self.ctrl.active_number:
            return f"对话树 {number}：{title}"
        return title

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
