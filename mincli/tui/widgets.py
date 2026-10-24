"""mincli TUI 自定义控件。"""

from __future__ import annotations

import os

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Static, TextArea

from mincli.attachments import Attachment
from mincli.helpers import split_path_args

# 锁定键（Caps Lock / Num Lock / Scroll Lock）绝不应产生输入。
# 注意：真正的拦截在 App 级优先级绑定（见 app.py 的 action_ignore_lock）；
# 本常量仅作参考/共享定义。启用 kitty 协议时部分终端会把锁定键上报为
# 按键事件；传统模式下 iTerm2 会把 Caps Lock 编码成裸大写字母、与真实
# 按键无法区分——真正的修复是禁用 kitty 协议（见 mincli/tui/__init__.py）。
LOCK_KEYS = frozenset({"caps_lock", "num_lock", "scroll_lock"})

MIN_INPUT_HEIGHT = 3
MAX_INPUT_HEIGHT = 8


class ChatInput(TextArea):
    """多行消息输入框。

    按键：
        Enter      提交消息（post `Submitted` 事件）
        Ctrl+J     插入换行（iTerm2 传统模式下可靠）
        Alt+Enter  插入换行（仅限能区分该按键的终端；iTerm2 传统模式下
                    Alt+Enter 会被当作 Enter 处理，见键盘协议说明）
        Ctrl+C     无选区时退出应用；输入框内有选区时先复制（TextArea 默认）
                   生成/命令执行进行中则先打断当前轮
        Esc        打断正在进行的生成/命令（空闲时无操作）
        Tab        命令补全：循环高亮候选（唯一候选直接补全）；命令写完则插入制表符
        Shift+Tab  命令补全：反向循环高亮候选
    """

    BINDINGS = [
        Binding("enter", "submit_message", "发送", show=False, priority=True),
        Binding("ctrl+j", "insert_newline", "换行", show=False),
        Binding("alt+enter", "insert_newline", "换行", show=False),
        Binding("tab", "complete_or_tab", "命令补全/Tab", show=False, priority=True),
        Binding("shift+tab", "complete_or_tab_back", "命令补全反向", show=False, priority=True),
        Binding("up", "scroll_answer(-1)", "上滚回答区", show=False),
        Binding("down", "scroll_answer(1)", "下滚回答区", show=False),
        # TextArea 默认会吞掉 Esc（切焦点），这里优先接管为「打断生成」
        Binding("escape", "interrupt_turn", "打断", show=False, priority=True),
    ]

    class Submitted(Message):
        """用户按下 Enter 提交消息。

        ``attachments`` 在提交瞬间就从正文里解析出来：提交后输入框会立刻
        清空，若等 App 收到消息再去解析，引用块已经不存在了。
        """

        def __init__(self, text: str, attachments: list[Attachment] | None = None) -> None:
            super().__init__()
            self.text = text
            self.attachments = list(attachments or [])

    class TextChanged(Message):
        """输入内容发生变化（用于命令补全弹窗）。"""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class FilesDropped(Message):
        """终端拖入文件 → 粘贴内容全为路径，请求直接导入。"""

        def __init__(self, paths: list[str]) -> None:
            super().__init__()
            self.paths = paths

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._target_height = MIN_INPUT_HEIGHT
        # 附件引用块：标签文本（含方括号）→ 附件。标签在正文里出现几次就有几个块。
        self._chips: dict[str, Attachment] = {}
        # 每行起点的字符偏移缓存（块样式与原子删除都要按偏移换算行列）
        self._line_starts_key: str | None = None
        self._line_starts: list[int] = [0]

    # ---------------- 附件引用块 ----------------

    CHIP_STYLE = Style(bgcolor="grey27", color="bright_cyan")

    def _ensure_line_starts(self) -> None:
        text = self.text
        if self._line_starts_key == text:
            return
        starts = [0]
        for index, char in enumerate(text):
            if char == "\n":
                starts.append(index + 1)
        self._line_starts_key = text
        self._line_starts = starts

    def _offset_of(self, location) -> int:
        self._ensure_line_starts()
        row, column = location
        if row < len(self._line_starts):
            return self._line_starts[row] + column
        return len(self.text)

    def _location_of(self, offset: int) -> tuple[int, int]:
        self._ensure_line_starts()
        starts = self._line_starts
        low, high = 0, len(starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if starts[middle] <= offset:
                low = middle
            else:
                high = middle - 1
        return low, offset - starts[low]

    def chip_spans(self) -> list[tuple[int, int]]:
        """正文里所有引用块的字符区间（按出现位置排序）。"""
        text = self.text
        spans: list[tuple[int, int]] = []
        for label in self._chips:
            start = text.find(label)
            while start != -1:
                spans.append((start, start + len(label)))
                start = text.find(label, start + len(label))
        spans.sort()
        return spans

    def _chip_cell_ranges(self, y: int) -> list[tuple[int, int]]:
        """某显示行上引用块占用的单元格区间（已考虑软换行与横向滚动）。"""
        try:
            scroll_x, scroll_y = self.scroll_offset
            line_index, section_offset = self.wrapped_document._offset_to_line_info[
                scroll_y + y
            ]
        except (AttributeError, IndexError, TypeError):
            return []
        self._ensure_line_starts()
        if line_index >= len(self._line_starts):
            return []
        line_start = self._line_starts[line_index]
        line = self.document.get_line(line_index)
        width = max(0, self.size.width)
        section_cell = cell_len(line[:section_offset])
        ranges: list[tuple[int, int]] = []
        for start, stop in self.chip_spans():
            if stop <= line_start or start >= line_start + len(line):
                continue
            left = max(0, start - line_start)
            right = min(len(line), stop - line_start)
            start_cell = cell_len(line[:left]) - section_cell - scroll_x
            end_cell = cell_len(line[:right]) - section_cell - scroll_x
            if end_cell <= 0 or start_cell >= width:
                continue
            ranges.append((max(0, start_cell), min(width, end_cell)))
        return ranges

    def render_line(self, y: int) -> Strip:
        """在 TextArea 自己的行渲染之后给引用块上色。

        TextArea 的 Theme 会用 base_style 覆盖整行，因此在 ``get_line`` 里
        着色会被冲掉；只能等 Strip 生成后，按单元格区间重新贴样式。
        """
        strip = super().render_line(y)
        if not getattr(self, "_chips", None):
            return strip
        width = strip.cell_length
        for start, stop in reversed(self._chip_cell_ranges(y)):
            start, stop = max(0, min(width, start)), max(0, min(width, stop))
            if stop <= start:
                continue
            pieces: list[Strip] = []
            if start > 0:
                pieces.append(strip.crop(0, start))
            middle = strip.crop(start, stop)
            pieces.append(
                Strip(
                    list(
                        Segment.apply_style(
                            middle._segments, post_style=self.CHIP_STYLE
                        )
                    ),
                    middle.cell_length,
                )
            )
            if stop < width:
                pieces.append(strip.crop(stop, width))
            strip = Strip.join(pieces)
        return strip

    def insert_chip(self, attachment: Attachment) -> None:
        """在光标处插入一个引用块（前后补空格，避免和文字粘连）。"""
        label = attachment.chip_label
        self._chips[label] = attachment
        position = self.cursor_location
        offset = self._offset_of(position)
        lead = "" if offset == 0 or self.text[offset - 1].isspace() else " "
        self.replace(f"{lead}{label} ", position, position)
        self._line_starts_key = None
        self._line_cache.clear()
        self.refresh()

    def take_chip_attachments(self) -> list[Attachment]:
        """取走正文里实际出现的附件（按出现顺序），并清空登记表。"""
        attachments: list[Attachment] = []
        text = self.text
        for start, stop in self.chip_spans():
            attachment = self._chips.get(text[start:stop])
            if attachment is not None:
                attachments.append(attachment)
        self._chips.clear()
        return attachments

    def _selection_empty(self) -> bool:
        start, end = self.selection
        return start == end

    def _chip_before_cursor(self) -> tuple[int, int] | None:
        offset = self._offset_of(self.cursor_location)
        for start, stop in self.chip_spans():
            if start < offset <= stop:
                return start, stop
        return None

    def _chip_after_cursor(self) -> tuple[int, int] | None:
        offset = self._offset_of(self.cursor_location)
        for start, stop in self.chip_spans():
            if start <= offset < stop:
                return start, stop
        return None

    def _delete_span(self, span: tuple[int, int]) -> None:
        start, stop = span
        self.replace("", self._location_of(start), self._location_of(stop))
        self.move_cursor(self._location_of(start))
        self._line_starts_key = None
        self._line_cache.clear()
        self.refresh()

    def action_delete_left(self) -> None:
        """退格：光标紧贴块尾或块内时整块删除。"""
        if self._selection_empty():
            span = self._chip_before_cursor()
            if span is not None:
                self._delete_span(span)
                return
        super().action_delete_left()

    def action_delete_right(self) -> None:
        """Delete：光标紧贴块首或块内时整块删除。"""
        if self._selection_empty():
            span = self._chip_after_cursor()
            if span is not None:
                self._delete_span(span)
                return
        super().action_delete_right()

    def action_cursor_left(self, select: bool = False) -> None:
        """左移：不把光标放进块内，直接跳到块首。"""
        if not select and self._selection_empty():
            offset = self._offset_of(self.cursor_location)
            for start, stop in self.chip_spans():
                if start < offset <= stop:
                    self.move_cursor(self._location_of(start))
                    return
        super().action_cursor_left(select)

    def action_cursor_right(self, select: bool = False) -> None:
        """右移：不把光标放进块内，直接跳到块尾。"""
        if not select and self._selection_empty():
            offset = self._offset_of(self.cursor_location)
            for start, stop in self.chip_spans():
                if start <= offset < stop:
                    self.move_cursor(self._location_of(stop))
                    return
        super().action_cursor_right(select)

    @staticmethod
    def _paths_from_paste(text: str) -> list[str] | None:
        """整段粘贴若能解析为文件路径/URL 列表则返回之，否则返回 None。

        终端把拖入的文件粘贴成路径（macOS/Linux 多为带引号或转义空格的
        POSIX 写法，Windows 为 ``C:\\...`` 反斜杠写法，带空格时加引号）。
        仅当所有 token 都 expanduser 后是存在的文件、或是 http(s) URL 时，
        才判定为「拖入导入」，避免劫持普通文本粘贴（如整句复制）。
        解析走 helpers.split_path_args（跨平台，兼容 Windows 反斜杠路径）。
        """
        text = (text or "").strip()
        if not text:
            return None
        tokens = split_path_args(text)
        if not tokens:
            return None
        paths: list[str] = []
        for tok in tokens:
            if tok.lower().startswith(("http://", "https://")):
                paths.append(tok)
                continue
            expanded = os.path.expanduser(tok)
            if not os.path.isfile(expanded):
                return None
            paths.append(expanded)
        return paths

    async def _on_paste(self, event: events.Paste) -> None:
        """拖入文件（路径粘贴）→ 转 FilesDropped 导入；普通粘贴照常插入。

        注意：Textual 的分发器会沿 MRO 调用「所有」_on_paste（含基类
        TextArea 的），因此路径粘贴需 prevent_default() 阻止基类把路径插入
        输入框 + stop() 阻止冒泡到 App；普通粘贴也需 prevent_default() 让
        分发器跳过基类，只由这里 super() 完成一次插入。
        """
        paths = self._paths_from_paste(event.text)
        if paths is not None:
            event.prevent_default()
            event.stop()
            self.post_message(self.FilesDropped(paths))
            return
        event.prevent_default()
        await super()._on_paste(event)

    def action_submit_message(self) -> None:
        """提交当前输入并清空输入框（引用块在清空之前先取走）。"""
        text = self.text.strip()
        if not text:
            return
        attachments = self.take_chip_attachments()
        self.post_message(self.Submitted(text, attachments))
        self.clear()
        self.focus()

    def action_insert_newline(self) -> None:
        """在光标处插入换行。"""
        self.insert("\n")

    def action_complete_or_tab(self) -> None:
        """Tab：命令补全模式下切换/补全；否则插入制表符。"""
        if self.app._advance_or_complete():  # type: ignore[attr-defined]
            return
        self.insert("\t")

    def action_complete_or_tab_back(self) -> None:
        """Shift+Tab：命令补全候选中反向切换（同一层级）。"""
        self.app._advance_or_complete(reverse=True)  # type: ignore[attr-defined]

    def action_scroll_answer(self, delta: int) -> None:
        """↑/↓：输入框为空时滚动回答区；有内容时保留默认光标移动。"""
        if self.text.strip():
            if delta < 0:
                self.action_cursor_up()
            else:
                self.action_cursor_down()
            return
        self.app._scroll_chat(delta)  # type: ignore[attr-defined]

    def action_interrupt_turn(self) -> None:
        """Esc：打断正在进行的生成/命令（空闲时无操作）。"""
        self.app.action_interrupt()  # type: ignore[attr-defined]

    def on_text_area_changed(self, event) -> None:
        """内容变化时按视觉行数自适应高度（3~8 行）并广播文本变化。"""
        self._update_height()
        self.post_message(self.TextChanged(self.text))

    def _chrome_rows(self) -> int:
        """输入框上下「非内容行」数（边框 + 内边距）。

        chat.tcss 里 ChatInput 是 `border: tall` + `padding: 0 1`，上下各占 1 行。
        高度必须把这 2 行算进去，否则内容会把最后一行挤掉（第一个 Ctrl+J 时
        lines+1 恰好等于 MIN_INPUT_HEIGHT，看起来就是「没长高、少一行」）。
        这里从实际样式取，改 CSS 后不需要同步改常量。
        """
        gutter = self.styles.gutter
        return int(gutter.top) + int(gutter.bottom)

    def _update_height(self) -> None:
        # 用视觉行数：TextArea 默认 soft_wrap=True，一个逻辑行折行后可能占多行
        lines = self.wrapped_document.height
        target = min(max(lines + self._chrome_rows(), MIN_INPUT_HEIGHT), MAX_INPUT_HEIGHT)
        if target != self._target_height:
            self._target_height = target
            self.styles.height = target


class TreeRow(Widget):
    """侧栏对话树列表的一行：行首色块 + 「编号 对话」+ 后台状态 + 当前树高亮。

    色块用 1 列宽的控件底色实现（不是用方块字符拼出来的），因此不引入任何
    图形符号；当前树的整行高亮沿用主题色（主题随当前树切换）。

    注意不要继承 Horizontal：chat.tcss 里有全局 `Horizontal { height: 1fr }`，
    会让每行在受限高度里被压扁（树多于 3 棵时行会叠在一起），这里用
    `layout: horizontal` 自己排布。
    """

    DEFAULT_CSS = """
    TreeRow {
        layout: horizontal;
        height: 1;
        width: 1fr;
    }

    TreeRow .tree-swatch {
        width: 1;
        height: 1;
    }

    TreeRow .tree-name {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    TreeRow:hover {
        background: $primary 12%;
    }

    TreeRow.active {
        background: $primary 30%;
    }

    TreeRow.active .tree-name {
        color: $text;
        text-style: bold;
    }
    """

    class Selected(Message):
        """点击某一行对话树（请求切换）。"""

        def __init__(self, number: int) -> None:
            super().__init__()
            self.number = number

    def __init__(
        self,
        number: int,
        color: int,
        label: str,
        active: bool = False,
        mark: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.number = number
        self.color = color
        self._label = label
        self._mark = mark
        self._active = active
        self._swatch: Static | None = None
        self._name: Static | None = None

    def compose(self):
        self._swatch = Static("", classes="tree-swatch")
        self._name = Static("", classes="tree-name", markup=False)
        yield self._swatch
        yield self._name

    def on_mount(self) -> None:
        from mincli.tui.theme import TREE_PALETTE

        index = max(0, min(len(TREE_PALETTE) - 1, int(self.color) - 1))
        if self._swatch is not None:
            self._swatch.styles.background = TREE_PALETTE[index]["primary"]
        if self._name is not None:
            text = self._label
            if self._mark == "done":
                text += "  完成"
            elif self._mark == "error":
                text += "  出错"
            self._name.update(text)
        self.set_class(self._active, "active")

    def on_click(self, event) -> None:
        event.stop()
        self.post_message(self.Selected(self.number))


class ToolCard(Static):
    """工具调用卡片：结构化显示工具名、参数与状态（不嵌入正文 Markdown 流）。

    用平面符号（非 emoji）：标题前缀 `▸`，参数逐键一行，状态行纯文字。
    开始事件 set_start() 显示「执行中」，结束事件 set_result() 更新为「完成」并展示结果。
    """

    def __init__(self, tool_name: str = "", args_lines: list[str] | None = None, **kwargs) -> None:
        # markup=False：工具名/参数/结果来自模型，形如 query="a b" 的片段
        # 会被 Textual 当样式标签解析并抛 MarkupError
        kwargs.setdefault("markup", False)
        super().__init__("", **kwargs)
        self.tool_name = tool_name
        self.args_lines = args_lines or []
        self._result = ""
        self._done = False

    def _render_text(self) -> str:
        lines = [f"▸ 工具调用：{self.tool_name}"]
        if self.args_lines:
            lines.append("  参数")
            for line in self.args_lines:
                lines.append(f"    {line}")
        if self._done:
            status = "完成"
        else:
            status = "执行中"
        lines.append(f"  状态：{status}")
        if self._result:
            lines.append(f"  结果：{self._result}")
        return "\n".join(lines)

    def set_start(self, tool_name: str, args_lines: list[str]) -> None:
        """工具开始：显示工具名+参数，状态执行中。"""
        self.tool_name = tool_name
        self.args_lines = args_lines
        self._done = False
        self._result = ""
        self.update(self._render_text())

    def set_result(self, summary: str) -> None:
        """工具结束：更新状态为完成，展示结果摘要。"""
        self._done = True
        self._result = summary
        self.update(self._render_text())

    def card_summary(self) -> str:
        """聚合 source 用：卡片文本。"""
        return self._render_text()

