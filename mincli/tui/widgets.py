"""mincli TUI 自定义控件。"""

from __future__ import annotations

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

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
    """

    BINDINGS = [
        Binding("enter", "submit_message", "发送", show=False, priority=True),
        Binding("ctrl+j", "insert_newline", "换行", show=False),
        Binding("alt+enter", "insert_newline", "换行", show=False),
        Binding("tab", "complete_or_tab", "命令补全/Tab", show=False, priority=True),
        Binding("up", "scroll_answer(-1)", "上滚回答区", show=False),
        Binding("down", "scroll_answer(1)", "下滚回答区", show=False),
    ]

    class Submitted(Message):
        """用户按下 Enter 提交消息。"""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class TextChanged(Message):
        """输入内容发生变化（用于命令补全弹窗）。"""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._target_height = MIN_INPUT_HEIGHT

    def action_submit_message(self) -> None:
        """提交当前输入并清空输入框。"""
        text = self.text.strip()
        if not text:
            return
        self.post_message(self.Submitted(text))
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

    def action_scroll_answer(self, delta: int) -> None:
        """↑/↓：输入框为空时滚动回答区；有内容时保留默认光标移动。"""
        if self.text.strip():
            if delta < 0:
                self.action_cursor_up()
            else:
                self.action_cursor_down()
            return
        self.app._scroll_chat(delta)  # type: ignore[attr-defined]

    def on_text_area_changed(self, event) -> None:
        """内容变化时按行数自适应高度（3~8 行）并广播文本变化。"""
        self._update_height()
        self.post_message(self.TextChanged(self.text))

    def _update_height(self) -> None:
        lines = self.document.line_count
        target = min(max(lines + 1, MIN_INPUT_HEIGHT), MAX_INPUT_HEIGHT)
        if target != self._target_height:
            self._target_height = target
            self.styles.height = target
