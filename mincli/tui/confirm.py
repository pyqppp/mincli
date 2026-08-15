"""确认对话框（ModalScreen）—— 供工具确认（写文件 / 执行命令等）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from asyncio import Future


class ConfirmScreen(ModalScreen[bool]):
    """是/否确认弹窗。dismiss(True/False)。"""

    # 默认聚焦“否/取消”：删除等破坏性操作不应默认落在确认上
    AUTO_FOCUS = "#confirm-no"

    BINDINGS = [
        Binding("left", "focus_button(-1)", "上一个按钮", show=False),
        Binding("right", "focus_button(1)", "下一个按钮", show=False),
    ]

    def __init__(
        self,
        title: str,
        text: str,
        result_future: Future | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._text = text
        self._result_future = result_future

    def action_focus_button(self, delta: int) -> None:
        """左右键在「是 / 否」按钮之间循环移动焦点。"""
        buttons = list(self.query(Button))
        if not buttons:
            return
        current = self.focused
        try:
            index = buttons.index(current)
        except ValueError:
            index = 0
        target = buttons[(index + delta) % len(buttons)]
        self.set_focus(target)

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self._title, id="confirm-title")
            yield Static(self._text, id="confirm-text")
            with Horizontal(id="confirm-buttons"):
                yield Button("是", variant="primary", id="confirm-yes")
                yield Button("否", id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        result = event.button.id == "confirm-yes"
        # 直接在当前弹窗自己的消息泵上解决 Future。不能依赖 push_screen 的
        # callback：它经 call_next 投递到 App 消息泵，而 App 泵此刻正卡在
        # await 这个 Future 的消息处理器里（处理器不返回，next callbacks 不
        # 会冲刷），会造成自死锁——弹窗能显示但整个应用卡死。
        if self._result_future is not None and not self._result_future.done():
            self._result_future.set_result(result)
        self.dismiss(result)
