"""输入框附件引用块：插入、整块删除、光标跳过与样式。"""

from textual.app import App, ComposeResult

from mincli.attachments import Attachment, KIND_DOCUMENT
from mincli.tui.widgets import ChatInput


class _Harness(App):
    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield ChatInput(id="input")

    def _advance_or_complete(self, reverse: bool = False) -> bool:
        return False

    def _scroll_chat(self, delta: int) -> None:
        pass


def _attachment(name: str = "报告.docx") -> Attachment:
    return Attachment(kind=KIND_DOCUMENT, name=name, path=f"/tmp/{name}", text="正文")


async def test_insert_chip_and_take_attachments():
    app = _Harness()
    async with app.run_test():
        widget = app.query_one("#input", ChatInput)
        attachment = _attachment()
        widget.insert_chip(attachment)
        await app.workers.wait_for_complete()
        assert widget.text == "[DOCX 报告.docx] "
        assert widget.take_chip_attachments() == [attachment]
        assert widget.take_chip_attachments() == []


async def test_backspace_deletes_whole_chip():
    app = _Harness()
    async with app.run_test():
        widget = app.query_one("#input", ChatInput)
        widget.insert_chip(_attachment())
        # 光标在尾随空格后：先删空格，再一次删整块（不会退化成逐字删除）
        widget.action_delete_left()
        assert widget.text == "[DOCX 报告.docx]"
        widget.action_delete_left()
        assert widget.text == ""


async def test_delete_key_and_cursor_skip():
    app = _Harness()
    async with app.run_test():
        widget = app.query_one("#input", ChatInput)
        widget.insert_chip(_attachment())
        start, stop = widget.chip_spans()[0]
        widget.move_cursor(widget._location_of(start))
        # 光标在块首，Delete 一次删掉整块
        widget.action_delete_right()
        assert widget.text == " "

        widget.insert_chip(_attachment())
        start, stop = widget.chip_spans()[0]
        # 光标跳到块尾：左键直接回块首，不落入块内；右键再跳到块尾
        widget.move_cursor(widget._location_of(stop))
        widget.action_cursor_left()
        assert widget.cursor_location == widget._location_of(start)
        widget.action_cursor_right()
        assert widget.cursor_location == widget._location_of(stop)


async def test_chip_line_is_styled():
    app = _Harness()
    async with app.run_test():
        widget = app.query_one("#input", ChatInput)
        widget.insert_chip(_attachment())
        strip = widget.render_line(0)
        styles = [segment.style for segment in strip._segments if segment.text == "[DOCX 报告.docx]"]
        assert styles, "引用块应当单独成段"
        assert styles[0].color.name == "bright_cyan"
        assert styles[0].bgcolor.name == "grey27"
