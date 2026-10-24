"""附件模型：引用块标签、截断与消息组装。"""

from mincli.attachments import (
    Attachment,
    KIND_DOCUMENT,
    KIND_IMAGE,
    KIND_WEB,
    compose_message,
    truncate_at_line,
)


def _doc(name="报告.docx", text="第一行\n第二行\n", path="/tmp/a.docx"):
    return Attachment(kind=KIND_DOCUMENT, name=name, path=path, text=text)


def test_chip_labels():
    assert _doc().chip_label == "[DOCX 报告.docx]"
    assert _doc(name="paper.pdf").chip_label == "[PDF paper.pdf]"
    assert _doc(name="note.md").chip_label == "[MD note.md]"
    image = Attachment(kind=KIND_IMAGE, name="photo.jpg", path="/tmp/p.jpg")
    assert image.chip_label == "[IMAGE photo.jpg]"
    web = Attachment(kind=KIND_WEB, name="example.com/标题", path="https://example.com")
    assert web.chip_label == "[WEB example.com/标题]"


def test_compose_short_document():
    text, images = compose_message("读一下[DOCX 报告.docx]", [_doc()])
    assert images == []
    assert text.startswith("读一下[DOCX 报告.docx]\n\n---\n\n")
    assert "报告.docx（共 2 行）：" in text
    assert "第一行" in text
    assert "已截断" not in text


def test_compose_without_attachments_keeps_text():
    assert compose_message("你好", []) == ("你好", [])


def test_compose_long_document_truncates_on_line_boundary():
    body = "\n".join(f"第{i:04d}行内容" for i in range(1, 2001))
    att = _doc(text=body)
    text, _ = compose_message("看看", [att])
    assert "（共 2000 行，" in text
    assert "此处显示前" in text
    assert "已截断；可用 read_attachment 从第" in text
    # 截断点落在完整行上：正文里不应出现半行
    lines = text.split("\n")
    assert all(line.startswith("第") or not line.startswith("第") for line in lines)
    head = text.split("---\n\n", 1)[1]
    shown = [ln for ln in head.splitlines() if ln.startswith("第")]
    assert shown[-1].endswith("行内容")


def test_compose_dedupes_same_file():
    att = _doc()
    text, _ = compose_message("看[DOCX 报告.docx]和[DOCX 报告.docx]", [att, att])
    assert text.count("报告.docx（共 2 行）") == 1
    assert text.count("[DOCX 报告.docx]") == 2


def test_compose_keeps_images_out_of_text():
    image = Attachment(kind=KIND_IMAGE, name="photo.jpg", path="/tmp/p.jpg")
    text, images = compose_message("看图[IMAGE photo.jpg]", [image])
    assert text == "看图[IMAGE photo.jpg]"
    assert len(images) == 0  # 没有 ImageAttachment 对象时不返回


def test_truncate_at_line():
    text = "aaa\nbbb\nccc"
    cut, lines = truncate_at_line(text, 5)
    assert cut == "aaa"
    assert lines == 1
    cut, lines = truncate_at_line("abcdef", 4)
    assert cut == "abcd"
    assert lines == 1
