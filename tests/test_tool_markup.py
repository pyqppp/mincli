"""工具调用标记清理：流式过滤与整段清理。"""

from mincli.tool_markup import MARKUP_FALLBACK, ToolMarkupFilter, strip_tool_markup

DSML = (
    '<|DSML| | calls> <|DSML| | invoke name="web_search"> '
    '<|DSML| | parameter name="query" string="true">新闻 用法</|DSML| | parameter> '
    "</|DSML| | invoke> </|DSML| | calls>"
)


def test_plain_text_passes_through():
    f = ToolMarkupFilter()
    assert f.feed("你好，") == "你好，"
    assert f.feed("世界") == "世界"
    assert f.flush() == ""
    assert f.leaked is False


def test_markup_is_dropped():
    f = ToolMarkupFilter()
    out = f.feed(DSML + "\n")
    assert "DSML" not in out
    assert f.leaked is True
    assert f.names == ["web_search"]
    assert "DSML" not in f.flush()


def test_markup_spanning_chunks():
    f = ToolMarkupFilter()
    pieces = [DSML[:4], DSML[4:18], DSML[18:60], DSML[60:]]
    out = "".join(f.feed(piece) for piece in pieces) + f.flush()
    assert "DSML" not in out
    assert "web_search" not in out
    assert f.leaked is True


def test_text_around_markup_kept():
    f = ToolMarkupFilter()
    out = f.feed("先说结论。\n" + DSML + "\n后面继续。") + f.flush()
    assert "先说结论。" in out
    assert "后面继续。" in out
    assert "DSML" not in out


def test_partial_marker_is_held_then_emitted():
    f = ToolMarkupFilter()
    out = f.feed("a <") + f.feed("b") + f.flush()
    assert out == "a <b"


def test_strip_tool_markup():
    clean, leaked, names = strip_tool_markup("答案在这里。\n" + DSML)
    assert leaked is True
    assert names == ["web_search"]
    assert "DSML" not in clean
    assert "答案在这里。" in clean


def test_strip_plain_text_unchanged():
    clean, leaked, names = strip_tool_markup("普通回答\n第二行")
    assert clean == "普通回答\n第二行"
    assert leaked is False and names == []
