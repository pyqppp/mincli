"""工具层测试：文件工具、联网工具、门控矩阵与分派。"""

import os

import pytest

from mincli.config import Config
from mincli.tools import builtin, file_tools, web_tools


# ---------------- file_tools ----------------


def test_read_file_numbers_lines_and_continues(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")

    text = file_tools.read_file(str(path), offset=1, limit=4)
    assert "共 10 行" in text
    assert "第 1-4 行" in text
    assert "1| line1" in text
    assert "4| line4" in text
    assert "line5" not in text
    assert "offset=5" in text

    tail = file_tools.read_file(str(path), offset=9, limit=100)
    assert "第 9-10 行" in text or "第 9-10 行" in tail
    assert "已到文件末尾" in tail


def test_read_file_errors(tmp_path):
    assert "文件不存在" in file_tools.read_file(str(tmp_path / "nope.txt"))
    assert "未提供文件路径" in file_tools.read_file("")
    assert "这是一个目录" in file_tools.read_file(str(tmp_path))


def test_read_file_empty_and_binary(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert "空文件" in file_tools.read_file(str(empty))

    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"\x00\x01\x02")
    assert "二进制" in file_tools.read_file(str(binary))

    out_of_range = tmp_path / "a.txt"
    out_of_range.write_text("x\n", encoding="utf-8")
    assert "超出范围" in file_tools.read_file(str(out_of_range), offset=9)


def test_write_file_and_confirm(tmp_path):
    target = tmp_path / "sub" / "out.txt"
    result = file_tools.write_file(str(target), "hello\nworld")
    assert "已写入" in result
    assert target.read_text(encoding="utf-8") == "hello\nworld"
    assert "2 行" in result

    denied = file_tools.write_file(str(target), "x", confirm=lambda _: False)
    assert "已取消" in denied
    assert target.read_text(encoding="utf-8") == "hello\nworld"


def test_edit_file_unique_match(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    ok = file_tools.edit_file(str(target), "beta", "BETA")
    assert "已修改" in ok
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"

    missing = file_tools.edit_file(str(target), "delta", "x")
    assert "未找到" in missing

    duplicated = file_tools.edit_file(str(target), "a", "A")
    assert "出现" in duplicated and "次" in duplicated

    assert "old_string 不能为空" in file_tools.edit_file(str(target), "", "x")


def test_list_directory_dirs_first(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")

    text = file_tools.list_directory(str(tmp_path))
    assert "2 项" in text
    assert text.index("b_dir/") < text.index("a.txt")
    assert "B" in text  # 文件大小带单位

    assert "目录不存在" in file_tools.list_directory(str(tmp_path / "nope"))
    assert "空目录" in file_tools.list_directory(str(tmp_path / "b_dir"))


# ---------------- 门控矩阵 ----------------


def _names(cfg, has_attachments=False):
    return builtin.available_tool_names(cfg, has_attachments)


def _offline(cfg):
    cfg.web_enabled = False
    return cfg


def test_gating_off():
    cfg = _offline(Config())
    assert _names(cfg) == []
    assert _names(cfg, True) == ["read_attachment"]


def test_gating_read():
    cfg = _offline(Config().with_files_mode("read"))
    assert _names(cfg) == ["read_file", "list_directory"]


def test_gating_confirm_and_direct():
    confirm = _offline(Config().with_files_mode("confirm"))
    assert _names(confirm) == ["read_file", "list_directory", "write_file", "edit_file"]
    direct = _offline(Config().with_files_mode("direct"))
    assert _names(direct) == _names(confirm)


def test_gating_web_and_recall_independent():
    cfg = Config().with_files_mode("off")
    cfg.web_enabled = False
    assert _names(cfg) == []

    cfg.web_enabled = True
    assert _names(cfg) == ["web_search", "web_fetch"]

    cfg.recall_enabled = True
    assert _names(cfg) == [
        "web_search",
        "web_fetch",
        "query_conversation_tree",
        "read_conversation_nodes",
    ]


def test_available_tools_schema():
    cfg = Config().with_files_mode("read")
    tools = builtin.available_tools(cfg, has_attachments=True)
    assert [t["function"]["name"] for t in tools][0] == "read_attachment"
    read = next(t for t in tools if t["function"]["name"] == "read_file")
    assert read["type"] == "function"
    assert read["function"]["parameters"]["required"] == ["path"]


def test_dispatch_unknown_tool():
    ctx = builtin.ToolContext(config=Config())
    assert "未知工具" in builtin.dispatch_tool("nope", {}, ctx)


def test_dispatch_read_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hi", encoding="utf-8")
    ctx = builtin.ToolContext(config=Config())
    assert "hi" in builtin.dispatch_tool("read_file", {"path": str(target)}, ctx)


# ---------------- read_attachment ----------------


class _FakeAttachment:
    def __init__(self, name, text="", kind="document", label=None):
        self.name = name
        self.text = text
        self.kind = kind
        self.path = ""
        self.stored_path = ""
        self.chip_label = label or f"[FILE {name}]"


def test_read_attachment_continues():
    att = _FakeAttachment("a.txt", "\n".join(f"row{i}" for i in range(1, 6)))
    ctx = builtin.ToolContext(config=Config(), attachments=[att])
    text = builtin.dispatch_tool(
        "read_attachment", {"name": "a.txt", "offset": 4, "limit": 1}, ctx
    )
    assert "第 4-4 行" in text
    assert "row4" in text
    assert "offset=5" in text


def test_read_attachment_lookup_and_errors():
    att = _FakeAttachment("报告.docx", "内容")
    ctx = builtin.ToolContext(config=Config(), attachments=[att])

    assert "未找到附件" in builtin.dispatch_tool(
        "read_attachment", {"name": "x.docx"}, ctx
    )
    assert "请提供附件名称" in builtin.dispatch_tool("read_attachment", {}, ctx)

    image = _FakeAttachment("p.jpg", "", kind="image")
    ctx2 = builtin.ToolContext(config=Config(), attachments=[image])
    assert "图片" in builtin.dispatch_tool("read_attachment", {"name": "p.jpg"}, ctx2)

    empty = builtin.ToolContext(config=Config())
    assert "没有可读取的附件" in builtin.dispatch_tool(
        "read_attachment", {"name": "a"}, empty
    )


def test_read_attachment_ambiguous_names():
    one = _FakeAttachment("a.txt", "one")
    two = _FakeAttachment("a.txt", "two")
    ctx = builtin.ToolContext(config=Config(), attachments=[one, two])
    assert "多个同名附件" in builtin.dispatch_tool(
        "read_attachment", {"name": "a.txt"}, ctx
    )


# ---------------- web 工具 ----------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("bad json")
        return self._payload


def test_web_search_formats_results(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        return FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "标题一",
                        "url": "https://example.com/a",
                        "summary": "摘要一",
                        "site_name": "Example",
                        "date": "2026-06-01T00:00:00+00:00",
                    }
                ]
            },
        )

    monkeypatch.setattr(web_tools.httpx, "post", fake_post)
    text = web_tools.web_search("测试", count=3)
    assert "标题一" in text and "https://example.com/a" in text and "摘要一" in text
    assert "Example | 2026-06-01" in text
    assert calls["url"].endswith("/v1/tools/search")
    assert calls["json"]["count"] == 3


def test_web_search_empty_and_errors(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        web_tools.httpx, "post", lambda *a, **k: FakeResponse(200, {"results": []})
    )
    assert "没有返回结果" in web_tools.web_search("空")

    monkeypatch.setattr(
        web_tools.httpx,
        "post",
        lambda *a, **k: FakeResponse(
            401, {"error": {"message": "Authentication Fails"}}
        ),
    )
    text = web_tools.web_search("x")
    assert "HTTP 401" in text and "API Key" in text

    monkeypatch.setattr(web_tools.httpx, "post", lambda *a, **k: FakeResponse(200, None))
    assert "合法" in web_tools.web_search("x")


def test_web_search_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert "DEEPSEEK_API_KEY" in web_tools.web_search("x")
    assert "DEEPSEEK_API_KEY" in web_tools.web_fetch("https://example.com")


def test_web_fetch_truncates(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("MINCLI_WEBPAGE_MAX_LENGTH", "100")
    long_content = "字" * 500
    monkeypatch.setattr(
        web_tools.httpx,
        "post",
        lambda *a, **k: FakeResponse(200, {"title": "标题", "content": long_content}),
    )
    text = web_tools.web_fetch("example.com")
    assert "标题" in text
    assert "链接：https://example.com" in text
    assert "已截断" in text
    assert len(text) < 400


def test_web_fetch_errors(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(web_tools.httpx, "post", lambda *a, **k: FakeResponse(404, {}))
    assert "HTTP 404" in web_tools.web_fetch("https://example.com/x")

    monkeypatch.setattr(
        web_tools.httpx, "post", lambda *a, **k: FakeResponse(200, {"title": "t"})
    )
    assert "没有可用的正文" in web_tools.web_fetch("https://example.com/x")


def test_web_search_validates_query(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert "不能为空" in web_tools.web_search("   ")
    assert "128" in web_tools.web_search("x" * 200)


def test_web_timeout(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def boom(*a, **k):
        raise web_tools.httpx.TimeoutException("timeout")

    monkeypatch.setattr(web_tools.httpx, "post", boom)
    assert "超时" in web_tools.web_search("x")


def test_read_attachment_accepts_dir_prefix():
    att = _FakeAttachment("报告.docx", "正文一行", label="[DOCX 报告.docx]")
    att.path = "/tmp/papers/报告.docx"
    ctx = builtin.ToolContext(config=Config(), attachments=[att])
    text = builtin.dispatch_tool(
        "read_attachment", {"name": "papers/报告.docx"}, ctx
    )
    assert "正文一行" in text
