"""fetch_webpage 单元测试（完全离线：monkeypatch 下载与抽取，不联网）。

运行：`venv/bin/python -m tests.test_web_fetch`
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli import config
from mincli.tools import web_fetch

PASS = 0
FAIL = 0

ENV_KEY = "MINCLI_WEBPAGE_MAX_LENGTH"


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ---------------- 测试替身 ----------------

class FakeResponse:
    """trafilatura.downloads.Response 的最小替身。"""

    def __init__(self, data=b"<html><body>x</body></html>", status=200, headers=None):
        self.data = data
        self.status = status
        self.url = "https://example.com/"
        self.headers = {"content-type": "text/html"} if headers is None else headers
        self.html = data.decode("utf-8", "replace")


@contextmanager
def env(value):
    """临时设置/清除 MINCLI_WEBPAGE_MAX_LENGTH（value=None 表示不设置）。"""
    old = os.environ.get(ENV_KEY)
    try:
        if value is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = value
        yield
    finally:
        if old is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = old


@contextmanager
def patched(fetch_result=None, extract_result=None, fetch_raises=None, extract_raises=None):
    """替换 web_fetch 的下载与抽取；记录调用参数。"""
    calls: dict = {"fetch": [], "extract": []}
    orig_fetch = web_fetch.fetch_response
    orig_extract = web_fetch.trafilatura.extract

    def fake_fetch(url, **kwargs):
        calls["fetch"].append((url, kwargs))
        if fetch_raises is not None:
            raise fetch_raises
        return fetch_result

    def fake_extract(html_text, **kwargs):
        calls["extract"].append((html_text, kwargs))
        if extract_raises is not None:
            raise extract_raises
        return extract_result

    web_fetch.fetch_response = fake_fetch
    web_fetch.trafilatura.extract = fake_extract
    try:
        yield calls
    finally:
        web_fetch.fetch_response = orig_fetch
        web_fetch.trafilatura.extract = orig_extract


# ---------------- 上限设置 ----------------

def test_webpage_max_length_setting():
    print("== 网页正文上限：可配置 + 硬上限 ==")
    check("常量：硬上限是默认值的 4 倍",
          config.WEBPAGE_MAX_LENGTH_LIMIT == 20000
          and config.WEBPAGE_MAX_LENGTH_LIMIT == config.WEBPAGE_MAX_LENGTH_DEFAULT * 4)
    with env(None):
        check("未配置：5000", config.webpage_max_length() == 5000)
    with env("12000"):
        check("配置 12000 生效", config.webpage_max_length() == 12000)
    with env(" 8000 "):
        check("两侧空白被忽略", config.webpage_max_length() == 8000)
    with env("20000"):
        check("恰好等于硬上限：允许", config.webpage_max_length() == 20000)
    with env("99999"):
        check("超过硬上限：夹到 20000", config.webpage_max_length() == 20000)
    with env("0"):
        check("0：夹到 1", config.webpage_max_length() == 1)
    with env("-3"):
        check("负数：夹到 1", config.webpage_max_length() == 1)
    with env("abc"):
        check("非法值：回退默认 5000", config.webpage_max_length() == 5000)
    with env("5000.5"):
        check("非整数：回退默认 5000", config.webpage_max_length() == 5000)
    with env(""):
        check("空串：回退默认 5000", config.webpage_max_length() == 5000)


# ---------------- 截断行为 ----------------

def test_truncation_follows_setting():
    print("== 截断：跟随配置，且不突破硬上限 ==")
    body = "A" * 30000
    with env("20000"):
        with patched(fetch_result=FakeResponse(), extract_result=body):
            out = web_fetch.fetch_webpage("https://example.com/long")
        check("配置 20000：正文截到 20000（默认时的 4 倍）",
              out.startswith("A" * 20000) and len(out) == 20000 + len(out[20000:]))
        check("截断标记保留旧文案前缀", out[20000:].startswith("\n\n...(已截断"))
        check("截断标记带原文长度", "原文共 30000 字符" in out)
    with env(None):
        with patched(fetch_result=FakeResponse(), extract_result=body):
            out = web_fetch.fetch_webpage("https://example.com/long")
        check("未配置：仍按 5000 截断", out.startswith("A" * 5000) and out[5000:].startswith("\n\n...(已截断"))
        check("未配置：截断标记带原文长度", "原文共 30000 字符" in out)
    with env("99999"):
        with patched(fetch_result=FakeResponse(), extract_result=body):
            out = web_fetch.fetch_webpage("https://example.com/long")
        check("配置超硬上限：实际按 20000 截断",
              out.startswith("A" * 20000) and "原文共 30000 字符" in out)
    with env("0"):
        with patched(fetch_result=FakeResponse(), extract_result=body):
            out = web_fetch.fetch_webpage("https://example.com/long")
        check("下限 1：只保留 1 个字符 + 标记",
              out.startswith("A\n\n...(已截断") and "原文共 30000 字符" in out)
    with env("20000"):
        with patched(fetch_result=FakeResponse(), extract_result="短正文" * 10):
            out = web_fetch.fetch_webpage("https://example.com/short")
        check("未超限：原样返回且无截断标记",
              out == "短正文" * 10 and "已截断" not in out)
    with env(None):
        with patched(fetch_result=FakeResponse(), extract_result=body):
            out = web_fetch.fetch_webpage("https://example.com/long")
        check("默认 5000 是硬上限的 1/4", config.WEBPAGE_MAX_LENGTH_LIMIT // 5000 == 4)


# ---------------- 失败文案带状态码 ----------------

def test_failure_messages_carry_status():
    print("== 失败文案：带上 HTTP 状态码 ==")
    from mincli.controller import ChatController

    def msg(fetch_result=None, extract_result=None, **kw):
        with patched(fetch_result=fetch_result, extract_result=extract_result, **kw):
            return web_fetch.fetch_webpage("https://example.com/p")

    # 状态码标签
    check("标签：404", web_fetch._status_label(404) == "HTTP 404 Not Found")
    check("标签：403", web_fetch._status_label(403) == "HTTP 403 Forbidden")
    check("标签：500", web_fetch._status_label(500) == "HTTP 500 Internal Server Error")
    check("标签：非标准码只给数字", web_fetch._status_label(999) == "HTTP 999")
    check("标签：缺失状态码", web_fetch._status_label(None) == "HTTP 状态未知")

    m404 = msg(fetch_result=FakeResponse(data=b"<html>nope</html>", status=404))
    check("404：无法获取网页内容 + HTTP 404 Not Found",
          m404.startswith("无法获取网页内容:") and "HTTP 404 Not Found" in m404)
    m403 = msg(fetch_result=FakeResponse(data=b"<html>denied</html>", status=403))
    check("403：带状态码", "HTTP 403 Forbidden" in m403)
    m500 = msg(fetch_result=FakeResponse(data=b"<html>boom</html>", status=500))
    check("500：带状态码", m500.startswith("无法获取网页内容:") and "HTTP 500" in m500)
    m999 = msg(fetch_result=FakeResponse(data=b"<html>x</html>", status=999))
    check("非标准码：带数字", "HTTP 999" in m999)

    m_empty = msg(fetch_result=FakeResponse(data=b"", status=200))
    check("响应为空：状态码 + 说明",
          m_empty.startswith("无法获取网页内容:") and "HTTP 200 OK" in m_empty and "响应内容为空" in m_empty)

    m_plain = msg(
        fetch_result=FakeResponse(data=b"RFC text", status=200,
                                  headers={"content-type": "text/plain;charset=utf-8"}),
        extract_result=None,
    )
    check("抽取失败：状态码 + content-type",
          m_plain.startswith("无法从网页中提取有效文本:")
          and "HTTP 200 OK" in m_plain and "text/plain" in m_plain)

    m_none = msg(fetch_result=None)
    check("无响应（网络不可达）：保留前缀并说明原因",
          m_none.startswith("无法获取网页内容:") and "无响应" in m_none)

    m_raise = msg(fetch_result=FakeResponse(), extract_raises=RuntimeError("boom"))
    check("抽取抛异常：前缀 + 状态码 + 异常信息",
          m_raise.startswith("抓取或解析失败:")
          and "HTTP 200 OK" in m_raise and "boom" in m_raise)

    m_fetch_raise = msg(fetch_raises=RuntimeError("ssl fail"))
    check("下载抛异常（无状态码）：前缀 + 异常信息",
          m_fetch_raise.startswith("抓取或解析失败:") and "ssl fail" in m_fetch_raise)

    all_msgs = [m404, m403, m500, m999, m_empty, m_plain, m_none, m_raise, m_fetch_raise]
    check("全部失败文案仍以 _IMPORT_FAIL_PREFIXES 开头（/import 判定不受影响）",
          all(m.startswith(ChatController._IMPORT_FAIL_PREFIXES) for m in all_msgs))
    check("失败文案都带上状态码（无响应/下载异常两种除外）",
          all("HTTP " in m for m in [m404, m403, m500, m999, m_empty, m_plain, m_raise]))


# ---------------- 下载参数 ----------------

def test_fetch_call_shape():
    print("== 下载调用：URL 归一化，与 fetch_url 同一路径 ==")
    with patched(fetch_result=FakeResponse(), extract_result="x") as calls:
        web_fetch.fetch_webpage("example.com/a?b=1")
        url, kwargs = calls["fetch"][0]
        check("无协议：自动补 https://", url == "https://example.com/a?b=1")
        check("解码与响应头都打开（失败时才能报状态码/类型）",
              kwargs.get("decode") is True and kwargs.get("with_headers") is True)
    with patched(fetch_result=FakeResponse(), extract_result="x") as calls:
        web_fetch.fetch_webpage("  http://a.com/x  ")
        check("两侧空白被去掉", calls["fetch"][0][0] == "http://a.com/x")
        check("抽取参数保持不变（不含评论、含表格）",
              calls["extract"][0][1] == {"include_comments": False, "include_tables": True})


if __name__ == "__main__":
    test_webpage_max_length_setting()
    test_truncation_follows_setting()
    test_failure_messages_carry_status()
    test_fetch_call_shape()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
