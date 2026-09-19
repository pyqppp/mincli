"""网页抓取：trafilatura 下载并提取正文，失败文案带上 HTTP 状态码。"""

from http import HTTPStatus

from mincli.config import webpage_max_length

# trafilatura 的导入要 0.2s 左右（自身 + htmldate/dateparser 等），而抓网页是
# 低频工具：改成首次真正抓取时才导入，mincli 主进程与内置 MCP server 的启动
# 都不再为它付钱。这里保留 trafilatura / fetch_response 两个模块级名字，测试
# 仍可像以前一样打桩（web_fetch.trafilatura.extract = ...）。


def __getattr__(name: str):
    """惰性暴露 trafilatura / fetch_response（PEP 562）。"""
    if name == "trafilatura":
        global trafilatura
        import trafilatura  # noqa: PLC0415 - 故意延后到首次使用
        return trafilatura
    if name == "fetch_response":
        global fetch_response
        from trafilatura.downloads import fetch_response  # noqa: PLC0415
        return fetch_response
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_deps() -> None:
    """把两个依赖绑成模块全局，供 fetch_webpage 直接引用（也尊重测试的打桩）。"""
    if "trafilatura" not in globals():
        __getattr__("trafilatura")
    if "fetch_response" not in globals():
        __getattr__("fetch_response")


def _status_label(status) -> str:
    """把状态码渲染成如 'HTTP 404 Not Found'；缺失或非标准码时给出可读描述。"""
    if not isinstance(status, int) or status <= 0:
        return "HTTP 状态未知"
    try:
        return f"HTTP {status} {HTTPStatus(status).phrase}"
    except ValueError:  # 非标准状态码（如 999）
        return f"HTTP {status}"


def fetch_webpage(url: str) -> str:
    _ensure_deps()
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    status = None
    try:
        # fetch_response 与 trafilatura.fetch_url 是同一条下载路径（同样的 UA、重试与
        # 编码检测），区别是它把非 200 响应也返回，因此失败文案能带上真正的状态码。
        response = fetch_response(url, decode=True, with_headers=True)
        if response is None:
            return f"无法获取网页内容: {url}（无响应：网络不可达、超时或被拒绝）"
        status = response.status
        if status != 200:
            return f"无法获取网页内容: {url}（{_status_label(status)}）"
        if not response.data:
            return f"无法获取网页内容: {url}（{_status_label(status)}：响应内容为空）"

        html_text = response.html or response.data.decode("utf-8", "replace")
        text = trafilatura.extract(html_text, include_comments=False, include_tables=True)
        if not text:
            ctype = (response.headers or {}).get("content-type") or "未知"
            return (
                f"无法从网页中提取有效文本: {url}"
                f"（{_status_label(status)}，content-type: {ctype}）"
            )

        text = text.strip()
        limit = webpage_max_length()
        if len(text) > limit:
            total = len(text)  # 截断前的原始长度
            text = text[:limit] + f"\n\n...(已截断，原文共 {total} 字符)"
        return text
    except Exception as e:
        suffix = f"（{_status_label(status)}）" if status is not None else ""
        return f"抓取或解析失败: {url}{suffix}: {e}"
