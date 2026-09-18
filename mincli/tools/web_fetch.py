"""网页抓取：trafilatura 下载并提取正文，失败文案带上 HTTP 状态码。"""

from http import HTTPStatus

import trafilatura
from trafilatura.downloads import fetch_response

from mincli.config import webpage_max_length


def _status_label(status) -> str:
    """把状态码渲染成如 'HTTP 404 Not Found'；缺失或非标准码时给出可读描述。"""
    if not isinstance(status, int) or status <= 0:
        return "HTTP 状态未知"
    try:
        return f"HTTP {status} {HTTPStatus(status).phrase}"
    except ValueError:  # 非标准状态码（如 999）
        return f"HTTP {status}"


def fetch_webpage(url: str) -> str:
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
