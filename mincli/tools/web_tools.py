"""联网工具：``web_search`` / ``web_fetch``，走 DeepSeek 官方 Search / Fetch API。

接口文档：https://api-docs.deepseek.com/zh-cn/guides/search/
- Search: ``POST /v1/tools/search``，字段 ``query`` / ``count`` / ``need_content``
- Fetch:  ``POST /v1/tools/fetch``，字段 ``url``，返回 ``title`` + Markdown ``content``

两个接口都按成功请求次数计费（各 5 元 / 1000 次），失败不计费。
"""

from __future__ import annotations

from typing import Any

import httpx

from mincli.config import DEEPSEEK_BASE_URL, api_key, webpage_max_length

SEARCH_TIMEOUT = 30.0
FETCH_TIMEOUT = 60.0
MAX_RESULTS = 50
DEFAULT_RESULTS = 5
MAX_SEARCH_CHARS = 12_000
MAX_FETCH_CHARS = 20_000
CONTENT_PREVIEW_CHARS = 1200

_ERROR_HINTS = {
    400: "请求参数不合法",
    401: "API Key 缺失或无效",
    402: "账号余额不足",
    403: "请求被安全策略拒绝",
    404: "未能获取可用的网页内容（URL 不可用、被站点拦截或没有可提取的正文）",
    408: "读取请求超时",
    413: "请求体超过大小限制",
    429: "超过接口限速，请稍后再试",
    500: "服务端内部错误",
    504: "下游请求超时",
}


class WebApiError(Exception):
    """联网接口错误，消息可直接展示给模型。"""


def _api_key() -> str:
    key = api_key()
    if not key:
        raise WebApiError("未配置 DEEPSEEK_API_KEY，联网搜索与网页抓取不可用")
    return key


def _error_message(status: int, payload: Any) -> str:
    detail = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "").strip()
    hint = _ERROR_HINTS.get(status) or ("超出请求次数限制" if status == 429 else "请求失败")
    return f"联网失败（HTTP {status}）：{hint}" + (f"；{detail}" if detail else "")


def _post(path: str, payload: dict, timeout: float) -> dict:
    key = _api_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    url = f"{DEEPSEEK_BASE_URL}{path}"
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        raise WebApiError(f"联网失败：请求超时（超过 {int(timeout)} 秒）") from None
    except httpx.HTTPError as exc:
        raise WebApiError(f"联网失败：{exc}") from None

    try:
        body = response.json()
    except ValueError:
        body = None

    if response.status_code != 200:
        raise WebApiError(_error_message(response.status_code, body))
    if not isinstance(body, dict):
        raise WebApiError("联网失败：响应不是合法的 JSON 对象")
    return body


def _as_int(value, default: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def web_search(query: str, count: int = DEFAULT_RESULTS, need_content: bool = False) -> str:
    """搜索网页，返回标题、链接与摘要。"""
    text = str(query or "").strip()
    if not text:
        return "错误：query 不能为空"
    if len(text) > 128:
        return "错误：query 不能超过 128 个字符"

    total = _as_int(count, DEFAULT_RESULTS, 1, MAX_RESULTS)
    payload = {"query": text, "count": total, "need_content": bool(need_content)}
    try:
        body = _post("/v1/tools/search", payload, SEARCH_TIMEOUT)
    except WebApiError as exc:
        return str(exc)

    results = body.get("results")
    if not isinstance(results, list) or not results:
        return f"搜索「{text}」没有返回结果"

    blocks: list[str] = []
    used = 0
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip() or "(无标题)"
        url = str(item.get("url") or "").strip()
        summary = str(item.get("summary") or item.get("snippet") or "").strip()
        meta = []
        site = str(item.get("site_name") or "").strip()
        date = str(item.get("date") or "").strip()
        if site:
            meta.append(site)
        if date:
            meta.append(date[:10])

        lines = [f"{index}. {title}"]
        if url:
            lines.append(f"   {url}")
        if meta:
            lines.append(f"   {' | '.join(meta)}")
        if summary:
            lines.append(f"   {summary}")

        content = str(item.get("content") or "").strip()
        if need_content and content:
            if len(content) > CONTENT_PREVIEW_CHARS:
                content = content[:CONTENT_PREVIEW_CHARS] + "…（正文已截断，可用 web_fetch 取全文）"
            lines.append(f"   正文摘录：{content}")

        block = "\n".join(lines)
        if used + len(block) + 2 > MAX_SEARCH_CHARS:
            blocks.append(f"（其余 {len(results) - index + 1} 条结果因长度限制未展示）")
            break
        blocks.append(block)
        used += len(block) + 2

    return f"搜索「{text}」，共 {len(results)} 条结果：\n\n" + "\n\n".join(blocks)


def normalize_url(url: str) -> str:
    """补全协议、去掉首尾空白；非法时抛 WebApiError。"""
    target = str(url or "").strip()
    if not target:
        raise WebApiError("错误：url 不能为空")
    if not target.startswith(("http://", "https://")):
        target = "https://" + target
    if len(target) > 8192:
        raise WebApiError("错误：url 超过 8192 个字符")
    return target


def fetch_page(url: str) -> tuple:
    """抓取网页，返回 ``(标题, Markdown 正文)``；失败抛 :class:`WebApiError`。"""
    target = normalize_url(url)
    body = _post("/v1/tools/fetch", {"url": target}, FETCH_TIMEOUT)
    title = str(body.get("title") or "").strip()
    content = str(body.get("content") or "").strip()
    if not content:
        raise WebApiError(f"抓取失败：{target} 没有可用的正文")
    return title, content


def web_fetch(url: str) -> str:
    """抓取网页，返回标题与 Markdown 正文（工具结果文本）。"""
    target = normalize_url(url)
    try:
        title, content = fetch_page(target)
    except WebApiError as exc:
        return str(exc)

    total = len(content)
    limit = min(webpage_max_length(), MAX_FETCH_CHARS)
    truncated = total > limit
    if truncated:
        content = content[:limit]
    header = f"网页：{title or target}\n链接：{target}"
    tail = f"\n\n（正文已截断，原文共 {total} 字符）" if truncated else ""
    return f"{header}\n\n{content}{tail}"
