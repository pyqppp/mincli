import os

import requests
import trafilatura

from mincli.config import WEBPAGE_MAX_LENGTH, BOCHA_API_BASE


def fetch_webpage(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"无法获取网页内容: {url}"
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if not text:
            return f"无法从网页中提取有效文本: {url}"

        text = text.strip()
        if len(text) > WEBPAGE_MAX_LENGTH:
            text = text[:WEBPAGE_MAX_LENGTH] + "\n\n...(已截断)"
        return text
    except Exception as e:
        return f"抓取或解析失败: {e}"


def web_search(query: str, freshness: str = "noLimit", count: int = 10) -> str:
    api_key = os.getenv("BOCHA_API_KEY")
    if not api_key:
        return "错误: 未配置 BOCHA_API_KEY"
    try:
        resp = requests.post(
            BOCHA_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "freshness": freshness, "summary": True, "count": min(count, 50)},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("data", {}).get("webPages", {}).get("value", [])
        if not pages:
            return f"搜索 \"{query}\" 未找到相关结果"
        lines = [f"搜索 \"{query}\" 共找到 {len(pages)} 条结果：\n"]
        for i, p in enumerate(pages, 1):
            name = p.get("name", "")
            url = p.get("url", "")
            snippet = p.get("snippet", "")
            date = (p.get("dateLastCrawled") or "")[:10]
            lines.append(f"{i}. {name}\n   链接: {url}\n   摘要: {snippet}\n   日期: {date}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索请求失败: {e}"
