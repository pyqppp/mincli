import trafilatura

from mincli.config import WEBPAGE_MAX_LENGTH


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
