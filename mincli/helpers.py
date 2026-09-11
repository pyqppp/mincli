import os
import re
import shlex
import sys
import datetime
from typing import Optional, List, Dict

import requests
import tiktoken
from openai import OpenAI

from mincli.config import (
    MODEL_V4_FLASH, TITLE_MAX_TOKENS, TITLE_MAX_LENGTH,
    SAVE_BASE_DIR, TEMPERATURE_MIN, TEMPERATURE_MAX,
    DEEPSEEK_PRICING, VISION_IMAGE_TOKEN_CAP,
)


def clear_screen() -> None:
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        sys.stdout.write("\033]1337;ClearScrollback\007")
        sys.stdout.flush()
    else:
        os.system('cls' if os.name == 'nt' else 'clear')


# ---------------- 跨平台路径/URL 参数解析（/import、拖入/粘贴导入） ----------------

# Windows 盘符路径（如 C:\Users\me\a.txt）。POSIX 版 shlex 会把 "\U" 当转义序列
# 处理从而吃掉反斜杠（C:\Users\me\a.txt → C:Usersmenotes.txt），导致 Windows
# 上拖入/粘贴与 /import 全部失效。
_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:\\")


def _strip_matching_quotes(token: str) -> str:
    """去掉 token 两端成对的引号（终端拖入路径常带引号）。"""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _is_import_target_token(token: str) -> bool:
    """token 是否可直接作为导入目标（http(s) URL 或存在的文件）。"""
    if token.lower().startswith(("http://", "https://")):
        return True
    try:
        return os.path.isfile(os.path.expanduser(token))
    except OSError:
        return False


def split_path_args(text: str) -> List[str]:
    """把一行（可含多个）文件路径/URL 解析为 token 列表，跨平台。

    决策顺序：
    1. 优先返回“全部 token 都能作为导入目标（存在的文件或 http(s) URL）”的
       解析结果，保证 POSIX 转义写法（如 ``/Users/a\\ b.txt``）与 Windows
       引号写法都能正确落地；
    2. 否则若文本含 Windows 盘符路径（``C:\\...``），使用**非 POSIX** 解析
       （保留反斜杠），避免 Windows 路径被当作转义序列破坏；
    3. 其余情况沿用 POSIX 解析（支持 ``\\`` 转义与引号），失败再退回非 POSIX。

    返回空列表表示没有任何可解析的参数。解析失败（引号不匹配）不抛异常，
    尽量返回可用的 token（错误文件由导入流程给出“文件不存在”提示）。
    """
    text = (text or "").strip()
    if not text:
        return []
    posix: List[str] = []
    nonposix: List[str] = []
    try:
        posix = [_strip_matching_quotes(t) for t in shlex.split(text, posix=True)]
    except ValueError:
        pass
    try:
        nonposix = [_strip_matching_quotes(t) for t in shlex.split(text, posix=False)]
    except ValueError:
        pass
    for tokens in (posix, nonposix):
        if tokens and all(_is_import_target_token(t) for t in tokens):
            return tokens
    if _WINDOWS_DRIVE_RE.search(text) and nonposix:
        return nonposix
    return posix or nonposix


def is_peak_hour(now: Optional[datetime.datetime] = None) -> bool:
    """DeepSeek 峰谷定价：北京时间高峰时段 9:00-12:00、14:00-18:00。"""
    if now is None:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    h = now.hour
    return (9 <= h < 12) or (14 <= h < 18)


def estimate_input_price(
    model: str,
    tokens: int,
    hit_ratio: Optional[float] = None,
    peak: bool = False,
) -> Optional[float]:
    """估算输入价格（元）：命中部分按缓存命中单价、其余按未命中单价。

    仅适配 DeepSeek 官方定价（config.DEEPSEEK_PRICING）；未知模型返回 None。
    """
    pricing = DEEPSEEK_PRICING.get(model)
    if not pricing or tokens <= 0:
        return None
    idx = 1 if peak else 0
    hit_price = pricing["hit"][idx]
    miss_price = pricing["miss"][idx]
    ratio = hit_ratio if hit_ratio is not None else 0.0
    ratio = max(0.0, min(1.0, ratio))
    avg_price = hit_price * ratio + miss_price * (1.0 - ratio)
    return tokens * avg_price / 1_000_000


def get_balance(client: OpenAI) -> Optional[List[Dict]]:
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        resp = requests.get("https://api.deepseek.com/user/balance", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("balance_infos")
    except Exception:
        return None


def format_balance(balance_infos: Optional[List[Dict]]) -> str:
    if not balance_infos:
        return ""
    parts = []
    for info in balance_infos:
        currency = info.get("currency", "")
        total = info.get("total_balance", "0.00")
        granted = info.get("granted_balance", "0.00")
        topped_up = info.get("topped_up_balance", "0.00")
        parts.append(f"{currency} ¥{total}（赠金:¥{granted} 充值:¥{topped_up}）")
    return " | ".join(parts)


def estimate_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return 0
    tokens = 0
    for msg in messages:
        tokens += 3
        for key, value in msg.items():
            if isinstance(value, str):
                tokens += len(encoding.encode(value))
            elif key == "content" and isinstance(value, list):
                # 多模态：content 为内容块数组（text 走 tiktoken，图片按上限估算）
                for block in value:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        tokens += len(encoding.encode(block.get("text") or ""))
                    elif btype in ("image_url", "file"):
                        tokens += VISION_IMAGE_TOKEN_CAP
            if key == "name":
                tokens += 1
    tokens += 3
    return tokens


def generate_conversation_title(client: OpenAI, user_msg: str) -> str:
    try:
        prompt = (
            "请用不超过30字的汉字概括以下用户请求，写一个简略标题，"
            "标题内容简略，只输出标题，不要有其他解释，"
            "不要包含标点符号和特殊字符。\n\n"
            f"用户：{user_msg}"
        )
        resp = client.chat.completions.create(
            model=MODEL_V4_FLASH,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=TITLE_MAX_TOKENS,
            extra_body={"thinking": {"type": "disabled"}},
        )
        title = resp.choices[0].message.content.strip()
        title = re.sub(r'[\\/*?:"<>|]', '', title)
        title = title.replace(' ', '_')
        if len(title) > TITLE_MAX_LENGTH:
            title = title[:TITLE_MAX_LENGTH]
        return title if title else f"对话_{datetime.datetime.now().strftime('%H%M%S')}"
    except Exception as e:
        print(f"⚠️ 生成标题失败: {e}")
        return f"对话_{datetime.datetime.now().strftime('%H%M%S')}"


def convert_formulas(text: str) -> str:
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text


def save_conversation_to_file(
    content: str,
    title: str,
    extra_prefix: str = "",
    token_stats: Optional[Dict[str, int]] = None,
) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{extra_prefix}_" if extra_prefix else ""
    filename = f"{prefix}{title}_{timestamp}.md"

    os.makedirs(SAVE_BASE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_BASE_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        if token_stats:
            f.write(f"\n## Token 统计\n\n")
            f.write(f"- 输入 tokens: {token_stats.get('input_tokens', 0)}\n")
            f.write(f"- 输出 tokens: {token_stats.get('output_tokens', 0)}\n")

    return filepath
