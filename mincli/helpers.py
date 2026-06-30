import os
import re
import sys
import datetime
from typing import Optional, List, Dict

import requests
import tiktoken
from openai import OpenAI

from mincli.config import (
    MODEL_V4_FLASH, TITLE_MAX_TOKENS, TITLE_MAX_LENGTH,
    SAVE_BASE_DIR, TEMPERATURE_MIN, TEMPERATURE_MAX,
)
from mincli.render import console


def clear_screen() -> None:
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        sys.stdout.write("\033]1337;ClearScrollback\007")
        sys.stdout.flush()
    else:
        os.system('cls' if os.name == 'nt' else 'clear')


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
            if key == "name":
                tokens += 1
    tokens += 3
    return tokens


def generate_conversation_title(client: OpenAI, user_msg: str, assistant_msg: str) -> str:
    try:
        prompt = (
            "请用不超过30字的汉字为以下内容写一个标题，标题内容简略，只输出标题，不要有其他解释，"
            "不要包含标点符号和特殊字符。\n\n"
            f"用户：{user_msg}\n助手：{assistant_msg}"
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
        console.print(f"[red]⚠️ 生成标题失败: {e}[/red]")
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
