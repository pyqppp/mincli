import os
import re
import json
import subprocess
from typing import Tuple

from openai import OpenAI

from mincli.config import MODEL_V4_FLASH
from mincli.streaming import stream_response
from mincli.tools.thinking import AUDIT_SYSTEM_PROMPT


DANGEROUS_PATTERNS = [
    r'\brm\s+-rf\s+/\s*(?:$|\s)',
    r'\brm\s+-rf\s+/\*',
    r'\brm\s+-rf\s+~',
    r'\bdd\s+if=',
    r'>\s*/dev/sd[a-z]',
    r'\bmkfs\.',
    r':\(\)\s*\{.*\};',
    r'\bchmod\s+-R\s+000\s+/',
    r'\bwget\s+.*\|\s*(bash|sh)\b',
    r'\bcurl\s+.*\|\s*(bash|sh)\b',
    r'\bshutdown\b',
    r'\breboot\b',
    r'\bpoweroff\b',
    r'\binit\s+0\b',
    r'\b>\s*/dev/sda',
]


def matches_dangerous(command: str) -> bool:
    return any(re.search(p, command) for p in DANGEROUS_PATTERNS)


def execute_command(command: str, timeout: int) -> str:
    try:
        workdir = os.path.expanduser("~")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=workdir,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"[stderr]\n{result.stderr}"
        output += f"\n[退出码: {result.returncode}]"
        return output.strip()
    except subprocess.TimeoutExpired as e:
        partial = ""
        if e.stdout:
            partial += e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8")
        if e.stderr:
            partial += f"[stderr]\n{e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8')}"
        partial = partial.strip()
        if partial:
            return f"{partial}\n[命令执行超时（{timeout}秒），以上为已产生的部分输出]"
        return f"命令执行超时（{timeout}秒），无任何输出"
    except Exception as e:
        return f"命令执行失败: {e}"


def audit_command(client: OpenAI, command: str) -> Tuple[int, str, str, str]:
    audit_messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请审核以下命令：\n\n```bash\n{command}\n```"},
    ]
    sr = stream_response(
        client, audit_messages, MODEL_V4_FLASH,
        0.3, command,
        thinking_enabled=True,
        reasoning_effort="high",
        tools=None,
        silent=True,
    )
    content = sr.content or ""
    reasoning = sr.reasoning or ""
    try:
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("level", 3), result.get("description", ""), result.get("risk", ""), reasoning
    except Exception:
        pass
    return 3, content.strip()[:100], "", reasoning
