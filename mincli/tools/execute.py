import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

from openai import OpenAI

from mincli.config import (
    EXEC_DEFAULT_MAX_OUTPUT,
    EXEC_DEFAULT_TIMEOUT,
    EXEC_MAX_OUTPUT,
    EXEC_MAX_TIMEOUT,
    MODEL_V4_FLASH,
)
from mincli.streaming import stream_response
from mincli.tools.thinking import AUDIT_SYSTEM_PROMPT


# ---------------- 高危命令模式（文本硬门） ----------------
# 命中任意一条即视为高危：所有审核级别（除“无审核”）都会强制要求用户确认。
DANGEROUS_PATTERNS = [
    # 递归删除 ~ 或根目录/顶级目录（含 -r -f 拆分、--no-preserve-root 变体）
    r'\brm\s+(-[a-zA-Z]+\s+)*-?[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(~(/[^\s]*)?|/(?:[^\s/]+)?)(?:\s|$)',
    r'\brm\s+(-[a-zA-Z]+\s+)*-?[a-zA-Z]*f[a-zA-Z]*\s+(~(/[^\s]*)?|/(?:[^\s/]+)?)(?:\s|$)',
    r'\brm\s+.*--no-preserve-root\b',
    # 磁盘/分区/格式化/直接写裸设备
    r'\bdd\s+if=',
    r'\b(mkfs|fdisk|parted|gdisk|sfdisk)\b',
    r'\bdiskutil\s+(erase|destroy|zeroDisk|partitionDisk|reformat)\b',
    r'>\s*/dev/(sd[a-z]|disk\w*|rdisk\w*)',
    r'\bmount\s+/dev/',
    # 系统关机/重启/停机
    r'\bshutdown\b',
    r'\breboot\b',
    r'\bpoweroff\b',
    r'\bhalt\b',
    r'\binit\s+0\b',
    # 系统目录权限/属主批量修改（仅根/顶级目录）
    r'\bchmod\s+(-R\s+)?[0-7]{3,4}\s+/(?:[^\s/]+)?(?:\s|$)',
    r'\bchmod\s+(-R\s+)?000\s+',
    r'\bchown\s+(-R\s+)?[^ ]+\s+/(?:[^\s/]+)?(?:\s|$)',
    # 下载即执行 / 管道提权
    r'\b(wget|curl)\s+.*(\||;)\s*(sudo\s+)?(bash|sh|zsh)\b',
    r'\bcurl\s+.*(-o\s+|>)+\s*/etc/',
    # 覆盖系统关键文件
    r'>\s*/etc/(passwd|shadow|sudoers|hosts|fstab|crontab)',
    r'\btee\s+/etc/(passwd|shadow|sudoers|hosts|fstab|crontab)',
    # 脚本内执行毁灭性操作
    r':\(\)\s*\{.*\};',
    r'python[23]?\s+-c\s+.*shutil\.rmtree',
    # 强制推送 / 危险 git 操作
    r'\bgit\s+push\s+.*--force',
    r'\bgit\s+reset\s+--hard',
    # 批量杀进程（影响面大）
    r'\b(pkill|killall)\b',
    r'\bkill\s+-9\s+-\d+',
    # 全盘 find 删除
    r'\bfind\s+/[^\s]*\s+.*\s+-delete\b',
    # macOS 自动化脚本（可控制系统）
    r'\bosascript\b',
    r'\blaunchctl\s+(load|unload|kickstart|bootout)\b',
    # sudo 直接执行毁灭性命令
    r'\bsudo\s+(rm|dd|mkfs|shutdown|reboot|poweroff|halt)\b',
]


def matches_dangerous(command: str) -> bool:
    """是否命中高危命令模式（文本硬门）。"""
    return any(re.search(p, command) for p in DANGEROUS_PATTERNS)


# ---------------- 只读快速通道 ----------------
# 无 shell 元字符 + 白名单只读命令 → 跳过 AI 审核（level-2 自动执行 / level-1 仅确认）。
# 注意：不包含 git（git 有写入/推送等副作用，走 AI 审核；审核结果有会话缓存）。
READ_ONLY_COMMANDS = frozenset({
    "ls", "pwd", "date", "echo", "cat", "head", "tail", "wc", "df", "du",
    "whoami", "uname", "which", "type", "file", "stat", "cal", "uptime",
    "env", "printenv", "grep", "find", "history", "ps", "free", "true",
    "false", "printf", "basename", "dirname", "hostname", "man", "help",
    "lsb_release", "sw_vers",
})

# 命中任一字符即视为“非只读”（重定向、管道、后台、命令替换、通配符、引号等）
_SHELL_METACHARS = re.compile(r"[|;&`$<>()\[\]{}\"'*?~!#\\\n]")

# 支持的 shell -> 可执行文件
_SHELLS = {"sh": "/bin/sh", "bash": "/bin/bash", "zsh": "/bin/zsh"}


def is_safe_readonly(command: str) -> bool:
    """是否为可跳过 AI 审核的纯只读命令（无 shell 元字符、白名单命令、非高危）。"""
    if not command or matches_dangerous(command):
        return False
    if _SHELL_METACHARS.search(command):
        return False
    tokens = command.strip().split()
    if not tokens:
        return False
    head = os.path.basename(tokens[0])
    return head in READ_ONLY_COMMANDS


# ---------------- 执行 ----------------

def _clamp(value, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _compose_body(stdout: str, stderr: str, exit_code: Optional[int]) -> str:
    """把 stdout/stderr/退出码拼成结构化文本：退出码前置，stderr 独立标记。"""
    parts = []
    if exit_code is not None:
        parts.append(f"[退出码: {exit_code}]")
    out = (stdout or "").rstrip()
    if out:
        parts.append(out)
    err = (stderr or "").strip()
    if err:
        parts.append(f"[stderr]\n{err}")
    return "\n".join(parts)


def _truncate_output(body: str, max_chars: int) -> str:
    """输出超限时保留首尾各一半，并把完整输出写入 /tmp 临时文件返回路径。"""
    if len(body) <= max_chars:
        return body
    half = max_chars // 2
    head = body[:half]
    tail = body[-half:]
    path = ""
    try:
        digest = hashlib.md5(body.encode("utf-8", "replace")).hexdigest()[:8]
        path = os.path.join(tempfile.gettempdir(), f"mincli_exec_{digest}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        path = f"；完整输出见 {path}"
    except Exception:
        path = ""
    marker = f"\n[输出已截断：共 {len(body)} 字符，仅保留首尾各 {half} 字符{path}]\n"
    return head + marker + tail


def execute_command(
    command: str,
    timeout: int = EXEC_DEFAULT_TIMEOUT,
    cwd: str = "",
    env: Optional[Dict[str, str]] = None,
    shell: str = "sh",
    max_output: int = EXEC_DEFAULT_MAX_OUTPUT,
) -> str:
    """执行 shell 命令（非交互：stdin 关闭，超时终止整个进程组）。"""
    # --- 参数校验/钳制 ---
    timeout = _clamp(timeout, EXEC_DEFAULT_TIMEOUT, 1, EXEC_MAX_TIMEOUT)
    max_output = _clamp(max_output, EXEC_DEFAULT_MAX_OUTPUT, 1, EXEC_MAX_OUTPUT)
    shell = shell if shell in _SHELLS else "sh"
    exe = _SHELLS[shell]
    workdir = os.path.expanduser(cwd) if cwd else (
        os.environ.get("MINCLI_WORKSPACE") or os.getcwd()
    )
    proc_env = {**os.environ, **({k: str(v) for k, v in env.items()} if env else {})}
    platform_line = (
        f"[平台: {platform.system()} {platform.release()} | "
        f"工作目录: {workdir} | shell: {shell}]"
    )

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=exe,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            env=proc_env,
            start_new_session=True,  # 独立进程组：超时可整组终止，避免孤儿进程
        )
    except Exception as e:
        return f"{platform_line}\n[命令执行失败] {e}"

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # 终止整个进程组（含子 shell 派生的孙进程）
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        body = _truncate_output(_compose_body(stdout, stderr, None), max_output)
        return f"{platform_line}\n[命令超时（{timeout}秒），已强制终止整个进程组，以上为已产生的部分输出]\n{body}"
    except Exception as e:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        return f"{platform_line}\n[命令执行失败] {e}"

    body = _truncate_output(_compose_body(stdout, stderr, proc.returncode), max_output)
    return f"{platform_line}\n{body}"


# ---------------- 审核 ----------------

def _extract_json(text: str) -> Optional[dict]:
    """从模型回复中提取第一个完整 JSON 对象。

    容忍 ```json 代码围栏、前后杂文与嵌套花括号（简单正则截取会误伤）。
    """
    text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# 会话内审核缓存：同一命令只审核一次（level-1/2 下反复执行同一命令省去重复审核）
_AUDIT_CACHE: Dict[str, Tuple[int, str, str, str]] = {}


def audit_command(
    client: OpenAI, command: str, use_cache: bool = True
) -> Tuple[int, str, str, str]:
    """AI 审核命令安全性，返回 (level 1-5, 描述, 风险, 审核思考)。

    level 经过强转与钳制（1..5），杜绝字符串/越界值导致的崩溃或策略扭曲。
    """
    cached = _AUDIT_CACHE.get(command) if use_cache else None
    if cached is not None:
        return cached

    platform_line = (
        f"当前运行环境: {platform.system()} {platform.release()} | "
        f"用户默认 shell: {os.environ.get('SHELL', '/bin/sh')}"
    )
    audit_messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT + "\n\n" + platform_line},
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
    level, desc, risk = 3, "", ""
    parsed = _extract_json(content)
    if parsed is not None:
        try:
            level = int(parsed.get("level", 3))
        except (TypeError, ValueError):
            level = 3
        level = max(1, min(5, level))
        desc = str(parsed.get("description", "") or "")[:200]
        risk = str(parsed.get("risk", "") or "")[:500]
    else:
        # 审核失败：降级为中性并附上原文片段，避免静默吞错
        desc = (content.strip() or "（审核无返回内容）")[:100]
        risk = "AI 审核未返回有效 JSON，按中性处理"
    result = (level, desc, risk, reasoning)
    if use_cache:
        _AUDIT_CACHE[command] = result
    return result
