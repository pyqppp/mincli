"""文件工具：``read_file`` / ``write_file`` / ``edit_file`` / ``list_directory``。

约定：
- 读取按行分段（``offset`` 1 起，``limit`` 行数，默认 200），输出带行号；
- 单次输出有字符上限，超长时在行边界截断并说明；
- 写入磁盘的动作由调用方通过 ``confirm`` 回调决定是否放行。
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

from rich.cells import cell_len

from mincli.tools.file_ops import parse_file

DEFAULT_READ_LIMIT = 200
MAX_READ_LIMIT = 2000
MAX_OUTPUT_CHARS = 60_000

# parse_file 的失败文案前缀（与 attachments 保持一致）
_FAIL_PREFIXES = (
    "文件不存在:",
    "无法读取:",
    "文件内容为空:",
    "不支持的文件格式:",
    "不支持二进制格式:",
    "不支持 .doc 格式",
    "文件解析失败:",
    "需安装",
)


def _is_failure(text: str) -> bool:
    return not text or text.startswith(_FAIL_PREFIXES)


def _as_int(value, default: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _format_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _pad(text: str, width: int) -> str:
    """按显示宽度补空格（中文占两格）。"""
    return text + " " * max(0, width - cell_len(text))


def _split_lines(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _extract_text(path: str) -> Tuple[Optional[str], Optional[str]]:
    """抽取可读文本，返回 (文本, 错误文案)。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdf", ".docx", ".doc", ".csv"):
        raw = parse_file(path)
        if _is_failure(raw):
            return None, raw
        header = f"{os.path.basename(path)}：\n"
        return (raw[len(header):] if raw.startswith(header) else raw), None

    try:
        with open(path, "rb") as f:
            head = f.read(8192)
    except OSError as exc:
        return None, f"读取失败：{exc}"
    if b"\x00" in head:
        return None, (
            f"无法读取二进制文件：{path}"
            "（支持文本、Markdown、CSV、PDF、DOCX；图片请用附件方式导入）"
        )
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), None
    except OSError as exc:
        return None, f"读取失败：{exc}"


def read_file(path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> str:
    """按行读取文件的一段，返回带行号的文本。"""
    raw_target = str(path or "").strip()
    if not raw_target:
        return "错误：未提供文件路径"
    target = os.path.expanduser(raw_target)
    if not os.path.exists(target):
        return f"文件不存在：{target}"
    if os.path.isdir(target):
        return f"这是一个目录：{target}（用 list_directory 查看里面有什么）"

    text, error = _extract_text(target)
    if error:
        return error

    lines = _split_lines(text or "")
    total = len(lines)
    if total == 0:
        return f"文件：{target}（空文件）"

    start = _as_int(offset, 1, 1, total + 1)
    count = _as_int(limit, DEFAULT_READ_LIMIT, 1, MAX_READ_LIMIT)
    if start > total:
        return f"起始行 {start} 超出范围：{target} 共 {total} 行"

    selected = lines[start - 1 : start - 1 + count]
    width = len(str(start + len(selected) - 1))

    rendered: list[str] = []
    used = 0
    for number, line in enumerate(selected, start=start):
        entry = f"{number:>{width}}| {line}"
        if rendered and used + len(entry) + 1 > MAX_OUTPUT_CHARS:
            break
        rendered.append(entry)
        used += len(entry) + 1

    end = start + len(rendered) - 1
    header = f"文件：{target}（共 {total} 行，以下为第 {start}-{end} 行）"
    body = "\n".join(rendered)
    if end < total:
        tail = f"（内容未结束，用 offset={end + 1} 继续读取）"
    else:
        tail = "（已到文件末尾）"
    return f"{header}\n{body}\n{tail}"


def read_whole_file(path: str) -> Tuple[Optional[str], Optional[str]]:
    """整篇读取（供附件导入复用），返回 (文本, 错误文案)。"""
    target = os.path.expanduser(str(path or "").strip())
    if not target or not os.path.isfile(target):
        return None, f"文件不存在：{target}"
    return _extract_text(target)


def write_file(
    path: str,
    content: str,
    confirm: Optional[Callable[[str], bool]] = None,
) -> str:
    """写入（覆盖）文件；``confirm`` 返回 False 时取消。"""
    raw_target = str(path or "").strip()
    if not raw_target:
        return "错误：未提供文件路径"
    target = os.path.expanduser(raw_target)
    if confirm is not None and not confirm(f"写入 {target}"):
        return "已取消：用户拒绝了这次写入"

    try:
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return f"写入失败：{exc}"

    lines = content.count("\n") + 1 if content else 0
    return f"已写入 {target}（{lines} 行 / {len(content)} 字符）"


def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    confirm: Optional[Callable[[str], bool]] = None,
) -> str:
    """把文件里唯一出现的 ``old_string`` 替换成 ``new_string``。"""
    raw_target = str(path or "").strip()
    if not raw_target:
        return "错误：未提供文件路径"
    target = os.path.expanduser(raw_target)
    if not os.path.isfile(target):
        return f"文件不存在：{target}"
    if not old_string:
        return "错误：old_string 不能为空；整体重写请用 write_file"

    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        return f"读取失败：{exc}"

    occurrences = content.count(old_string)
    if occurrences == 0:
        return (
            "未找到要替换的原文。请先用 read_file 查看当前内容，"
            "确保 old_string 与文件完全一致（含空格与换行）"
        )
    if occurrences > 1:
        return (
            f"old_string 在文件中出现 {occurrences} 次，无法确定替换位置。"
            "请补充前后文，让它只匹配一处"
        )

    if confirm is not None and not confirm(f"修改 {target}"):
        return "已取消：用户拒绝了这次修改"

    new_content = content.replace(old_string, new_string, 1)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as exc:
        return f"写入失败：{exc}"

    return f"已修改 {target}（替换 1 处；{len(content)} → {len(new_content)} 字符）"


def list_directory(path: str = ".") -> str:
    """列出目录内容（不递归），目录在前。"""
    target = os.path.expanduser(str(path or ".").strip() or ".")
    if not os.path.isdir(target):
        return f"目录不存在：{target}"

    try:
        entries = sorted(
            os.scandir(target),
            key=lambda entry: (not entry.is_dir(), entry.name.lower()),
        )
    except PermissionError:
        return f"权限不足，无法读取目录：{target}"
    except OSError as exc:
        return f"读取目录失败：{exc}"

    rows: list[str] = []
    name_width = 0
    with_size: list[tuple[str, str]] = []
    for entry in entries:
        name = entry.name + ("/" if entry.is_dir() else "")
        try:
            size = "" if entry.is_dir() else _format_size(entry.stat().st_size)
        except OSError:
            size = ""
        with_size.append((name, size))
        name_width = min(48, max(name_width, cell_len(name)))

    for name, size in with_size:
        rows.append(f"  {_pad(name, name_width)}  {size}".rstrip())

    if not rows:
        return f"目录：{target}（空目录）"
    return f"目录：{target}（{len(rows)} 项）\n" + "\n".join(rows)
