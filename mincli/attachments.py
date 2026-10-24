"""附件模型：导入解析、引用块标签、截断与消息组装。

设计约定（与重构规格一致）：

- 输入框里只放引用块（如 ``[DOCX 报告.docx]``），正文内容不塞进用户原话；
- 发送时内容统一追加在用户文字之后，用 ``---`` 分隔；
- 同一附件在一轮消息里只附一次内容，引用块出现几次就保留几次；
- 正文超过 ``inline_max_chars`` 时按行边界截断，并提示用 ``read_attachment``
  从下一行继续读取；
- 图片不进文本段，仍以多模态块发送，文本里只留引用块。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from mincli.tools.file_ops import parse_file
from mincli.tools.images import (
    ImageAttachment,
    is_image_path,
    looks_like_image_target,
    make_path_attachment,
    make_url_attachment,
)
from mincli.tools.web_tools import WebApiError, fetch_page

# 附件正文内联上限（字符）。超出的部分不随消息发送，改由 read_attachment 分段读取。
INLINE_MAX_CHARS = 3000

# 引用块里网页标题的最大显示长度
WEB_TITLE_MAX_CHARS = 24

# parse_file / fetch_webpage 的失败文案前缀（视为导入失败而非正文）
_FAIL_PREFIXES = (
    "文件不存在:",
    "无法获取网页内容:",
    "无法从网页中提取有效文本:",
    "抓取或解析失败:",
    "无法读取:",
    "文件内容为空:",
    "不支持的文件格式:",
    "不支持二进制格式:",
    "不支持 .doc 格式",
    "文件解析失败:",
    "需安装",
)

KIND_IMAGE = "image"
KIND_DOCUMENT = "document"
KIND_WEB = "web"


@dataclass
class Attachment:
    """一个已导入的附件（文件、网页或图片）。

    ``text`` 只在文档/网页附件上有值；图片走 ``image``，正文由多模态块承载。
    """

    kind: str
    name: str                    # 引用块里显示的名字（文件名，或网页的 域名/标题）
    path: str = ""               # 本地绝对路径或 URL
    text: str = ""               # 抽取出的正文（文档 / 网页）
    image: Optional[ImageAttachment] = None
    is_url: bool = False
    attachment_id: str = ""
    stored_path: str = ""        # 正文落盘副本（长附件续读用）

    @property
    def dedupe_key(self) -> str:
        """同一轮去重的键：文件按真实路径，网页按 URL。"""
        if self.is_url:
            return self.path
        return os.path.realpath(self.path) if self.path else self.name

    @property
    def chip_type(self) -> str:
        """引用块里的类型标记：IMAGE / WEB / 扩展名大写。"""
        if self.kind == KIND_IMAGE:
            return "IMAGE"
        if self.kind == KIND_WEB:
            return "WEB"
        ext = os.path.splitext(self.name)[1].lstrip(".").lower()
        return (ext or "file").upper()

    @property
    def chip_label(self) -> str:
        return f"[{self.chip_type} {self.name}]"

    @property
    def line_count(self) -> int:
        if not self.text:
            return 0
        return self.text.count("\n") + 1

    @property
    def char_count(self) -> int:
        return len(self.text or "")

    def metadata(self) -> dict:
        """随节点序列化的元数据（不含正文，正文在 stored_path）。"""
        return {
            "id": self.attachment_id,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "stored_path": self.stored_path,
            "lines": self.line_count,
            "chars": self.char_count,
        }

    @classmethod
    def from_metadata(cls, data: dict) -> "Attachment":
        """从节点元数据恢复一个只带落盘正文的附件。"""
        return cls(
            kind=str(data.get("kind") or KIND_DOCUMENT),
            name=str(data.get("name") or ""),
            path=str(data.get("path") or ""),
            is_url=bool(data.get("path", "").startswith(("http://", "https://"))),
            attachment_id=str(data.get("id") or ""),
            stored_path=str(data.get("stored_path") or ""),
        )


def _is_failure(text: str) -> bool:
    return not text or text.startswith(_FAIL_PREFIXES)


def _name_from_url(url: str) -> str:
    tail = os.path.basename(urlparse(url).path.rstrip("/"))
    return tail or urlparse(url).netloc or "image"


def _title_from_text(text: str) -> str:
    """从网页正文首行猜一个标题（Fetch API 的正式标题在 Phase 3 接入）。"""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        line = line.replace("]", "）").replace("[", "（")
        return line[:24]
    return ""


def _strip_parse_header(raw: str, path: str) -> str:
    """parse_file 会在正文前加 ``文件名：``，去掉它只留正文。"""
    header = f"{os.path.basename(path)}：\n"
    if raw.startswith(header):
        return raw[len(header):]
    return raw


def _make_id(key: str, text: str) -> str:
    """内容哈希：同一路径不同内容会得到不同 id，避免旧副本被复用。"""
    digest = hashlib.sha1()
    digest.update(key.encode("utf-8", "ignore"))
    digest.update(b"\0")
    digest.update((text or "").encode("utf-8", "ignore"))
    return digest.hexdigest()[:16]


def load_attachment(target: str, detail: str = "auto") -> Attachment:
    """把路径或 URL 解析成 Attachment；失败抛 ``ValueError``（消息可直接展示）。"""
    target = (target or "").strip()
    if not target:
        raise ValueError("空路径")

    if re.match(r"^https?://", target):
        if looks_like_image_target(target):
            try:
                image = make_url_attachment(target, detail)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            return Attachment(
                kind=KIND_IMAGE,
                name=image.name or _name_from_url(target),
                path=target,
                image=image,
                is_url=True,
                attachment_id=_make_id(target, ""),
            )
        try:
            title, text = fetch_page(target)
        except WebApiError as exc:
            raise ValueError(str(exc)) from exc
        domain = urlparse(target).netloc or target
        title = (title or "").strip().replace("]", "）").replace("[", "（")
        if len(title) > WEB_TITLE_MAX_CHARS:
            title = title[:WEB_TITLE_MAX_CHARS]
        name = f"{domain}/{title}" if title else domain
        return Attachment(
            kind=KIND_WEB,
            name=name,
            path=target,
            text=text,
            is_url=True,
            attachment_id=_make_id(target, text),
        )

    path = os.path.expanduser(target)
    if not os.path.isfile(path):
        raise ValueError(f"文件不存在: {path}")
    abspath = os.path.abspath(path)

    if is_image_path(path):
        try:
            image = make_path_attachment(abspath, detail)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return Attachment(
            kind=KIND_IMAGE,
            name=image.name or os.path.basename(abspath),
            path=abspath,
            image=image,
            attachment_id=_make_id(abspath, ""),
        )

    raw = parse_file(abspath)
    if _is_failure(raw):
        raise ValueError(raw)
    text = _strip_parse_header(raw, abspath)
    return Attachment(
        kind=KIND_DOCUMENT,
        name=os.path.basename(abspath),
        path=abspath,
        text=text,
        attachment_id=_make_id(abspath, text),
    )


def truncate_at_line(text: str, limit: int) -> Tuple[str, int]:
    """截到不超过 ``limit`` 字符的完整行，返回 (截断结果, 已含行数)。

    整篇只有一行且超长时退化为按字符截断（此时已含行数为 1）。
    """
    if len(text) <= limit:
        return text, text.count("\n") + 1
    cut = text[:limit]
    newline = cut.rfind("\n")
    if newline > 0:
        cut = cut[:newline]
    return cut, cut.count("\n") + 1


def _document_block(att: Attachment, inline_max_chars: int) -> str:
    raw = (att.text or "").replace("\r\n", "\n").strip("\n")
    total_lines = raw.count("\n") + 1
    if len(raw) <= inline_max_chars:
        return f"{att.name}（共 {total_lines} 行）：\n{raw}"

    cut, shown_lines = truncate_at_line(raw, inline_max_chars)
    if "\n" in cut:
        header = (
            f"{att.name}（共 {total_lines} 行，"
            f"此处显示前 {shown_lines} 行 / {len(cut)} 字符）："
        )
        marker = (
            f"（已截断；可用 read_attachment 从第 {shown_lines + 1} 行继续读取）"
        )
    else:
        # 单行长文：按字符截断，没有行号可续
        header = (
            f"{att.name}（共 {len(raw)} 字符，此处显示前 {len(cut)} 字符）："
        )
        marker = "（已截断；如需后续内容请说明）"
    return f"{header}\n{cut}\n\n{marker}"


def compose_message(
    user_text: str,
    attachments: Sequence[Attachment],
    inline_max_chars: int = INLINE_MAX_CHARS,
) -> Tuple[str, List[ImageAttachment]]:
    """组装发给模型的一条用户消息。

    返回 ``(消息文本, 图片附件列表)``：文本里保留用户原话与引用块，
    正文统一追加在 ``---`` 之后；图片单独返回，由调用方按多模态发送。
    """
    images: List[ImageAttachment] = []
    blocks: List[str] = []
    seen: set[str] = set()

    for att in attachments:
        key = att.dedupe_key
        if key in seen:
            continue
        seen.add(key)
        if att.kind == KIND_IMAGE:
            if att.image is not None:
                images.append(att.image)
            continue
        blocks.append(_document_block(att, inline_max_chars))

    if not blocks:
        return user_text, images
    body = "\n---\n".join(blocks)
    return f"{user_text}\n\n---\n\n{body}", images


# ---------------- 正文落盘（长附件续读与旧会话恢复） ----------------


def store_attachment_text(att: Attachment, directory: str) -> str:
    """把附件正文写到 ``<directory>/<id>.txt``，返回落盘路径（图片或空正文返回空串）。"""
    if not att.text or not att.attachment_id:
        return att.stored_path
    directory = os.path.expanduser(directory)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{att.attachment_id}.txt")
    if not os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(att.text)
        os.replace(tmp, path)
    att.stored_path = path
    return path


def load_stored_text(att: Attachment) -> str:
    """读取附件落盘正文（内存里没有时用）。"""
    if att.text:
        return att.text
    if not att.stored_path:
        return ""
    try:
        with open(os.path.expanduser(att.stored_path), "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def collect_stale_attachments(directory: str, keep_ids: set) -> int:
    """删除目录里不再被任何会话引用的正文文件，返回删除数量。"""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return 0
    removed = 0
    for name in os.listdir(directory):
        if not name.endswith(".txt"):
            continue
        stem = name[: -len(".txt")]
        if stem in keep_ids:
            continue
        try:
            os.remove(os.path.join(directory, name))
            removed += 1
        except OSError:
            continue
    return removed
