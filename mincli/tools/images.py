"""图片附件工具：格式嗅探、尺寸解析、base64 编码、token 估算。

全部使用标准库（无 PIL 依赖）。格式按文件内容（magic bytes）识别，
与 DeepSeek API 的行为一致（API 也按内容而非扩展名/声明 MIME 判断，
实测 JPEG 内容声明为 png 仍可正常识别，BMP 会被 400 拒绝）。

token 估算基于真实 API 实测校准（见 config.VISION_SIZE_EXTRA_ANCHORS 注释）：
每图固定开销 117 token，尺寸附加额按面积线性插值、封顶 240，总计封顶 384；
仅用于 /compact 前后统计与 usage 缺失时的兜底估算，实际计费以接口 usage 为准。
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mincli.config import (
    VISION_BASE_IMAGE_TOKENS,
    VISION_DEFAULT_DETAIL,
    VISION_IMAGE_MAX_BYTES,
    VISION_IMAGE_TOKEN_CAP,
    VISION_SIZE_EXTRA_ANCHORS,
    VISION_SIZE_EXTRA_CAP,
    VISION_URL_MAX_CHARS,
)

# 图片格式支持：API 按内容识别，这里本地先行校验（避免 400）
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# JPEG SOF0-SOF15 标记（帧起始，含宽高；C4/C8/CC 为其他用途）
_JPEG_SOF_MARKERS = frozenset({
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
})


@dataclass
class ImageAttachment:
    """一张待发送/已发送的图片。

    只保存来源（路径或 URL）与元数据，**不保存 base64**（会话文件不膨胀）。
    file_id 为 Files API 上传后的引用：有则用 file 内容块（请求体极小、
    序列化稳定、不破坏前缀缓存），无则发送时回退 base64 内联。
    """

    source: str                  # 本地路径或 http(s) URL
    detail: str = VISION_DEFAULT_DETAIL
    file_id: Optional[str] = None  # Files API 上传后的 file-api-...
    name: str = ""               # 文件名（显示用）
    is_url: bool = False
    size_bytes: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    tokens_est: int = 0          # 估算 token（发送前预估值，实际以 usage 为准）

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "detail": self.detail,
            "file_id": self.file_id,
            "name": self.name,
            "is_url": self.is_url,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "tokens_est": self.tokens_est,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageAttachment":
        return cls(
            source=data.get("source", ""),
            detail=data.get("detail", VISION_DEFAULT_DETAIL),
            file_id=data.get("file_id"),
            name=data.get("name", ""),
            is_url=data.get("is_url", False),
            size_bytes=data.get("size_bytes", 0),
            width=data.get("width"),
            height=data.get("height"),
            tokens_est=data.get("tokens_est", 0),
        )


# ---------------- 格式嗅探 / 尺寸解析 ----------------

def sniff_format(data: bytes) -> Optional[str]:
    """按文件内容（magic bytes）识别图片格式；不支持返回 None。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def read_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """从文件头解析像素尺寸（宽, 高）；解析失败返回 None。"""
    fmt = sniff_format(data)
    if fmt == "png":
        if len(data) >= 24:
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            if w and h:
                return (w, h)
    elif fmt == "gif":
        if len(data) >= 10:
            w = int.from_bytes(data[6:8], "little")
            h = int.from_bytes(data[8:10], "little")
            if w and h:
                return (w, h)
    elif fmt == "jpeg":
        # 遍历段直到 SOF 标记（JPEG 的宽高在 SOF 段内）
        i, n = 2, len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            if marker in _JPEG_SOF_MARKERS:
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                if w and h:
                    return (w, h)
            i += 2 + seg_len
    elif fmt == "webp":
        chunk_type = data[12:16]
        if chunk_type == b"VP8X" and len(data) >= 30:
            # canvas 尺寸：24-26（宽-1，LE 3 字节）、27-29（高-1）
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return (w, h)
        if chunk_type == b"VP8L" and len(data) >= 25:
            # 签名 0x2F 之后 4 字节：宽 14 位、高 14 位
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            w = 1 + (((b1 & 0x3F) << 8) | b0)
            h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
            return (w, h)
        if chunk_type == b"VP8 " and len(data) >= 30:
            # 帧标记 3 字节 + 起始码 3 字节，随后 2 字节宽、2 字节高（14 位）
            w = 1 + (int.from_bytes(data[26:28], "little") & 0x3FFF)
            h = 1 + (int.from_bytes(data[28:30], "little") & 0x3FFF)
            return (w, h)
    return None


def is_image_path(path: str) -> bool:
    """按内容判断本地文件是否为受支持的图片。"""
    try:
        with open(os.path.expanduser(path), "rb") as f:
            head = f.read(16)
        return sniff_format(head) is not None
    except OSError:
        return False


def looks_like_image_target(target: str) -> bool:
    """判断路径/URL 是否看起来是图片（扩展名快速判断，URL 无法嗅探内容）。"""
    path_part = target.split("?", 1)[0].lower()
    return path_part.endswith(tuple(_IMAGE_EXTENSIONS))


# ---------------- 编码 / token 估算 ----------------

def encode_data_url(data: bytes, fmt: str) -> str:
    """把图片字节编码为 data: URL（base64 内联）。"""
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


def _interp_extra(area: float) -> float:
    """按实测锚点表线性插值出尺寸附加 token；封顶 VISION_SIZE_EXTRA_CAP。"""
    prev_x, prev_y = 0.0, 0.0
    for x, y in VISION_SIZE_EXTRA_ANCHORS:
        if area <= x:
            if x == prev_x:
                return prev_y
            return prev_y + (y - prev_y) * (area - prev_x) / (x - prev_x)
        prev_x, prev_y = x, y
    return min(prev_y, float(VISION_SIZE_EXTRA_CAP))


def estimate_image_tokens(
    width: Optional[int], height: Optional[int], detail: str = VISION_DEFAULT_DETAIL
) -> int:
    """估算一张图片消耗的 token（近似；实际以接口 usage 为准）。"""
    if not width or not height:
        return VISION_IMAGE_TOKEN_CAP
    w, h = float(width), float(height)
    if detail == "low":
        # 官方：low 缩放至 512×512（保持长宽比）
        scale = min(512.0 / w, 512.0 / h, 1.0)
        w, h = w * scale, h * scale
    area = w * h
    total = VISION_BASE_IMAGE_TOKENS + int(round(_interp_extra(area)))
    return min(VISION_IMAGE_TOKEN_CAP, total)


# ---------------- 附件构造 ----------------

def make_path_attachment(
    path: str, detail: str = VISION_DEFAULT_DETAIL
) -> ImageAttachment:
    """从本地图片文件构造附件；校验失败抛 ValueError（中文提示）。"""
    path = os.path.expanduser(path.strip())
    if not os.path.exists(path):
        raise ValueError(f"文件不存在: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"不是文件: {path}")
    size = os.path.getsize(path)
    if size > VISION_IMAGE_MAX_BYTES:
        raise ValueError(
            f"图片过大（{size / 1024 / 1024:.1f} MiB > 32 MiB 内联上限）"
        )
    with open(path, "rb") as f:
        data = f.read()
    fmt = sniff_format(data)
    if fmt is None:
        raise ValueError(f"不支持的图片格式（仅支持 JPEG/PNG/GIF/WebP）: {path}")
    dims = read_dimensions(data)
    width, height = dims if dims else (None, None)
    return ImageAttachment(
        source=path,
        detail=detail,
        name=os.path.basename(path),
        size_bytes=size,
        width=width,
        height=height,
        tokens_est=estimate_image_tokens(width, height, detail),
    )


def make_url_attachment(
    url: str, detail: str = VISION_DEFAULT_DETAIL
) -> ImageAttachment:
    """从外部图片 URL 构造附件（API 下载；本地无法嗅探内容）。"""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("图片 URL 必须以 http:// 或 https:// 开头")
    if len(url) > VISION_URL_MAX_CHARS:
        raise ValueError(f"图片 URL 过长（{len(url)} > 8192 字符）")
    name = url.split("?")[0].rsplit("/", 1)[-1] or url
    return ImageAttachment(
        source=url,
        detail=detail,
        name=name[:120],
        is_url=True,
        tokens_est=VISION_IMAGE_TOKEN_CAP,
    )


def build_image_block(att: ImageAttachment) -> dict:
    """构造 OpenAI 兼容内容块。

    优先级：Files API file_id → 外部 URL → base64 内联 → 文本占位（降级）。
    """
    if att.file_id:
        return {"type": "file", "file_id": att.file_id}
    if att.is_url:
        return {
            "type": "image_url",
            "image_url": {"url": att.source, "detail": att.detail},
        }
    # 本地路径：读取并 base64 内联（上传失败/未上传时的回退路径）
    path = os.path.expanduser(att.source)
    try:
        with open(path, "rb") as f:
            data = f.read()
        fmt = sniff_format(data)
        if fmt is None:
            return {"type": "text", "text": f"[图片: {att.name}（格式不受支持）]"}
        return {
            "type": "image_url",
            "image_url": {
                "url": encode_data_url(data, fmt),
                "detail": att.detail,
            },
        }
    except OSError:
        return {"type": "text", "text": f"[图片: {att.name}（文件已删除，未发送）]"}


def image_placeholder_text(att: ImageAttachment) -> str:
    """附件在聊天区/导出/压缩源中的文本占位（WxH 已知时附带尺寸）。"""
    if att.width and att.height:
        return f"[图片: {att.name} ({att.width}x{att.height})]"
    return f"[图片: {att.name}]"


def collect_inline_bytes(messages: List[Dict]) -> int:
    """统计消息列表中所有 base64 内联图片的字节总量（48MiB 预检用）。"""
    total = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            url = (block.get("image_url") or {}).get("url") or ""
            if url.startswith("data:"):
                total += len(url)
    return total
