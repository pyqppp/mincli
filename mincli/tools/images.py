"""图片附件工具：格式嗅探、尺寸解析、base64 编码、token 估算。

全部使用标准库（无 PIL 依赖）。格式按文件内容（magic bytes）识别，
与 DeepSeek API 的行为一致（API 也按内容而非扩展名/声明 MIME 判断，
实测 JPEG 内容声明为 png 仍可正常识别，BMP 会被 400 拒绝）。

token 估算按官方预处理流程精确换算（见 estimate_image_tokens）：小图先放大到
约 544×544，大图缩到上限 1024 token（约 1300×1300），结果与官方文档给的
「2000×2000 与 5000×5000 同值」「单图上限 1024」完全吻合。尺寸未知（外链 URL）
时退回 pricing.json 的可配置值 image_tokens（默认 1024，即上限）。估算只用于
状态条与压缩前后统计，实际计费以接口返回的 usage 为准。
"""

from __future__ import annotations

import base64
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mincli.config import (
    VISION_DEFAULT_DETAIL,
    VISION_DOWNSAMPLE_RATIO,
    VISION_FILE_ID_IMAGE_MAX_BYTES,
    VISION_INLINE_IMAGE_MAX_BYTES,
    VISION_LOW_SCALE_SIDE,
    VISION_MAX_IMAGE_TOKENS,
    VISION_MIN_PIXELS,
    VISION_PATCH_SIZE,
    VISION_URL_MAX_CHARS,
)
from mincli.pricing import image_tokens_per_image

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


def _llm_grid(best_h: int, best_w: int) -> Tuple[int, int]:
    """对齐后的像素尺寸 → 视觉 token 网格（官方实现同名函数）。"""
    return (
        math.ceil((best_h // VISION_PATCH_SIZE) / VISION_DOWNSAMPLE_RATIO),
        math.ceil((best_w // VISION_PATCH_SIZE) / VISION_DOWNSAMPLE_RATIO),
    )


def _num_image_tokens(n_h: int, n_w: int) -> int:
    """官方实现：n_h 行，每行 n_w 个图像 token 加 1 个换行，外加首尾 2 个标记。"""
    return n_h * (n_w + 1) + 2


def _solve_resize_ratio(height: int, width: int, max_tokens: int) -> Tuple[int, int]:
    """token 超上限时，求长宽比不变的最大像素尺寸（官方实现同名函数）。

    返回 (best_height, best_width)，两者都是 patch 的整数倍。
    """
    cell = VISION_PATCH_SIZE * VISION_DOWNSAMPLE_RATIO
    r = height / width
    max_w = math.sqrt((max_tokens - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    if max_w < 1.0:      # 极窄：压成单列
        return (max_tokens - 2) // 2 * cell, cell
    if max_h < 1.0:      # 极宽：压成单行
        return cell, (max_tokens - 3) * cell
    beta = min(
        math.floor(max_w) * cell / width,
        math.floor(max_h) * cell / height,
    )
    return (
        math.floor(height * beta / VISION_PATCH_SIZE) * VISION_PATCH_SIZE,
        math.floor(width * beta / VISION_PATCH_SIZE) * VISION_PATCH_SIZE,
    )


def estimate_image_tokens(
    width: Optional[int], height: Optional[int], detail: str = VISION_DEFAULT_DETAIL
) -> int:
    """按官方预处理流程估算一张图片消耗的 token。

    流程（与 deepseek-ai/DeepSeek-V4.1-Flash 的 vision_config + vLLM 参考实现
    ``mm_preprocess.load_image`` 一致）：

    1. ``detail=low``：先把图片缩到单边不超过 512px；
    2. 总像素小于 544×544 的图片按长宽比放大到约 544×544；
    3. 宽高各自对齐到 patch(14) 的整数倍，再按下采样比 3 得到 token 网格；
    4. 单图 token = n_h*(n_w+1)+2，超过上限 1024 时缩小到刚好不超。

    尺寸未知（外链 URL 无法本地嗅探）时退回 ``image_tokens`` 配置值（默认 1024，
    即官方上限），属于保守估计。
    """
    if not width or not height:
        return image_tokens_per_image()
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return image_tokens_per_image()
    if detail == "low" and max(w, h) > VISION_LOW_SCALE_SIDE:
        ratio = VISION_LOW_SCALE_SIDE / max(w, h)
        w, h = max(1, int(w * ratio)), max(1, int(h * ratio))
    if w * h < VISION_MIN_PIXELS:
        ratio = (VISION_MIN_PIXELS / (w * h)) ** 0.5
        w, h = max(1, int(w * ratio)), max(1, int(h * ratio))
    best_w = math.ceil(w / VISION_PATCH_SIZE) * VISION_PATCH_SIZE
    best_h = math.ceil(h / VISION_PATCH_SIZE) * VISION_PATCH_SIZE
    n_h, n_w = _llm_grid(best_h, best_w)
    if _num_image_tokens(n_h, n_w) > VISION_MAX_IMAGE_TOKENS:
        best_h, best_w = _solve_resize_ratio(h, w, VISION_MAX_IMAGE_TOKENS)
        n_h, n_w = _llm_grid(best_h, best_w)
    return _num_image_tokens(n_h, n_w)


# ---------------- 附件构造 ----------------

def make_path_attachment(
    path: str, detail: str = VISION_DEFAULT_DETAIL
) -> ImageAttachment:
    """从本地图片文件构造附件；校验失败抛 ValueError（中文提示）。

    保存绝对路径（source=abspath）：会话持久化后即使从其他工作目录
    启动，历史图片路径仍可解析。
    """
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.exists(path):
        raise ValueError(f"文件不存在: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"不是文件: {path}")
    size = os.path.getsize(path)
    if size > VISION_FILE_ID_IMAGE_MAX_BYTES:
        raise ValueError(
            f"图片过大（{size / 1024 / 1024:.1f} MiB > "
            f"{VISION_FILE_ID_IMAGE_MAX_BYTES // 1024 // 1024} MiB 单图上限）"
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
        tokens_est=estimate_image_tokens(None, None, detail),
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
    if att.size_bytes and att.size_bytes > VISION_INLINE_IMAGE_MAX_BYTES:
        # 超过内联单图上限：仅 Files API 的 file_id 可承载（本地已尽力上传）
        return {
            "type": "text",
            "text": (
                f"[图片: {att.name}（{att.size_bytes / 1024 / 1024:.1f} MiB 超过 "
                f"{VISION_INLINE_IMAGE_MAX_BYTES // 1024 // 1024} MiB 内联上限，"
                "上传 Files API 失败，未发送）]"
            ),
        }
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


def oversize_inline_attachments(attachments: List[ImageAttachment]) -> List[str]:
    """返回会走内联路径（无 file_id、非外部 URL）且超过 32MiB 单图上限的图片名。

    这类图片只能通过 Files API 的 file_id 发送；若上传失败则无法发送。
    """
    return [
        att.name
        for att in attachments
        if not att.file_id
        and not att.is_url
        and att.size_bytes
        and att.size_bytes > VISION_INLINE_IMAGE_MAX_BYTES
    ]


def oversize_side_attachments(attachments: List[ImageAttachment], limit: int) -> List[str]:
    """返回单边超过 limit 像素的图片名（尺寸未知的跳过，交给 API 校验）。"""
    out: List[str] = []
    for att in attachments:
        if att.width and att.height and max(att.width, att.height) > limit:
            out.append(f"{att.name}（{att.width}x{att.height}）")
    return out


def total_tokens_est(attachments: List[ImageAttachment]) -> int:
    """一组附件的估算 token 合计（尺寸未知的按配置上限计）。"""
    return sum(
        att.tokens_est or estimate_image_tokens(att.width, att.height, att.detail)
        for att in attachments
    )


def split_low_inline(
    attachments: List[ImageAttachment], budget_bytes: int
) -> Tuple[List[ImageAttachment], List[ImageAttachment]]:
    """把「还没有 file_id 的本地图片」分成 (走内联, 走上传) 两组。

    ``file`` 内容块不支持 detail 字段（官方明确：通过 file_id 传图时 detail 被
    忽略），所以要让 ``detail=low`` 真正省 token，只能走内联的 ``image_url``。
    这里把 detail=low 的本地图片按体积升序放进内联预算，预算用完的（以及
    超过内联单图上限、本地文件缺失的）仍走上传，由调用方提示 detail 不生效。
    """
    inline: List[ImageAttachment] = []
    upload: List[ImageAttachment] = []
    candidates: List[ImageAttachment] = []
    budget = max(0, int(budget_bytes))
    for att in attachments:
        if att.file_id or att.is_url:
            continue
        if (
            att.detail != "low"
            or not att.size_bytes
            or att.size_bytes > VISION_INLINE_IMAGE_MAX_BYTES
        ):
            upload.append(att)
            continue
        candidates.append(att)
    for att in sorted(candidates, key=lambda a: a.size_bytes):
        if att.size_bytes <= budget:
            budget -= att.size_bytes
            inline.append(att)
        else:
            upload.append(att)
    return inline, upload


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
