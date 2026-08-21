"""tools/images.py 单元测试（纯标准库，手工构造图片字节，不联网）。

运行：`venv/bin/python -m tests.test_images`
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli.tools.images import (
    ImageAttachment,
    build_image_block,
    collect_inline_bytes,
    encode_data_url,
    estimate_image_tokens,
    is_image_path,
    looks_like_image_target,
    make_path_attachment,
    make_url_attachment,
    read_dimensions,
    sniff_format,
)

PASS = 0
FAIL = 0

_TMP = tempfile.mkdtemp(prefix="mincli_imgtest_")


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ---------------- 手工构造最小合法图片字节 ----------------

def png_bytes(w: int, h: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + w.to_bytes(4, "big")
        + h.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def gif_bytes(w: int, h: int) -> bytes:
    return b"GIF89a" + w.to_bytes(2, "little") + h.to_bytes(2, "little") + b"\x00\x00\x00"


def jpeg_bytes(w: int, h: int) -> bytes:
    # 最小 SOF0 段：FF C0 + 段长(0x0011=17) + 精度(1) + 高(2) + 宽(2) + 分量数(1) + 3×3
    seg = (
        b"\xff\xc0\x00\x11\x08"
        + h.to_bytes(2, "big")
        + w.to_bytes(2, "big")
        + b"\x03"
        + b"\x01\x22\x00" * 3
    )
    return b"\xff\xd8" + seg + b"\xff\xd9"


def webp_vp8l_bytes(w: int, h: int) -> bytes:
    packed = ((w - 1) | ((h - 1) << 14)).to_bytes(4, "little")
    return b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x0a\x00\x00\x00" + b"\x2f" + packed


def webp_vp8x_bytes(w: int, h: int) -> bytes:
    return (
        b"RIFF\x00\x00\x00\x00WEBPVP8X"
        + b"\x0a\x00\x00\x00"
        + b"\x00"
        + b"\x00\x00\x00"
        + (w - 1).to_bytes(3, "little")
        + (h - 1).to_bytes(3, "little")
    )


def write(tmp_name: str, data: bytes) -> str:
    path = os.path.join(_TMP, tmp_name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def test_sniff_and_dimensions():
    print("== 格式嗅探 / 尺寸解析 ==")
    check("PNG 嗅探", sniff_format(png_bytes(100, 50)) == "png")
    check("GIF 嗅探", sniff_format(gif_bytes(100, 50)) == "gif")
    check("JPEG 嗅探", sniff_format(jpeg_bytes(100, 50)) == "jpeg")
    check("WebP VP8L 嗅探", sniff_format(webp_vp8l_bytes(100, 50)) == "webp")
    check("WebP VP8X 嗅探", sniff_format(webp_vp8x_bytes(100, 50)) == "webp")
    check("BMP 不支持", sniff_format(b"BM\x00\x00") is None)
    check("纯文本不支持", sniff_format(b"hello world") is None)

    check("PNG 尺寸", read_dimensions(png_bytes(640, 480)) == (640, 480))
    check("GIF 尺寸", read_dimensions(gif_bytes(320, 200)) == (320, 200))
    check("JPEG 尺寸", read_dimensions(jpeg_bytes(1024, 768)) == (1024, 768))
    check("WebP VP8L 尺寸", read_dimensions(webp_vp8l_bytes(100, 50)) == (100, 50))
    check("WebP VP8X 尺寸", read_dimensions(webp_vp8x_bytes(800, 600)) == (800, 600))
    check("无头数据返回 None", read_dimensions(b"") is None)


def test_encode():
    print("== base64 编码 ==")
    url = encode_data_url(b"\x89PNG", "png")
    check("data URL 前缀", url.startswith("data:image/png;base64,"))
    import base64
    check("内容可解码", base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG")


def test_estimate():
    print("== token 估算 ==")
    check("100x100 = 固定开销 117", estimate_image_tokens(100, 100) == 117)
    check("800x800 = 349", estimate_image_tokens(800, 800) == 349)
    check("超大图封顶 357", estimate_image_tokens(5000, 4000) == 357)
    check("low 大幅降低", estimate_image_tokens(1600, 1200, "low") == 144)
    check("无尺寸按上限", estimate_image_tokens(None, None) == 384)
    check("400x400 无附加额", estimate_image_tokens(400, 400) == 117)


def test_attachments():
    print("== 附件构造 ==")
    png = write("a.png", png_bytes(800, 600))
    att = make_path_attachment(png, detail="low")
    check("路径附件字段", att.name == "a.png" and att.size_bytes > 0)
    check("尺寸已解析", (att.width, att.height) == (800, 600))
    check("detail 生效", att.detail == "low")
    check("token 估算写入", att.tokens_est == 144)  # low: 512×384 → 117 + 27

    check("不存在的文件报错", "文件不存在" in str(_exc(make_path_attachment, "/nonexistent/x.png")))
    check("不支持的格式报错", "不支持" in str(_exc(make_path_attachment, write("b.txt", b"hello"))))

    # 超过 32MiB（稀疏文件快速构造）
    big = os.path.join(_TMP, "big.png")
    with open(big, "wb") as f:
        f.write(png_bytes(1, 1))
        f.truncate(33 * 1024 * 1024)
    err = _exc(make_path_attachment, big)
    check("超大文件报错", "32 MiB" in str(err))

    att_url = make_url_attachment("https://example.com/x.jpg?size=1", detail="auto")
    check("URL 附件", att_url.is_url and att_url.source.startswith("http"))
    check("URL 非 http 报错", "http" in str(_exc(make_url_attachment, "ftp://x/y.png")))
    long_url = "https://e.com/" + "a" * 9000
    check("URL 过长报错", "8192" in str(_exc(make_url_attachment, long_url)))


def _exc(fn, *args):
    try:
        fn(*args)
        return None
    except ValueError as e:
        return str(e)


def test_build_block():
    print("== 内容块构造 ==")
    png = write("c.png", png_bytes(100, 100))

    att = ImageAttachment(source=png, name="c.png", file_id="file-api-x")
    block = build_image_block(att)
    check("file_id → file 块", block == {"type": "file", "file_id": "file-api-x"})

    att2 = ImageAttachment(source="https://e.com/i.png", name="i.png", is_url=True)
    block2 = build_image_block(att2)
    check("URL → image_url 块", block2["type"] == "image_url"
          and block2["image_url"]["url"] == "https://e.com/i.png")

    att3 = ImageAttachment(source=png, name="c.png")
    block3 = build_image_block(att3)
    check("路径 → base64 内联", block3["type"] == "image_url"
          and block3["image_url"]["url"].startswith("data:image/png;base64,"))

    att4 = ImageAttachment(source="/gone/ghost.png", name="g.png")
    block4 = build_image_block(att4)
    check("文件已删 → 占位文本", block4["type"] == "text" and "文件已删除" in block4["text"])


def test_serialization():
    print("== ImageAttachment 序列化 ==")
    att = ImageAttachment(
        source="/tmp/x.png", detail="low", file_id="file-api-z",
        name="x.png", size_bytes=100, width=10, height=20, tokens_est=200,
    )
    d = att.to_dict()
    att2 = ImageAttachment.from_dict(d)
    check("往返一致", att2.source == att.source and att2.file_id == att.file_id
          and att2.width == 10 and att2.height == 20 and att2.tokens_est == 200)
    empty = ImageAttachment.from_dict({})
    check("空 dict 容错", empty.source == "" and empty.file_id is None)


def test_collect_inline():
    print("== 48MiB 预检统计 ==")
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "user", "content": [{"type": "file", "file_id": "file-api-x"}]},
        {"role": "user", "content": "纯文本"},
    ]
    check("仅统计 data: URL", collect_inline_bytes(messages) == len("data:image/png;base64,AAAA"))
    check("无 data 为 0", collect_inline_bytes([{"role": "user", "content": "x"}]) == 0)


def test_helpers():
    print("== 杂项 ==")
    png = write("d.png", png_bytes(64, 64))
    check("is_image_path 按内容", is_image_path(png))
    txt = write("e.txt", b"plain")
    check("文本非图片", not is_image_path(txt))
    check("URL 扩展名识别", looks_like_image_target("https://e.com/a.JPG?x=1"))
    check("非图片 URL 不识别", not looks_like_image_target("https://e.com/a.html"))


if __name__ == "__main__":
    test_sniff_and_dimensions()
    test_encode()
    test_estimate()
    test_attachments()
    test_build_block()
    test_serialization()
    test_collect_inline()
    test_helpers()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
