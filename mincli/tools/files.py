"""Files API 客户端：图片文件上传 / 列表 / 删除。

配合 deepseek-v4-flash-vision-exp 使用：图片上传一次后通过 file_id 引用，
多个请求复用同一张图片无需重复上传（请求体极小、序列化稳定，不破坏前缀缓存）。
限制：单文件 ≤64MiB、purpose 必须为 user_data、默认永久有效（不传 expires_after）。
"""

from __future__ import annotations

import os
from typing import Dict, List


class FilesAPIError(RuntimeError):
    """Files API 操作失败（错误信息为中文，直接展示给用户）。"""


def upload_image(client, path: str, purpose: str = "user_data") -> str:
    """上传本地图片，返回 file_id（形如 file-api-...）。"""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FilesAPIError(f"文件不存在: {path}")
    files = getattr(client, "files", None)
    if files is None:
        raise FilesAPIError("Files API 不可用（当前客户端不支持）")
    try:
        with open(path, "rb") as f:
            resp = files.create(file=f, purpose=purpose)
        file_id = getattr(resp, "id", None)
        if not file_id:
            raise FilesAPIError("上传响应缺少 file_id")
        return file_id
    except FilesAPIError:
        raise
    except Exception as e:
        raise FilesAPIError(f"图片上传失败: {e}") from e


def list_files(client) -> List[Dict]:
    """列出已上传的图片文件（分页第 1 页，按创建时间）。"""
    files = getattr(client, "files", None)
    if files is None:
        raise FilesAPIError("Files API 不可用（当前客户端不支持）")
    try:
        resp = files.list()
        items = getattr(resp, "data", None)
        if not items:
            return []
        out = []
        for f in items:
            out.append({
                "id": getattr(f, "id", ""),
                "name": getattr(f, "filename", ""),
                "bytes": getattr(f, "bytes", 0),
                "created_at": getattr(f, "created_at", 0),
                "expires_at": getattr(f, "expires_at", None),
            })
        return out
    except FilesAPIError:
        raise
    except Exception as e:
        raise FilesAPIError(f"文件列表获取失败: {e}") from e


def delete_file(client, file_id: str) -> bool:
    """删除一个已上传的文件；失败抛 FilesAPIError。"""
    files = getattr(client, "files", None)
    if files is None:
        raise FilesAPIError("Files API 不可用（当前客户端不支持）")
    try:
        files.delete(file_id)
        return True
    except FilesAPIError:
        raise
    except Exception as e:
        raise FilesAPIError(f"文件删除失败: {e}") from e
