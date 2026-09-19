"""Files API 客户端：图片文件上传 / 列表 / 查询 / 删除。

配合 deepseek-flash（图片理解）使用：图片上传一次后通过 file_id 引用，
多个请求复用同一张图片无需重复上传（请求体极小、序列化稳定，不破坏前缀缓存）。
限制：单文件 ≤64MiB、purpose 必须为 user_data、默认永久有效（不传 expires_after）、
单用户 ≤10000 个文件 / 25GiB。

列表接口的官方默认是 ``order=asc``（最旧在前）且单页 1000 条，这里统一改成
``desc``（最新在前）并透出 ``has_more``，避免「新传的图片反而看不到、超过一页
静默截断」。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from mincli.config import FILES_LIST_PAGE, FILES_LIST_PAGE_MAX


class FilesAPIError(RuntimeError):
    """Files API 操作失败（错误信息为中文，直接展示给用户）。"""


def _files_of(client):
    """取出 client.files；不可用时抛 FilesAPIError。"""
    files = getattr(client, "files", None)
    if files is None:
        raise FilesAPIError("Files API 不可用（当前客户端不支持）")
    return files


def _file_dict(f) -> Dict:
    """把 SDK 的文件对象归一化为普通 dict（字段缺失一律兜底）。"""
    return {
        "id": getattr(f, "id", "") or "",
        "name": getattr(f, "filename", "") or "",
        "bytes": getattr(f, "bytes", 0) or 0,
        "created_at": getattr(f, "created_at", 0) or 0,
        "expires_at": getattr(f, "expires_at", None),
    }


def upload_image(client, path: str, purpose: str = "user_data") -> str:
    """上传本地图片，返回 file_id（形如 file-api-...）。"""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FilesAPIError(f"文件不存在: {path}")
    files = _files_of(client)
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


def list_files(
    client,
    limit: int = FILES_LIST_PAGE,
    order: str = "desc",
    after: Optional[str] = None,
) -> Dict:
    """列出一页已上传文件（默认最新在前）。

    返回 ``{"items": [...], "has_more": bool}``；has_more 为 True 说明还有更旧的
    文件没有返回（官方列表默认 asc + 单页 1000 条，老实现会静默截断）。
    """
    files = _files_of(client)
    limit = max(1, min(int(limit or FILES_LIST_PAGE), FILES_LIST_PAGE_MAX))
    kwargs = {"limit": limit, "order": order if order in ("asc", "desc") else "desc"}
    if after:
        kwargs["after"] = after
    try:
        resp = files.list(**kwargs)
        items = getattr(resp, "data", None) or []
        return {
            "items": [_file_dict(f) for f in items],
            "has_more": bool(getattr(resp, "has_more", False)),
        }
    except FilesAPIError:
        raise
    except Exception as e:
        raise FilesAPIError(f"文件列表获取失败: {e}") from e


def list_all_files(client, page_size: int = 200, max_pages: int = 60) -> List[Dict]:
    """用 after 游标翻完所有页（/files clean 需要全量清单）。

    max_pages 兜底：配额 10000 个文件，按 200/页最多 50 页，留出余量。
    """
    out: List[Dict] = []
    after: Optional[str] = None
    for _ in range(max(1, max_pages)):
        page = list_files(client, limit=page_size, order="asc", after=after)
        items = page["items"]
        out.extend(items)
        if not page["has_more"] or not items:
            break
        after = items[-1]["id"]
    return out


def retrieve_file(client, file_id: str) -> Dict:
    """查询单个文件信息（GET /files/{id}）。"""
    file_id = (file_id or "").strip()
    if not file_id:
        raise FilesAPIError("缺少 file_id")
    files = _files_of(client)
    try:
        return _file_dict(files.retrieve(file_id))
    except FilesAPIError:
        raise
    except Exception as e:
        raise FilesAPIError(f"文件信息查询失败: {e}") from e


def delete_file(client, file_id: str) -> bool:
    """删除一个已上传的文件；失败抛 FilesAPIError。"""
    files = _files_of(client)
    try:
        files.delete(file_id)
        return True
    except FilesAPIError:
        raise
    except Exception as e:
        raise FilesAPIError(f"文件删除失败: {e}") from e
