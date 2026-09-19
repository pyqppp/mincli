"""定价与峰谷时段配置（~/.mincli/pricing.json）。

DeepSeek 价格会频繁调整，因此内置价格只是默认值，全部可通过配置文件覆盖：

    {
      "peak": {                          # 高峰时段判定（可省略）
        "days": [1, 2, 3, 4, 5],         # ISO 星期：周一=1 … 周日=7
        "ranges": [[9, 12], [14, 18]],   # [起始小时, 结束小时)
        "timezone_offset_hours": 8       # 北京时间为 8
      },
      "models": {                        # 按模型覆盖内置价；未列出的沿用内置
        "deepseek-flash": {"hit": [0.02, 0.04], "miss": [1.0, 2.0], "output": [4.0, 8.0]},
        "deepseek-v4-pro": {"miss": 4.5, "output": 13.5}   # 单数字 = 不分峰谷
      },
      "image_tokens": 1024               # 尺寸未知图片的估算兜底（默认=官方单图上限）
    }

单位为「元 / 百万 tokens」；价格字段支持 ``[空闲价, 高峰价]`` 或单个数字。
文件损坏/字段非法时静默回退默认值，不影响程序运行。
路径可用环境变量 ``MINCLI_PRICING_PATH`` 覆盖。
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Dict, Optional, Tuple

from mincli.config import (
    DEFAULT_PEAK_CONFIG,
    DEEPSEEK_PRICING,
    PRICING_PATH,
    VISION_IMAGE_TOKENS_DEFAULT,
    normalize_model_name,
)

_cache: Optional[dict] = None
_cache_mtime: Optional[float] = None
_cache_path: Optional[str] = None


def _to_pair(value, default: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """把价格值规范为 (空闲价, 高峰价)；单数字表示不分峰谷。非法返回 default。"""
    if isinstance(value, bool):
        return default
    try:
        if isinstance(value, (int, float)):
            v = float(value)
            return (v, v)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        pass
    return default


def _parse_peak(raw) -> dict:
    """解析高峰时段配置；非法字段回退默认值。"""
    peak = {
        "days": {int(d) for d in DEFAULT_PEAK_CONFIG["days"]},
        "ranges": [tuple(r) for r in DEFAULT_PEAK_CONFIG["ranges"]],
        "tz": float(DEFAULT_PEAK_CONFIG["timezone_offset_hours"]),
    }
    if not isinstance(raw, dict):
        return peak
    days = raw.get("days")
    if isinstance(days, list) and days:
        try:
            parsed = {int(d) for d in days}
            parsed = {d for d in parsed if 1 <= d <= 7}
            if parsed:
                peak["days"] = parsed
        except (TypeError, ValueError):
            pass
    ranges = raw.get("ranges")
    if isinstance(ranges, list):
        parsed_ranges = []
        for item in ranges:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    start, end = int(item[0]), int(item[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= start < end <= 24:
                    parsed_ranges.append((start, end))
        if parsed_ranges:
            peak["ranges"] = parsed_ranges
    tz = raw.get("timezone_offset_hours")
    if isinstance(tz, (int, float)) and not isinstance(tz, bool) and -12 <= tz <= 14:
        peak["tz"] = float(tz)
    return peak


def _merge_models(raw) -> Dict[str, dict]:
    """内置定价表 + 配置文件（按模型、按字段合并；未列出的沿用内置）。"""
    models: Dict[str, dict] = {name: dict(vals) for name, vals in DEEPSEEK_PRICING.items()}
    if not isinstance(raw, dict):
        return models
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        base = dict(models.get(name, {}))
        for field in ("hit", "miss", "output"):
            if field in entry:
                pair = _to_pair(entry[field], base.get(field))
                if pair is not None:
                    base[field] = pair
        if base:
            models[name] = base
    return models


def load_pricing(force: bool = False) -> dict:
    """读取并合并定价配置（带 mtime 缓存）。

    返回：{"models": {...}, "peak": {...}, "image_tokens": int,
           "path": str|None（实际生效的配置文件，未加载到则为 None）}
    """
    global _cache, _cache_mtime, _cache_path
    path = PRICING_PATH
    try:
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
    except OSError:
        mtime = None
    if (
        not force
        and _cache is not None
        and _cache_mtime == mtime
        and _cache_path == path
    ):
        return _cache

    data = None
    if mtime is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = None

    image_tokens = VISION_IMAGE_TOKENS_DEFAULT
    if isinstance(data, dict):
        value = data.get("image_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            image_tokens = int(value)

    _cache = {
        "models": _merge_models(data.get("models") if data else None),
        "peak": _parse_peak(data.get("peak") if data else None),
        "image_tokens": image_tokens,
        "path": path if data is not None else None,
    }
    _cache_mtime = mtime
    _cache_path = path
    return _cache


def model_pricing(model: str, pricing: Optional[dict] = None) -> Optional[dict]:
    """取某模型的有效价格 {hit, miss, output}（各自为 (空闲, 高峰) 二元组）。"""
    table = (pricing or load_pricing())["models"]
    if not table:
        return None
    normalized = normalize_model_name(model)
    return table.get(model) or table.get(normalized)


def is_peak_hour(now: Optional[datetime.datetime] = None,
                 pricing: Optional[dict] = None) -> bool:
    """当前是否处于高峰时段（按配置的星期集合、时段与时区判定）。"""
    peak = (pricing or load_pricing())["peak"]
    if now is None:
        tz = datetime.timezone(datetime.timedelta(hours=peak["tz"]))
        now = datetime.datetime.now(tz)
    if now.isoweekday() not in peak["days"]:
        return False
    hour = now.hour
    return any(start <= hour < end for start, end in peak["ranges"])


def estimate_input_price(
    model: str,
    tokens: int,
    hit_ratio: Optional[float] = None,
    peak: bool = False,
    pricing: Optional[dict] = None,
) -> Optional[float]:
    """估算输入价格（元）：命中部分按缓存命中单价、其余按未命中单价。

    未知模型或非法 token 数返回 None。
    """
    entry = model_pricing(model, pricing)
    if not entry or tokens <= 0:
        return None
    idx = 1 if peak else 0
    hit_price = entry["hit"][idx]
    miss_price = entry["miss"][idx]
    ratio = hit_ratio if hit_ratio is not None else 0.0
    ratio = max(0.0, min(1.0, ratio))
    avg_price = hit_price * ratio + miss_price * (1 - ratio)
    return tokens * avg_price / 1_000_000


def image_tokens_per_image(pricing: Optional[dict] = None) -> int:
    """尺寸未知时的图片估算兜底值（默认 1024=官方单图上限，可在 pricing.json 调整）。"""
    return int((pricing or load_pricing())["image_tokens"])
