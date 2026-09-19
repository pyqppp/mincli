"""多对话树持久化（每棵树一个文件 + 一个索引文件）。

目录布局（默认 `~/.mincli/trees`，可用 MINCLI_TREES_PATH 覆盖）：

    index.json        索引：编号清单、每树颜色、上次激活的树、全局设置
    <编号>.json       单棵树的全部数据

为什么一棵树一个文件：

- 单棵树文件损坏或写入失败不影响其它树；
- 新建 / 切换 / 删除只动小文件，不必重写全部历史；
- 树数量不设上限时，索引文件依然很小。

写入一律先写同目录临时文件再 `os.replace`，避免中途崩溃留下半个 JSON。
编号从 1 起单调递增、永不复用；颜色由编号固定推导（见 `color_of`），
因此删树不会让其它树变色。
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

# 默认目录：与 config.SAVE_BASE_DIR 同风格，支持环境变量覆盖
TREES_DIR = os.path.expanduser(
    os.getenv("MINCLI_TREES_PATH", "~/.mincli/trees")
)

INDEX_NAME = "index.json"
INDEX_VERSION = 1

# 一棵树的界面颜色数量：预定义 8 色，第 9 棵回到第 1 色（见 tui/theme.py）
TREE_COLOR_COUNT = 8


def color_of(number: int) -> int:
    """编号 → 颜色序号（1..TREE_COLOR_COUNT，循环复用）。"""
    try:
        n = int(number)
    except (TypeError, ValueError):
        return 1
    if n < 1:
        return 1
    return ((n - 1) % TREE_COLOR_COUNT) + 1


def _atomic_write_json(path: str, data: Any) -> bool:
    """原子写 JSON（同目录临时文件 + os.replace）；失败返回 False。"""
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".mincli-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception:
        return False


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class TreeStore:
    """对话树文件仓库：只管编号、颜色、文件读写，不关心对话内容结构。"""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = os.path.expanduser(base_dir) if base_dir else TREES_DIR
        self._index: Dict[str, Any] = self._load_index()
        self._prune_missing()

    # ---------------- 路径 ----------------

    @property
    def index_path(self) -> str:
        return os.path.join(self.base_dir, INDEX_NAME)

    def tree_path(self, number: int) -> str:
        return os.path.join(self.base_dir, f"{int(number)}.json")

    # ---------------- 索引 ----------------

    def _empty_index(self) -> Dict[str, Any]:
        return {
            "version": INDEX_VERSION,
            "next": 1,
            "active": None,
            "trees": {},       # "1": {"color": 1}
            "global": {},      # 全局设置（模型/温度/系统提示词…由 controller 解释）
        }

    def _load_index(self) -> Dict[str, Any]:
        data = _read_json(self.index_path)
        if not isinstance(data, dict):
            return self._empty_index()
        index = self._empty_index()
        index["version"] = int(data.get("version") or INDEX_VERSION)
        trees = data.get("trees")
        if isinstance(trees, dict):
            for key, meta in trees.items():
                try:
                    number = int(key)
                except (TypeError, ValueError):
                    continue
                if number < 1:
                    continue
                color = 1
                if isinstance(meta, dict):
                    try:
                        color = int(meta.get("color") or color_of(number))
                    except (TypeError, ValueError):
                        color = color_of(number)
                index["trees"][str(number)] = {"color": color}
        try:
            index["next"] = max(1, int(data.get("next") or 1))
        except (TypeError, ValueError):
            index["next"] = 1
        active = data.get("active")
        try:
            index["active"] = int(active) if active is not None else None
        except (TypeError, ValueError):
            index["active"] = None
        if isinstance(data.get("global"), dict):
            index["global"] = data["global"]
        # next 至少要大于已有最大编号，防止手工改坏索引后编号撞车
        if index["trees"]:
            index["next"] = max(index["next"], max(int(n) for n in index["trees"]) + 1)
        return index

    def _prune_missing(self) -> None:
        """索引里有、磁盘上没有的树 → 视为不存在（上次创建中途崩溃）。"""
        missing = [n for n in self._index["trees"] if not os.path.exists(self.tree_path(n))]
        if not missing:
            return
        for n in missing:
            del self._index["trees"][n]
        if self._index["active"] is not None and str(self._index["active"]) not in self._index["trees"]:
            self._index["active"] = None
        self.save_index()

    def save_index(self) -> bool:
        return _atomic_write_json(self.index_path, self._index)

    def reload(self) -> None:
        """重新从磁盘读取索引（/tree 列表或外部改动后刷新）。"""
        self._index = self._load_index()
        self._prune_missing()

    # ---------------- 全局设置 ----------------

    def global_settings(self) -> Dict[str, Any]:
        data = self._index.get("global")
        return dict(data) if isinstance(data, dict) else {}

    def set_global_settings(self, data: Dict[str, Any]) -> None:
        self._index["global"] = dict(data or {})
        self.save_index()

    # ---------------- 编号 / 颜色 / 激活 ----------------

    def numbers(self) -> List[int]:
        return sorted(int(n) for n in self._index["trees"])

    def has_trees(self) -> bool:
        return bool(self._index["trees"])

    def color_of(self, number: int) -> int:
        meta = self._index["trees"].get(str(int(number)))
        if isinstance(meta, dict):
            try:
                color = int(meta.get("color") or 0)
            except (TypeError, ValueError):
                color = 0
            if 1 <= color <= TREE_COLOR_COUNT:
                return color
        return color_of(number)

    def next_number(self) -> int:
        return int(self._index["next"])

    def allocate(self) -> int:
        """分配一个新编号并登记进索引（颜色按编号推导），返回编号。

        只登记编号，不创建树文件——调用方随后应立刻 `save_tree`；
        若中途失败，下次启动的 `_prune_missing` 会把空编号清掉。
        """
        number = int(self._index["next"])
        self._index["next"] = number + 1
        self._index["trees"][str(number)] = {"color": color_of(number)}
        self.save_index()
        return number

    def set_active(self, number: Optional[int]) -> None:
        self._index["active"] = int(number) if number is not None else None
        self.save_index()

    def active(self) -> Optional[int]:
        active = self._index["active"]
        if active is None:
            return None
        return int(active) if str(active) in self._index["trees"] else None

    # ---------------- 树文件 ----------------

    def has_tree(self, number: int) -> bool:
        return str(int(number)) in self._index["trees"]

    def load_tree(self, number: int) -> Optional[Dict[str, Any]]:
        data = _read_json(self.tree_path(number))
        return data if isinstance(data, dict) else None

    def save_tree(self, number: int, payload: Dict[str, Any]) -> bool:
        return _atomic_write_json(self.tree_path(number), payload)

    def delete_tree(self, number: int) -> bool:
        """删除树文件与索引条目（编号不复用）。"""
        number = int(number)
        existed = self.has_tree(number)
        try:
            os.remove(self.tree_path(number))
        except OSError:
            pass
        self._index["trees"].pop(str(number), None)
        if self._index["active"] == number:
            remaining = self.numbers()
            self._index["active"] = remaining[-1] if remaining else None
        self.save_index()
        return existed
