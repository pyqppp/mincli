"""对话树存储层测试（mincli/trees.py，不联网）。

运行：`python3 -m tests.test_trees`
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli.trees import TREE_COLOR_COUNT, TreeStore, color_of

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def fresh_dir() -> str:
    path = tempfile.mkdtemp(prefix="mincli_trees_test_")
    return path


def test_color_mapping():
    print("== 编号 → 颜色 ==")
    check("1 号是青色", color_of(1) == 1)
    check("8 号是第 8 色", color_of(8) == TREE_COLOR_COUNT)
    check("9 号循环回第 1 色", color_of(9) == 1)
    check("17 号循环回第 1 色", color_of(17) == 1)
    check("非法编号退回第 1 色", color_of("x") == 1 and color_of(0) == 1)


def test_allocate_and_numbers():
    print("== 编号分配 ==")
    base = fresh_dir()
    try:
        store = TreeStore(base)
        check("初始没有任何树", store.numbers() == [] and not store.has_trees())
        check("初始 next 为 1", store.next_number() == 1)

        n1 = store.allocate()
        check("首个编号是 1", n1 == 1)
        store.save_tree(n1, {"number": n1})
        n2 = store.allocate()
        store.save_tree(n2, {"number": n2})
        check("第二个编号是 2", n2 == 2)
        check("编号列表有序", store.numbers() == [1, 2])

        store.delete_tree(2)
        check("删除后只剩 1", store.numbers() == [1])
        check("树文件已删除", not os.path.exists(store.tree_path(2)))
        n3 = store.allocate()
        check("编号不复用（删了 2 号，新建是 3 号）", n3 == 3)
        check("颜色按编号推导", store.color_of(3) == 3)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_tree_roundtrip():
    print("== 树文件读写 ==")
    base = fresh_dir()
    try:
        store = TreeStore(base)
        number = store.allocate()
        payload = {
            "number": number,
            "color": store.color_of(number),
            "tree": {"nodes": {}, "root_id": None},
            "draft": "草稿内容",
            "audit_level": 3,
            "file_confirm": False,
            "workspace": "/tmp/x",
            "system_tools": False,
            "mcp_tools": ["tavily_search"],
        }
        check("写入树文件成功", store.save_tree(number, payload))
        loaded = store.load_tree(number)
        check("读回内容一致", loaded == payload)
        check("新建的 store 也能读到", TreeStore(base).load_tree(number) == payload)
        check("没有残留临时文件",
              not [p for p in os.listdir(base) if p.endswith(".tmp")])

        # 不存在的树
        check("不存在的树返回 None", store.load_tree(999) is None)
        check("删除不存在的树不报错", store.delete_tree(999) is False)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_active_tracking():
    print("== 激活的树 ==")
    base = fresh_dir()
    try:
        store = TreeStore(base)
        check("初始没有激活树", store.active() is None)
        n1 = store.allocate()
        store.save_tree(n1, {"number": n1})
        n2 = store.allocate()
        store.save_tree(n2, {"number": n2})
        store.set_active(n2)
        check("激活编号可持久化", TreeStore(base).active() == n2)

        store.delete_tree(n2)
        check("删除激活树后落到剩余树", store.active() == n1)
        store.delete_tree(n1)
        check("全删完后没有激活树", store.active() is None)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_global_settings():
    print("== 全局设置 ==")
    base = fresh_dir()
    try:
        store = TreeStore(base)
        check("初始全局设置为空", store.global_settings() == {})
        store.set_global_settings({"model": "deepseek-flash", "temperature": 1.0})
        again = TreeStore(base)
        check("全局设置可持久化",
              again.global_settings().get("model") == "deepseek-flash")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_corrupt_index():
    print("== 索引损坏容错 ==")
    base = fresh_dir()
    try:
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "index.json"), "w", encoding="utf-8") as f:
            f.write("{ 这不是合法 JSON")
        store = TreeStore(base)
        check("损坏索引退化为空", store.numbers() == [] and store.active() is None)
        n = store.allocate()
        check("损坏后仍可新建树", n == 1)

        # 索引里登记但文件缺失 → 视为不存在
        with open(os.path.join(base, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"version": 1, "next": 5, "active": 4, "trees": {"4": {"color": 4}}}, f)
        store2 = TreeStore(base)
        check("缺失文件的编号被清理", store2.numbers() == [] and store2.active() is None)
        check("next 至少大于最大已用编号", store2.next_number() >= 5)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_color_persisted():
    print("== 颜色随编号固定 ==")
    base = fresh_dir()
    try:
        store = TreeStore(base)
        n = store.allocate()
        store.save_tree(n, {"number": n})
        check("9 号树颜色循环到 1", store.color_of(9) == 1)
        check("3 号树颜色是 3", store.color_of(3) == 3)
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    test_color_mapping()
    test_allocate_and_numbers()
    test_tree_roundtrip()
    test_active_tracking()
    test_global_settings()
    test_corrupt_index()
    test_color_persisted()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
