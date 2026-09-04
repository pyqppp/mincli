"""工作流（/wf）数据模型与持久化存储 —— 纯逻辑，无 UI 依赖。

工作流 = 一份可复用的“操作规范文档”（Markdown 文本）+ 元信息，与会话分开
长期保存在 ~/.mincli/workflows.json。

规范文档格式（模型提炼产出 / 落盘 / 编辑器修改 / 运行时注入共用同一文本）：

    目标：<一句话：这类任务要完成什么>

    变量：
    - {旧版本}：起始版本标签（示例：v1.0）
    - {新版本}：目标版本标签（示例：v2.0）

    步骤：
    1. <本步目的与动作>；若需命令：`<命令，动态部分用 {旧版本}>`
    2. …

每次执行会变化的具体数据（版本号、日期、路径、本次正文等）一律不写入文档，
改写为 {占位符}（变量名仅限 [A-Za-z_][A-Za-z0-9_]*），或交给“用户本次输入”
确定。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
STEP_RE = re.compile(r"^\s*\d+[.、]")

# 未能自动提炼时存入文档首行的标记，提示用户用 /wf edit 修正
FALLBACK_MARK = "<!-- 未能自动提炼，请用 /wf edit 名 修改说明 修正 -->"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def valid_name(name: str) -> bool:
    return bool(name) and bool(NAME_RE.match(name))


def doc_placeholders(doc: str) -> List[str]:
    """扫描文档中全部 {占位符} 变量名，保持出现顺序、去重。"""
    seen: List[str] = []
    for m in VAR_RE.finditer(doc or ""):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def substitute(doc: str, values: Dict[str, str]) -> tuple[str, List[str]]:
    """按 values 替换 {键}；未提供值的占位符原样保留并列入 missing。"""
    out = doc or ""
    for key, val in (values or {}).items():
        pat = re.compile(r"\{" + re.escape(str(key)) + r"\}")
        out = pat.sub(str(val), out)
    missing = [ph for ph in doc_placeholders(doc) if ph not in (values or {})]
    return out, missing


def doc_summary(doc: str) -> tuple[str, int]:
    """返回 (目标摘要, 步骤数)；解析失败时兜底。"""
    goal = ""
    for line in (doc or "").splitlines():
        m = re.match(r"^目标\s*[:：]\s*(.+)$", line.strip())
        if m:
            goal = m.group(1).strip()
            break
    if len(goal) > 60:
        goal = goal[:60] + "…"
    steps = sum(1 for ln in (doc or "").splitlines() if STEP_RE.match(ln))
    return goal, steps


def write_doc_tempfile(doc: str) -> str:
    """把规范文档写入临时 .md 文件（供系统编辑器打开），返回路径。"""
    fd, path = tempfile.mkstemp(prefix="mincli_wf_", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(doc or "")
    return path


@dataclass
class Workflow:
    name: str
    doc: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_nodes: List[str] = field(default_factory=list)
    run_count: int = 0
    last_run_at: Optional[str] = None

    def placeholders(self) -> List[str]:
        return doc_placeholders(self.doc)

    def goal(self) -> str:
        return doc_summary(self.doc)[0]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "doc": self.doc,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_nodes": list(self.source_nodes),
            "run_count": self.run_count,
            "last_run_at": self.last_run_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Workflow"]:
        if not isinstance(data, dict) or not valid_name(str(data.get("name", ""))):
            return None
        return cls(
            name=data["name"],
            doc=str(data.get("doc", "") or ""),
            created_at=str(data.get("created_at", "") or ""),
            updated_at=str(data.get("updated_at", "") or ""),
            source_nodes=list(data.get("source_nodes", []) or []),
            run_count=int(data.get("run_count", 0) or 0),
            last_run_at=data.get("last_run_at"),
        )


class WorkflowStore:
    """~/.mincli/workflows.json 读写；原子写入 + 损坏兜底。"""

    def __init__(self, path: Optional[str] = None):
        from mincli.config import WORKFLOWS_PATH

        self.path = path or WORKFLOWS_PATH
        self._data: Dict[str, dict] = {}
        self._loaded = False

    # ---------------- 读取 ----------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            wfs = data.get("workflows") if isinstance(data, dict) else None
            if isinstance(wfs, dict):
                self._data = {k: v for k, v in wfs.items() if isinstance(v, dict)}
        except Exception:
            # 损坏文件改名保留，从空开始（同 session 文件兜底策略）
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass
            self._data = {}

    def list(self) -> List[Workflow]:
        self._ensure_loaded()
        out: List[Workflow] = []
        for name in sorted(self._data):
            wf = Workflow.from_dict(dict(self._data[name], name=name))
            if wf:
                out.append(wf)
        return out

    def get(self, name: str) -> Optional[Workflow]:
        self._ensure_loaded()
        data = self._data.get(name)
        if not data:
            return None
        return Workflow.from_dict(dict(data, name=name))

    def has(self, name: str) -> bool:
        self._ensure_loaded()
        return name in self._data

    # ---------------- 写入 ----------------

    def _flush(self) -> None:
        self._ensure_loaded()
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        payload = {"version": 1, "workflows": self._data}
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix=".workflows_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    def put(self, wf: Workflow, overwrite: bool = False) -> bool:
        """保存/覆盖工作流；同名且非 overwrite 返回 False。"""
        self._ensure_loaded()
        if wf.name in self._data and not overwrite:
            return False
        if not wf.created_at:
            wf.created_at = now_iso()
        wf.updated_at = now_iso()
        self._data[wf.name] = wf.to_dict()
        self._flush()
        return True

    def delete(self, name: str) -> bool:
        self._ensure_loaded()
        if name not in self._data:
            return False
        del self._data[name]
        self._flush()
        return True

    def rename(self, old: str, new: str) -> Optional[str]:
        """重命名；成功返回 None，失败返回错误信息。"""
        self._ensure_loaded()
        if old not in self._data:
            return f"工作流「{old}」不存在"
        if not valid_name(new):
            return "新名称需为 1-32 位字母/数字/_/-"
        if new in self._data:
            return f"工作流「{new}」已存在"
        data = dict(self._data.pop(old), name=new)
        data["updated_at"] = now_iso()
        self._data[new] = data
        self._flush()
        return None
