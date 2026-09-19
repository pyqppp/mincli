"""新建 / 编辑对话树能力的向导弹窗。

能力分三组：

- 对话：必选（锁定的复选框，仅表意）；
- 系统工具：整组一个开关（内置 server 的文件/命令/网页工具 + 进程内树查询工具）；
- 外置 MCP 工具：按 server 分组、逐个工具勾选（新建时默认一个都不勾）。

MCP 还在后台连接时，外置区先显示「正在连接 MCP…」占位，连完后由 App 调用
`refresh_tools()` 填充（不阻塞建树）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from mincli.mcp_client import BUNDLED_NAME


class TreeWizardScreen(ModalScreen[Optional[dict]]):
    """勾选能力并创建（或修改）一棵对话树。

    关闭时返回 ``{"system_tools": bool, "mcp_tools": [工具名...]}``；
    取消返回 ``None``。
    """

    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def __init__(
        self,
        heading: str = "新建对话树",
        note: str = "",
        groups: Optional[Dict[str, List[str]]] = None,
        servers: Optional[List[str]] = None,
        initial: Optional[dict] = None,
        mcp_ready: bool = True,
    ) -> None:
        super().__init__()
        self.heading = heading
        self.note = note
        self._groups = dict(groups or {})
        self._servers = list(servers or [])
        self._mcp_ready = mcp_ready
        initial = initial or {}
        self._system_tools = bool(initial.get("system_tools", True))
        self._checked = set(initial.get("mcp_tools") or [])
        # 工具名 → 复选框（刷新后重建）
        self._boxes: Dict[str, Checkbox] = {}

    # ---------------- 布局 ----------------

    def compose(self) -> ComposeResult:
        with Vertical(id="wizard-box"):
            yield Static(self.heading, id="wizard-title")
            if self.note:
                yield Static(self.note, id="wizard-note", markup=False)
            with VerticalScroll(id="wizard-body"):
                yield Static("对话（必选）", classes="wizard-group")
                yield Checkbox(
                    "AI 对话本身（始终启用）",
                    value=True,
                    id="cap-chat",
                    disabled=True,
                )
                yield Static("系统工具", classes="wizard-group")
                yield Checkbox(
                    "读写文件、列目录、执行命令、抓取网页，以及查询对话树的两个内置工具",
                    value=self._system_tools,
                    id="cap-system",
                )
                yield Static("外置 MCP 工具", classes="wizard-group")
                yield Static("", id="wizard-hint", markup=False)
                yield Vertical(id="wizard-ext")
            with Horizontal(id="wizard-buttons"):
                yield Button("确定", variant="primary", id="wizard-ok")
                yield Button("取消", id="wizard-cancel")

    def on_mount(self) -> None:
        self._render_ext()
        try:
            self.query_one("#wizard-ok", Button).focus()
        except Exception:
            pass

    def _render_ext(self) -> None:
        """按 server 分组列出可选的外置工具（只增不减，避免异步卸载）。

        刚开始 MCP 可能还没连完，此时显示占位提示；连完后由 App 调用
        refresh_tools() 再挂上复选框。
        """
        hint = self.query_one("#wizard-hint", Static)
        container = self.query_one("#wizard-ext", Vertical)
        external = {
            name: tools
            for name, tools in self._groups.items()
            if name != BUNDLED_NAME and tools
        }
        if not external:
            if self._mcp_ready:
                hint.update("（没有可用的外置 MCP 工具；可用 /mcp add 添加 server）")
            else:
                hint.update(
                    "正在连接 MCP…（连完后工具列表会自动出现；也可以先建树，"
                    "之后用 /tree N tools 补）"
                )
            return
        hint.update("")
        order = list(self._servers) or list(external)
        for name in order:
            tools = [t for t in external.get(name, []) if t not in self._boxes]
            if not tools:
                continue
            if not self.query(f"#wizard-server-{self._safe_id(name)}"):
                container.mount(
                    Static(
                        name,
                        classes="wizard-server",
                        id=f"wizard-server-{self._safe_id(name)}",
                        markup=False,
                    )
                )
            for tool in tools:
                box = Checkbox(tool, value=tool in self._checked, classes="wizard-tool")
                self._boxes[tool] = box
                container.mount(box)

    @staticmethod
    def _safe_id(name: str) -> str:
        """server 名 → 合法 Textual id（只留字母数字下划线连字符）。"""
        cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))
        return cleaned or "server"

    def refresh_tools(
        self,
        groups: Dict[str, List[str]],
        servers: Optional[List[str]] = None,
        mcp_ready: bool = True,
    ) -> None:
        """MCP 就绪后填充/刷新外置工具列表（保留已勾选项）。"""
        self._save_checked()
        self._groups = dict(groups or {})
        if servers is not None:
            self._servers = list(servers)
        self._mcp_ready = mcp_ready
        self._render_ext()

    def _save_checked(self) -> None:
        for name, box in self._boxes.items():
            if box.value:
                self._checked.add(name)
            else:
                self._checked.discard(name)
        self._system_tools = bool(self.query_one("#cap-system", Checkbox).value)

    # ---------------- 交互 ----------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "wizard-ok":
            self._save_checked()
            self.dismiss(
                {
                    "system_tools": self._system_tools,
                    "mcp_tools": sorted(self._checked),
                }
            )
        elif event.button.id == "wizard-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
