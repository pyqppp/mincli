"""ChatApp headless 验证（阶段 2b/3：注入 FakeController，覆盖流式/树/命令）。

运行：`venv/bin/python -m tests.test_tui`
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli.models import ConversationTree
from mincli.controller import ChatController, ControllerEvent
from mincli.tui.app import ChatApp
from mincli.tui.widgets import ChatInput
from textual import events
from textual.containers import Horizontal
from textual.widgets import Button, Markdown, Static, Tree
from textual.widgets._markdown import (
    MarkdownBulletList,
    MarkdownOrderedList,
    MarkdownTableContent,
)

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


class _FakeDelta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _FakeChunk:
    def __init__(self, content=None, reasoning_content=None):
        self.choices = [SimpleNamespace(delta=_FakeDelta(content, reasoning_content))]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeChatResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(_FakeMessage(content))]


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)

    def create(self, **kwargs):
        if not self.script:
            raise AssertionError("脚本已用完")
        return self.script.pop(0)


class _FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))


class FakeController(ChatController):
    """真实 ChatController + 固定事件流（不联网）。"""

    SAVE_FILE = os.path.join(
        tempfile.mkdtemp(prefix="mincli_tui_test_"), "session.json"
    )

    def __init__(self):
        super().__init__(
            client=_FakeClient([]),
            default_system="sys",
            default_temperature=1.0,
            auto_start_mcp=False,
        )
        self.saved = False
        self.closed = False

    def send_message(self, text, emit):
        node = self.tree.create_root(text, "你好，世界！", "思考中", "测试标题", 10, 5)
        emit(ControllerEvent.node_created(node))
        emit(ControllerEvent.stream("你好", "思考中"))
        emit(ControllerEvent.stream("，世界！", ""))
        table = (
            "\n\n| 列A | 列B | 列C | 列D | 列E |\n"
            "|---|---|---|---|---|\n"
            "| 这是一个需要自动换行的超长单元格内容很长很长 | "
            "这是一个需要自动换行的超长单元格内容很长很长 | "
            "这是一个需要自动换行的超长单元格内容很长很长 | "
            "这是一个需要自动换行的超长单元格内容很长很长 | "
            "这是一个需要自动换行的超长单元格内容很长很长 |"
        )
        emit(ControllerEvent.stream(table, ""))
        emit(
            ControllerEvent.stream(
                "\n\n1. 有序第一项\n2. 有序第二项\n\n- 无序甲\n- 无序乙", ""
            )
        )
        emit(ControllerEvent.done(node))
        return node

    def save_session(self):
        self.saved = True

    def close(self):
        self.closed = True


async def main() -> int:
    print("== ChatApp headless 验证（2b） ==")
    fake = FakeController()
    app = ChatApp(controller=fake)
    async with app.run_test(size=(100, 30)) as pilot:
        # --- 1. 布局 ---
        check("布局：输入框存在且聚焦", app.query_one("#chat-input", ChatInput) is not None)
        check("布局：消息流存在", bool(app.query_one("#chat-log", Markdown)))
        check("布局：会话树存在", bool(app.query_one("#tree", Tree)))

        # --- 2. 发送消息 → 流式渲染 ---
        inp = app.query_one("#chat-input", ChatInput)
        await pilot.press("你", "好")
        check("输入中文正常", inp.text == "你好")
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
            if "世界！" in app.query_one("#chat-log", Markdown).source:
                break
        chat = app.query_one("#chat-log", Markdown)
        check("流式内容已追加", "你好，世界！" in chat.source)
        check("思考已显示", "思考中" in chat.source)
        check("token 统计已显示", "tokens" in chat.source)

        # --- 3. 会话树更新 + 光标跟随新节点 ---
        for _ in range(5):
            await pilot.pause()
        tree = app.query_one("#tree", Tree)
        check("树根节点已更新", str(tree.root.label) == "main: 测试标题")
        check(
            "树光标跟随新节点",
            tree.cursor_node is not None and tree.cursor_node.data == "main",
        )

        # --- 3.5 表格按终端宽度换行（不被横向截断） ---
        for _ in range(10):
            await pilot.pause()
            if app.query(MarkdownTableContent):
                break
        chat = app.query_one("#chat-log", Markdown)
        tables = list(app.query(MarkdownTableContent))
        check("表格已渲染", len(tables) >= 1)
        if tables:
            tc = tables[0]
            check("表格宽度不超出消息区", tc.region.width <= chat.region.width - 4)
            heights = [c.region.height for c in tc.query(".cell")]
            check("超长单元格自动换行", heights and max(heights) > 1)

        # --- 3.6 消息区紧贴右边缘（滚动条不悬空） ---
        check(
            "消息区紧贴终端右边缘",
            chat.region.x + chat.region.width == app.screen.size.width,
        )

        # --- 3.7 列表项紧凑无大间隔 ---
        for _ in range(10):
            await pilot.pause()
            if app.query(MarkdownBulletList) or app.query(MarkdownOrderedList):
                break
        list_rows = [
            w
            for w in app.query(Horizontal)
            if w.parent is not None
            and type(w.parent).__name__ in ("MarkdownBulletList", "MarkdownOrderedList")
        ]
        check("列表项行高为内容高度", list_rows and all(r.region.height == 1 for r in list_rows))

        # --- 4. 点击树节点切换 ---
        tree.select_node(tree.root)
        await pilot.pause()
        check("切换节点显示内容", "测试标题" in app.query_one("#chat-log", Markdown).source)

        # --- 5. 锁定键过滤 ---
        inp.clear()
        app.post_message(events.Key("caps_lock", "A"))
        await pilot.pause()
        check("caps_lock 不产生输入", inp.text == "")

        # --- 6. 确认弹窗 ---
        check("confirm 回调存在", callable(fake.confirm))

        # --- 7. 斜杠命令 ---
        async def type_command(cmd: str) -> None:
            inp.clear()
            await pilot.press(*list(cmd))
            await pilot.press("enter")

        await type_command("/help")
        for _ in range(10):
            await pilot.pause()
            if "📖 帮助" in app.query_one("#chat-log", Markdown).source:
                break
        check("命令：/help 显示帮助", "📖 帮助" in app.query_one("#chat-log", Markdown).source)

        await type_command("/set model pro")
        await pilot.pause()
        check("命令：/set model pro", fake.current_model.endswith("pro"))

        await type_command("/tree")
        for _ in range(10):
            await pilot.pause()
            if "main: 测试标题" in app.query_one("#chat-log", Markdown).source:
                break
        check("命令：/tree 显示对话树", "main: 测试标题" in app.query_one("#chat-log", Markdown).source)

        # --- 7.5 命令补全弹窗 + /delete 确认删除（节点此时已存在） ---
        fake.tree.add_child(fake.tree.root, "子问题", "子回答", "", "子节点", 1, 1)
        inp.clear()
        await pilot.press("/", "d", "e", "l")
        for _ in range(5):
            await pilot.pause()
        popup = app.query_one("#cmd-popup")
        body = app.query_one("#cmd-popup-body", Static)
        check("补全：/del 显示候选列表", popup.has_class("visible") and "/delete" in str(body.content))

        await pilot.press("tab")
        for _ in range(5):
            await pilot.pause()
        check("补全：Tab 补全为 /delete", inp.text == "/delete")
        check("补全：补全后转为命令提示", "用法: /delete" in str(body.content))

        await pilot.press(" ", "a", "1")
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
            if app.screen.query("#confirm-yes"):
                break
        check("删除：确认弹窗出现", bool(app.screen.query("#confirm-yes")))
        app.screen.query_one("#confirm-yes", Button).press()
        for _ in range(10):
            await pilot.pause()
        check("删除：确认后节点已删除", fake.tree.nodes.get("a1") is None)

        await type_command("/clear")
        await pilot.pause()
        check("命令：/clear 清空会话", fake.tree.root is None)

        await type_command("/set show")
        for _ in range(10):
            await pilot.pause()
            if "当前配置" in app.query_one("#chat-log", Markdown).source:
                break
        check("命令：/set show 显示配置", "当前配置" in app.query_one("#chat-log", Markdown).source)

        await type_command("/unknown_cmd")
        await pilot.pause()
        check("未知命令不发送给 LLM", fake.client.chat.completions.script == [])

        # --- 8. Ctrl+C 退出 ---
        await pilot.press("ctrl+c")

    check("退出时保存会话", fake.saved)
    check("退出时关闭控制器", fake.closed)

    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
