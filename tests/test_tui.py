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
        self.files = SimpleNamespace(
            create=lambda file, purpose: SimpleNamespace(id="file-api-tui"),
            list=lambda: SimpleNamespace(data=[]),
            delete=lambda file_id: SimpleNamespace(deleted=True),
        )


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

    def fetch_balance(self):
        # 测试不联网：余额置空，避免真实请求 DeepSeek /user/balance
        return None


async def main() -> int:
    print("== ChatApp headless 验证（2b） ==")
    test_markdown_safety()
    test_selection_safety()
    test_screen_forward_safety()
    fake = FakeController()
    app = ChatApp(controller=fake)
    async with app.run_test(size=(100, 30)) as pilot:
        # --- 1. 布局 ---
        check("布局：输入框存在且聚焦", app.query_one("#chat-input", ChatInput) is not None)
        check("布局：消息流存在", bool(app.query_one("#chat-log", Markdown)))
        check("布局：会话树存在", bool(app.query_one("#tree", Tree)))
        check("布局：状态条存在", bool(app.query_one("#usage-bar", Horizontal)))
        check("布局：状态条两分栏", bool(app.query_one("#usage-left", Static))
              and bool(app.query_one("#usage-right", Static)))
        for _ in range(10):
            await pilot.pause()
        usage_left = str(app.query_one("#usage-left", Static).content)
        check("状态条左栏显示缓存/余额", "缓存" in usage_left)

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
        # 思考过程不折叠：完整内容以灰色块引用显示（无点击展开交互）
        check("思考以引用格式显示", "思考过程" in chat.source and "> 思考中" in chat.source)
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

        # --- 7.6 多模态：/import、/files、/set detail、/set model vision、节点视图占位 ---
        from mincli.tools.images import ImageAttachment
        _tmp_img = tempfile.mkdtemp(prefix="mincli_tui_img_")
        _tpng = os.path.join(_tmp_img, "t.png")
        with open(_tpng, "wb") as f:
            f.write(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\x0dIHDR"
                + (800).to_bytes(4, "big")
                + (600).to_bytes(4, "big")
                + b"\x08\x06\x00\x00\x00"
            )
        _txt = os.path.join(_tmp_img, "note.txt")
        with open(_txt, "w", encoding="utf-8") as f:
            f.write("hello")

        await type_command(f"/import {_tpng}")
        for _ in range(10):
            await pilot.pause()
        check("命令：/import 添加待发送图片", len(fake.pending_images) == 1)
        hint = app.query_one("#import-status", Static)
        check("导入状态栏显示", "已导入 1 个文件" in str(hint.content) and hint.has_class("visible"))

        # 一次导入多个文件（图片 + 文本）+ 悬停弹窗（完整文件名列表）
        await type_command("/import clear")
        for _ in range(10):
            await pilot.pause()
        await type_command(f"/import {_tpng} {_txt}")
        for _ in range(10):
            await pilot.pause()
        check("命令：/import 多文件", len(fake.pending_images) == 1 and len(fake.imported_files) == 1)
        check("状态栏数量与前2文件名", "已导入 2 个文件" in str(hint.content) and "note.txt" in str(hint.content))
        popup = app.query_one("#import-popup", Static)
        sr = hint.region
        app.post_message(events.MouseMove(None, 1, 1, 0, 0, 0, False, False, False,
                                          screen_x=sr.x + 1, screen_y=sr.y))
        for _ in range(5):
            await pilot.pause()
        check("悬停显示完整列表", popup.has_class("visible") and "note.txt" in str(popup.content))
        app.post_message(events.MouseMove(None, 1, 1, 0, 0, 0, False, False, False,
                                          screen_x=sr.x + 1, screen_y=max(0, sr.y - 5)))
        for _ in range(5):
            await pilot.pause()
        check("移开鼠标自动消失", not popup.has_class("visible"))

        await type_command("/import clear")
        for _ in range(10):
            await pilot.pause()
        check("命令：/import clear 清空", len(fake.pending_images) == 0 and len(fake.imported_files) == 0
              and not app.query_one("#import-status", Static).has_class("visible"))

        # --- 7.6b 拖入文件直接导入（终端把路径粘贴进输入框） ---
        inp.clear()
        await pilot.pause()
        app.post_message(events.Paste(f'"{_tpng}" "{_txt}"'))
        for _ in range(20):
            await pilot.pause()
        check("拖入：多文件自动导入", len(fake.pending_images) == 1 and len(fake.imported_files) == 1)
        check("拖入：输入框未残留路径文本", inp.text == "")
        check("拖入：状态栏显示数量", "已导入 2 个文件" in str(hint.content) and hint.has_class("visible"))

        await type_command("/import clear")
        for _ in range(10):
            await pilot.pause()

        # 普通文本粘贴不触发导入、照常插入
        inp.clear()
        await pilot.pause()
        app.post_message(events.Paste("这是一段普通粘贴的文本"))
        for _ in range(10):
            await pilot.pause()
        check("普通粘贴不导入", len(fake.pending_images) == 0 and len(fake.imported_files) == 0)
        check("普通粘贴照常插入（仅一次）", inp.text == "这是一段普通粘贴的文本")

        # 焦点不在输入框时（如在对话树），路径粘贴仍兜底导入
        inp.clear()
        app.set_focus(app.query_one("#tree", Tree))
        await pilot.pause()
        app.post_message(events.Paste(_tpng))
        for _ in range(20):
            await pilot.pause()
        check("焦点在树时拖入也导入", len(fake.pending_images) == 1)
        inp.focus()
        await pilot.pause()
        await type_command("/import clear")
        for _ in range(10):
            await pilot.pause()
        check("拖入后 clear 清空", len(fake.pending_images) == 0 and len(fake.imported_files) == 0)

        await type_command("/set detail low")
        await pilot.pause()
        check("命令：/set detail low", fake.image_detail == "low")

        await type_command("/set model vision")
        await pilot.pause()
        check("命令：/set model vision", fake.current_model == "deepseek-v4-flash-vision-exp")

        await type_command("/files list")
        for _ in range(10):
            await pilot.pause()
        check("命令：/files list 显示空列表", "已上传图片文件" in app.query_one("#chat-log", Markdown).source)

        # 节点视图：带图片的节点渲染占位（直接驱动 node_created 事件）
        node = fake.tree.create_root("看图", "", "", "图题", 0, 0)
        node.user_images = [ImageAttachment(source=_tpng, name="t.png", width=800, height=600)]
        fake.tree.current_node = node
        await app._handle_event(ControllerEvent.node_created(node))
        for _ in range(10):
            await pilot.pause()
        check("节点视图含图片占位", "[图片: t.png (800x600)]" in app.query_one("#chat-log", Markdown).source)
        await type_command("/clear")

        # --- 7.6c 多轮工具调用：多个思考块穿插正文，各自按引用格式显示 ---
        node_m = fake.tree.create_root("多轮问题", "", "", "多轮标题", 0, 0)
        fake.tree.current_node = node_m
        await app._handle_event(ControllerEvent.node_created(node_m))
        await app._handle_event(ControllerEvent.stream("", "第一轮思考内容"))
        await app._handle_event(ControllerEvent.stream("第一轮正文", ""))
        await app._handle_event(ControllerEvent.tool("execute_command", '{"command":"ls"}', "（完成）"))
        await app._handle_event(ControllerEvent.stream("", "第二轮思考内容"))
        await app._handle_event(ControllerEvent.stream("第二轮正文", ""))
        await app._handle_event(ControllerEvent.done(node_m))
        for _ in range(30):
            await pilot.pause()
        src_m = app.query_one("#chat-log", Markdown).source
        check("多轮：两个思考块各带标题", src_m.count("思考过程") == 2)
        check("多轮：第一轮思考按引用显示", "> 第一轮思考内容" in src_m)
        check("多轮：第二轮思考按引用显示", "> 第二轮思考内容" in src_m)
        check("多轮：两轮正文都显示", "第一轮正文" in src_m and "第二轮正文" in src_m)
        node_m.reasoning = "第一轮思考内容\n第二轮思考内容"  # 模拟 controller 汇总全部轮次思考
        check("多轮：节点视图也含思考引用", "思考过程" in app._node_content(node_m)
              and "> 第二轮思考内容" in app._node_content(node_m))
        await type_command("/clear")

        await type_command("/unknown_cmd")
        await pilot.pause()
        check("未知命令不发送给 LLM", fake.client.chat.completions.script == [])

        # --- 7.7 /compact 命令（全部压缩 + 新建摘要节点） ---
        fake.tree.create_root("问题1", "回答1", "", "标题1", 1, 1)
        n2 = fake.tree.add_child(fake.tree.root, "问题2", "回答2", "", "标题2", 1, 1)
        n3 = fake.tree.add_child(n2, "问题3", "回答3", "", "标题3", 1, 1)
        n4 = fake.tree.add_child(n3, "问题4", "回答4", "", "标题4", 1, 1)
        fake.tree.current_node = n4
        fake.client.chat.completions.script.append(_FakeChatResponse(content="【摘要】TUI 压缩测试内容"))

        # 带参数被拦截
        await type_command("/compact 0")
        for _ in range(10):
            await pilot.pause()
        check("命令：/compact 带参数被拦截", fake.tree.compaction is None)

        await type_command("/compact")
        for _ in range(30):
            await pilot.pause()
            if fake.tree.compaction and fake.tree.current_node.id == fake.tree.compaction["boundary_id"]:
                break
        chat = app.query_one("#chat-log", Markdown)
        check("命令：/compact 新建摘要节点", fake.tree.compaction is not None)
        node_id = fake.tree.compaction["boundary_id"]
        check(
            "命令：/compact 显示摘要",
            "上下文压缩摘要" in chat.source and "TUI 压缩测试内容" in chat.source,
        )
        check("命令：/compact 摘要节点为当前", fake.tree.current_node.id == node_id)

        # 当前已是摘要节点 → 再次 /compact 被拦截
        await type_command("/compact")
        for _ in range(10):
            await pilot.pause()
        check("命令：/compact 禁止重复压缩",
              fake.tree.compaction["boundary_id"] == node_id and len(fake.tree.nodes) == 5)

        # --- 7.8 /delete 多节点 ---
        await type_command("/clear")
        fake.tree.create_root("根", "回", "", "根", 1, 1)
        da1 = fake.tree.add_child(fake.tree.root, "q", "a", "", "t", 1, 1)
        db1 = fake.tree.add_child(fake.tree.root, "q", "a", "", "t", 1, 1)
        db2 = fake.tree.add_child(db1, "q", "a", "", "t", 1, 1)
        fake.tree.current_node = db2
        await type_command(f"/delete {da1.id} {db1.id} {db2.id}")
        for _ in range(30):
            await pilot.pause()
            if app.screen.query("#confirm-yes"):
                break
        check("删除：多节点确认弹窗", bool(app.screen.query("#confirm-yes")))
        app.screen.query_one("#confirm-yes", Button).press()
        for _ in range(10):
            await pilot.pause()
        check("删除：父节点级联删除子节点",
              da1.id not in fake.tree.nodes and db1.id not in fake.tree.nodes and db2.id not in fake.tree.nodes)
        check("删除：根节点保留", fake.tree.root is not None and fake.tree.current_node is not None)

        # --- 7.9 文字选择 + 复制 ---
        chat = app.query_one("#chat-log", Markdown)
        check("选区：ALLOW_SELECT 已开启", app.ALLOW_SELECT)
        await pilot.mouse_down("#chat-log", offset=(10, 4))
        await pilot.pause()
        await pilot.mouse_up("#chat-log", offset=(45, 14))
        for _ in range(8):
            await pilot.pause()
        check("选区：拖选后产生选区", bool(app.screen.selections))
        sel_text = app.screen.get_selected_text()
        check("选区：可提取选中文本", bool(sel_text))

        copied: list[str] = []
        app.copy_to_clipboard = lambda t: copied.append(t)  # 记录而非真复制
        await pilot.press("ctrl+c")  # 有选区 → 复制而非退出（所有平台统一）
        for _ in range(8):
            await pilot.pause()
        check("复制：Ctrl+C 复制选中文本而非退出", bool(copied) and copied[0] == sel_text)
        check("复制：有选区时未退出", app.screen is not None)
        del app.copy_to_clipboard  # 恢复为类方法
        app.screen.clear_selection()
        await pilot.pause()
        check("选区：清除后无选区", not app.screen.selections)

        # --- 8. Ctrl+C 退出 ---
        await pilot.press("ctrl+c")

    check("退出时保存会话", fake.saved)
    check("退出时关闭控制器", fake.closed)

    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    return 0 if FAIL == 0 else 1


def test_selection_safety():
    """Textual 选区提取越界防御：拖选跨越流式重建时锚点越界不崩溃。"""
    from textual.selection import Selection
    from textual.geometry import Offset

    # 崩溃路径：start/end 行号 == 内容行数（内容在选中期间被重建/缩短）
    sel = Selection.from_offsets(Offset(0, 3), Offset(0, 3))
    try:
        result = sel.extract("line1\nline2\nline3")
        ok = True
    except IndexError:
        ok = False
    check("选区提取越界不崩溃", ok and result == "line1\nline2\nline3")
    # 正常路径不受影响
    sel2 = Selection.from_offsets(Offset(0, 0), Offset(5, 0))
    check("选区提取正常路径不变", sel2.extract("hello world") == "hello")


def test_screen_forward_safety():
    """Screen._forward_event 崩溃重试补丁：MouseDown 命中已分离 widget
    （选区初始化 AttributeError）→ 临时关选区重试，不崩溃且 ALLOW_SELECT 恢复。"""
    from mincli.tui import app as app_mod
    from textual.screen import Screen

    app_mod._patch_textual_screen_forward_event()  # 幂等
    check(
        "Screen._forward_event 已打补丁",
        getattr(Screen, "_mincli_safe_forward_event", False),
    )

    class _FakeApp:
        ALLOW_SELECT = True

    class _FakeSelf:
        app = _FakeApp()

        def _orig(self, event):
            self.calls += 1
            if self.calls == 1:
                raise AttributeError("'NoneType' object has no attribute 'region'")
            return "ok"

    fake = _FakeSelf()
    fake.calls = 0
    mousedown = events.MouseDown(None, 1, 1, 0, 0, 0, False, False, False)
    saved = app_mod._ORIG_SCREEN_FORWARD_EVENT
    app_mod._ORIG_SCREEN_FORWARD_EVENT = _FakeSelf._orig  # 未绑定：orig(self, event)
    try:
        result = app_mod._safe_screen_forward_event(fake, mousedown)
    finally:
        app_mod._ORIG_SCREEN_FORWARD_EVENT = saved
    check("命中分离 widget 重试不崩溃", result == "ok" and fake.calls == 2)
    check("重试后 ALLOW_SELECT 恢复", _FakeApp.ALLOW_SELECT is True)

    # 非 MouseDown 事件直接转发（不经过重试逻辑）
    passthrough = {"n": 0}

    def _orig2(self, event):
        passthrough["n"] += 1
        return "pass-through"

    app_mod._ORIG_SCREEN_FORWARD_EVENT = _orig2
    r2 = app_mod._safe_screen_forward_event(fake, "not-mousedown")
    check("非 MouseDown 事件直接转发", r2 == "pass-through" and passthrough["n"] == 1)
    app_mod._ORIG_SCREEN_FORWARD_EVENT = saved


def test_markdown_safety():
    """markdown_it 防御补丁：规则被包装、越界不崩、正常解析不变。"""
    import markdown_it.main

    from mincli.markdown_safe import _patch_markdown_it

    _patch_markdown_it()  # 幂等
    parser = markdown_it.main.MarkdownIt()
    rules = parser.block.ruler.__rules__
    by_name = {r.name: r.fn for r in rules}
    safe = {
        name
        for name, fn in by_name.items()
        if getattr(fn, "__name__", "") == "_safe"
    }
    check(
        "补丁包装核心块规则",
        {"html_block", "table", "blockquote", "fence"} <= safe,
    )
    # 正常 markdown 解析结果与未包装时一致（token 数）
    md = "## 标题\n\n- 列表项\n\n```python\nprint(1)\n```\n\n| A | B |\n|:--:|:--:|\n| 1 | 2 |\n"
    check("正常解析 token 数不变", len(parser.parse(md)) == 14)
    # 极端破坏输入不崩溃（引用块内表格被截断等形态）
    import random

    random.seed(2026)
    crashed = False
    for _ in range(120):
        lines = []
        for ln in (
            "用户补充了：",
            "- 还有4道错题：22、23、24、25",
            "| 题号 | 题型 | 失分原因 | 涉及知识点 | 模块 |",
            "|:----:|:----:|:----:|:--------:|:----:|",
            "| 6 | 选择 | 计算失误 | 解方程忘检验 | 代数 |",
        ):
            mode = random.random()
            if mode < 0.3:
                lines.append(f"> {ln}")
            elif mode < 0.5:
                lines.append(ln)
            elif mode < 0.7:
                cut = random.randint(1, max(1, len(ln) - 1))
                lines.append(f"> {ln[:cut]}")
                lines.append(ln[cut:])
            else:
                lines.append(f"> > {ln}")
        src = "\n".join(lines) + ("\n" if random.random() < 0.5 else "")
        try:
            parser.parse(src)
        except IndexError:
            crashed = True
            break
    check("极端输入不触发 IndexError", not crashed)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
