"""ChatApp headless 验证（阶段 2b/3：注入 FakeController，覆盖流式/树/命令）。

运行：`venv/bin/python -m tests.test_tui`
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli.models import ConversationTree
from mincli.controller import ChatController, ControllerEvent
from mincli.tui.app import ChatApp
from mincli.tui.commands import CMD_SPEC, complete, help_markdown, usage_line
from mincli.tui.widgets import ChatInput
from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Markdown, Static, Tree
from textual.widgets._markdown import (
    MarkdownBulletList,
    MarkdownOrderedList,
    MarkdownTableContent,
)

PASS = 0
FAIL = 0

# 触发 Textual markup 解析崩溃的文本形态：未闭合的 `[` + 键="跨行值"
# （markup 的 quoted value 正则 `".*?"` 不跨行；工具参数里的 JSON 数组 +
# 多行字符串、模型生成的标题都可能是这种形态）
MARKUP_BAD_TEXT = (
    '结果：[{"title":"a"}, '
    'tool_choice="Tools 表 web_search 内置工具 支持情况\ntool_choice 联网搜索"'
)


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


class _FakeFiles:
    """Files API 模拟：内存列表，支持 order/limit/after 与 retrieve。"""

    def __init__(self):
        # 按创建时间升序保存（官方列表默认 asc，desc 时反转）
        self.items = [
            {"id": "file-api-old", "filename": "old.jpg", "bytes": 1024,
             "created_at": 1690000000, "expires_at": None},
            {"id": "file-api-tui", "filename": "tui.png", "bytes": 2048,
             "created_at": 1700000000, "expires_at": None},
        ]
        self.created = []

    def create(self, file, purpose="user_data"):
        item = {"id": f"file-api-{len(self.items)}", "filename": "up.png",
                "bytes": 1, "created_at": 1700000000, "expires_at": None}
        self.items.append(item)
        self.created.append(item["id"])
        return SimpleNamespace(id=item["id"])

    def list(self, limit=1000, order="asc", after=None):
        items = list(self.items)
        if order == "desc":
            items = list(reversed(items))
        if after:
            ids = [i["id"] for i in items]
            items = items[ids.index(after) + 1:] if after in ids else items
        return SimpleNamespace(
            data=[SimpleNamespace(**i) for i in items[:limit]],
            has_more=len(items) > limit,
        )

    def retrieve(self, file_id):
        for item in self.items:
            if item["id"] == file_id:
                return SimpleNamespace(**item)
        raise RuntimeError(f"file not found: {file_id}")

    def delete(self, file_id):
        self.items = [i for i in self.items if i["id"] != file_id]
        return SimpleNamespace(deleted=True)


class _FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))
        self.files = _FakeFiles()


class FakeController(ChatController):
    """真实 ChatController + 固定事件流（不联网）。"""

    # 工作流提炼/修订的固定产出（不联网）
    WF_STUB_DOC = (
        "目标：测试工作流\n\n"
        "变量：\n"
        "- {path}：目标文件（示例：a.txt）\n\n"
        "步骤：\n"
        "1. 读取 {path} 并总结\n"
    )

    def __init__(self, with_tree: bool = True, trees_dir: str | None = None):
        wf_dir = tempfile.mkdtemp(prefix="mincli_tui_wf_")
        self.WORKFLOWS_FILE = os.path.join(wf_dir, "workflows.json")
        # 每个控制器一个独立树目录：树会落盘，共享目录会让用例互相污染
        self.TREES_DIR = trees_dir or tempfile.mkdtemp(prefix="mincli_tui_trees_")
        super().__init__(
            client=_FakeClient([]),
            default_system="sys",
            default_temperature=1.0,
            auto_start_mcp=False,
            trees_dir=self.TREES_DIR,
        )
        # 默认先建一棵树：没有树时 App 启动会弹建树向导，绝大多数用例不测这个
        if with_tree and not self.has_trees:
            self.create_tree()
        self.saved = False
        self.closed = False
        self.mcp_start_count = 0

    def start_mcp(self):
        """测试桩：不真的连接 MCP（会被 ChatApp 在挂载时调用）。

        真实实现会起子进程 + 连远程 server，测试里既慢又要联网；这里只记录
        调用次数，_mcp 保持未设置，App 侧也就不会派生就绪监听 worker。
        """
        self.mcp_start_count += 1

    def _wf_call_model(self, messages, max_tokens):
        user = str((messages[-1] or {}).get("content", ""))
        if "用户修改要求" in user:
            return "目标：已按修改要求更新的工作流\n\n步骤：\n1. 修订后的步骤"
        return self.WF_STUB_DOC

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


def test_packaging():
    """打包回归：运行时读取的非 .py 数据文件必须随 wheel/sdist 一起安装。

    背景：pyproject 的 package-data 键是「包名」，子包数据要单独声明；
    mincli.tui 漏声明导致 chat.tcss 不进安装包，`pip install .` 后
    `mincli chat` 启动即 StylesheetError: unable to read CSS file。
    """
    import fnmatch
    import re

    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")

    # 解析 [tool.setuptools.package-data]（容忍单行/多行数组）
    declared: dict[str, list[str]] = {}
    head = re.search(r"(?m)^\[tool\.setuptools\.package-data\]\s*$", text)
    if head:
        body = []
        for line in text[head.end():].splitlines():
            if line.lstrip().startswith("["):
                break
            body.append(line)
        for km in re.finditer(
            r'(?m)^\s*"?([\w.\-]+)"?\s*=\s*\[(.*?)\]', "\n".join(body), re.S
        ):
            declared[km.group(1)] = re.findall(r'["\']([^"\']+)["\']', km.group(2))

    def matches(rel: str, pattern: str) -> bool:
        # 与 setuptools 的 glob 一致：* 不跨目录分隔符
        rel_parts, pat_parts = rel.split("/"), pattern.split("/")
        return len(rel_parts) == len(pat_parts) and all(
            fnmatch.fnmatchcase(a, b) for a, b in zip(rel_parts, pat_parts)
        )

    def covered(rel: str) -> bool:
        return any(
            matches(rel, f"{pkg.replace('.', '/')}/{pat}")
            for pkg, patterns in declared.items()
            for pat in patterns
        )

    data_files = sorted(
        f"{p.parent.relative_to(root).as_posix()}/{p.name}"
        for p in (root / "mincli").rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in {".py", ".pyc"}
        and p.name != ".DS_Store"
    )
    uncovered = [f for f in data_files if not covered(f)]
    check("打包：package-data 覆盖全部运行时数据文件",
          bool(data_files) and not uncovered)
    if uncovered:
        print(f"       未声明: {uncovered}")

    # chat.tcss 必须与 app.py 同目录，且能像 Textual 启动时那样被读取
    import mincli.tui.app as tui_app

    css = Path(tui_app.__file__).resolve().parent / Path(str(ChatApp.CSS_PATH)).name
    check("打包：chat.tcss 与 app.py 同目录存在", css.is_file())
    readable = False
    if css.is_file():
        try:
            from textual.css.stylesheet import Stylesheet

            Stylesheet().read_all([css])
            readable = True
        except Exception as exc:  # pragma: no cover - 防御分支
            print(f"       读取 CSS 失败: {exc!r}")
    check("打包：chat.tcss 可被 Textual 读取（启动无 StylesheetError）", readable)


def test_markup_safety():
    """模型/用户可控文本里的 markup 形态不得让 Textual 渲染崩溃。

    背景：Textual 8 的 Static/Toast 默认按 markup 解析字符串，而它的
    quoted value 正则 `".*?"` 不跨行；内容里只要出现「未闭合的 `[` + 键="跨行值"」
    （工具参数里的 JSON 数组 + 多行字符串、模型生成的标题等）就会抛
    MarkupError: Expected markup value，整个渲染/通知报错。
    """
    from textual.content import Content
    from textual.widgets import Static

    from mincli.tui.widgets import ToolCard

    bad = MARKUP_BAD_TEXT
    rejected = False
    try:
        Content.from_markup(bad)
    except Exception:
        rejected = True
    check("markup 解析器确实会拒绝该形态（回归用例有效）", rejected)

    card = ToolCard("tavily_research", [bad])
    try:
        card.set_start("tavily_research", [bad])
        card.set_result(bad)
        card_ok = True
    except Exception:
        card_ok = False
    check("工具卡片按纯文本渲染（不再抛 MarkupError）", card_ok)
    check("工具卡片保留原始方括号文本", "tool_choice" in card._render_text())


def _max_quote_depth(md_text: str) -> int:
    """Markdown 里引用块的最大嵌套层数（>1 即渲染出第二道竖线）。"""
    from markdown_it import MarkdownIt

    depth = peak = 0
    for tok in MarkdownIt().parse(md_text):
        if tok.type == "blockquote_open":
            depth += 1
            peak = max(peak, depth)
        elif tok.type == "blockquote_close":
            depth -= 1
    return peak


def _inline_text(md_text: str) -> str:
    """取 Markdown 渲染出的纯文本（inline token 内容拼接）。"""
    from markdown_it import MarkdownIt

    return "\n".join(
        tok.content for tok in MarkdownIt().parse(md_text) if tok.type == "inline"
    )


def test_quote_marker_safety():
    """思考过程引用行不得与我们的 `> ` 前缀叠加成嵌套引用（界面显示成 >>）。

    背景：思考过程整体按灰色块引用渲染（每行加 `> `）。模型思考里自带引用行
    （`> 注意…`），或流式 chunk 边界正好落在 `>` 之前时，前缀会叠加成 `>>` /
    `> >`，被解析成「引用块里套引用块」——多出一道竖线。
    """
    from mincli.tui.app import (
        ChatApp,
        escape_leading_quote_markers,
        quote_block_text,
    )

    check("未转义的叠加确实会形成嵌套引用（回归用例有效）",
          _max_quote_depth("> 思考过程\n>\n> > 注意") == 2)

    # 行首引用标记转义
    check("行首 `>` 被转义", escape_leading_quote_markers("> 注意") == "\\> 注意")
    check("连写 `>>` 被逐个转义", escape_leading_quote_markers(">> 连写") == "\\>\\> 连写")
    check("`> >` 多层标记被转义", escape_leading_quote_markers("> > 多层") == "\\> \\> 多层")
    check("正文中间的 `>` 不动", escape_leading_quote_markers("a > b") == "a > b")
    check("缩进代码块（4 空格）不动", escape_leading_quote_markers("    > code") == "    > code")

    # 节点视图整段引用
    node_md = ChatApp._build_reasoning_md("第一轮\n> 引用行\n\n结尾")
    check("节点视图引用无嵌套", _max_quote_depth(node_md) == 1)
    check("节点视图引用行仍显示字面 `>`", "> 引用行" in _inline_text(node_md))
    check("节点视图标题在最前", node_md.splitlines()[0] == "> 思考过程")

    # 流式 chunk 拼接（chunk 末尾换行 + 下一 chunk 以 `>` 开头 → 曾经的 `>>`）
    streamed = ChatApp._reasoning_chunk_md("先想一下。\n") + ChatApp._reasoning_chunk_md(
        "> 注意：要核对。\n"
    )
    check("流式拼接不出现字面 `>>`", ">>" not in streamed)
    check("流式拼接不出现字面 `> >`", "> >" not in streamed)
    check("流式拼接无嵌套引用", _max_quote_depth("> 思考过程\n>\n" + streamed) == 1)
    check("流式拼接仍显示字面 `>`", "> 注意：要核对。" in _inline_text(streamed))

    # 状态/告警多行文本同样是模型可控内容
    quoted = quote_block_text("第一行\n> 第二行")
    check("多行状态逐行引用", quoted.count("\n> ") == 1 and quoted.startswith("> 第一行"))
    check("多行状态无嵌套引用", _max_quote_depth(quoted) == 1)


async def test_reasoning_quote_after_tool():
    """工具调用后的新一轮思考：仍带「思考过程」标题与 `> ` 前缀，且不嵌套引用。

    工具卡片会把正文段固化（新建 Markdown 段）。若思考块状态不复位，新一轮思考
    会以为块还开着，丢掉标题和前缀；而模型思考自带的 `>` 会与残留前缀叠加成 `>>`。
    """
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownBlockQuote, MarkdownParagraph

    ctrl = FakeController()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 40)) as pilot:
        for _ in range(5):
            await pilot.pause()
        node = ctrl.tree.create_root("测试", "", "", "引用标题", 10, 5)
        ctrl.tree.current_node = node
        await app._handle_event(ControllerEvent.node_created(node))
        # 第一轮：只有思考、没有正文（真实情况：思考完直接调工具）
        await app._handle_event(ControllerEvent.stream("", "先想一下。\n"))
        await asyncio.sleep(0.15)  # 触发一次批量渲染，模拟真实 chunk 边界
        await app._handle_event(ControllerEvent.tool("web_search", '{"query":"x"}', ""))
        await app._handle_event(
            ControllerEvent.tool("web_search", '{"query":"x"}', "工具 web_search 完成")
        )
        # 工具之后继续思考：思考里自带引用行
        await app._handle_event(ControllerEvent.stream("", "> 注意：结果有冲突。\n"))
        await asyncio.sleep(0.15)
        await app._handle_event(ControllerEvent.stream("", "> 再核对一次。\n"))
        await app._handle_event(ControllerEvent.stream("结论：一致。", ""))
        await app._handle_event(ControllerEvent.done(node))
        for _ in range(20):
            await pilot.pause()

        src = app._chat_source()
        check("工具调用后新一轮思考仍有标题", src.count("思考过程") == 2)
        check("第二轮思考带引用前缀", "> \\> 注意：结果有冲突。" in src)
        check("正文里没有字面 `>>`", ">>" not in src)
        check("正文里没有字面 `> >`", "> >" not in src)
        check("工具卡片仍在两段之间",
              [type(b).__name__ for b in app._chat_blocks]
              == ["Markdown", "ToolCard", "Markdown"])

        quotes = list(app.query(MarkdownBlockQuote))
        nested = [
            q for q in quotes
            if any(isinstance(a, MarkdownBlockQuote) for a in q.ancestors)
        ]
        check("两个思考块各自成引用", len(quotes) == 2)
        check("没有嵌套引用控件（不再出现第二道竖线）", not nested)
        rendered = [
            str(p._content) for q in quotes for p in q.query(MarkdownParagraph)
        ]
        check("转义只影响源码、界面仍显示字面 `>`",
              any("> 注意：结果有冲突。" in t for t in rendered))
        await pilot.press("ctrl+c")


async def test_interrupted_node_kept():
    """生成中断：节点保留（带 ⚠ 标记与「继续」提示），不再整轮回滚。"""

    class InterruptedController(FakeController):
        def send_message(self, text, emit):
            node = self.tree.create_root(text, "前半句", "先想想", "中断标题", 3, 2)
            node.error = "Error code: 400 - Content Exists Risk"
            emit(ControllerEvent.node_created(node))
            emit(ControllerEvent.stream("前半句", "先想想"))
            emit(ControllerEvent.error(node.error))
            emit(ControllerEvent.done(node))
            return node

    ctrl = InterruptedController()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#chat-input", ChatInput)
        await pilot.press("写", "个", "长", "回", "答")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()

        source = app._chat_source()
        check("中断后正文仍在（部分回答未丢）", "前半句" in source)
        check("中断后提示已保存", "本轮已保存到当前节点" in source)
        check("中断原因显示在正文里", "Content Exists Risk" in source)
        check("API 报错提示：不含 emoji",
              "⚠️" not in source and "💾" not in source)
        check("API 报错提示：不再提示「继续」",
              "可直接输入「继续」接着生成" not in source)

        tree = app.query_one("#tree", Tree)
        labels = [str(tree.root.label)] + [str(n.label) for n in tree.root.children]
        check("树中保留中断节点", any("中断标题" in lb for lb in labels))
        check("树中不再有中断标记", not any("⚠" in lb for lb in labels))
        check("中断节点仍是当前节点", ctrl.tree.current_node is not None
              and ctrl.tree.current_node.error != "")

        # 节点视图不再渲染「生成中断/已保存可继续」提示，但部分回答仍在
        node_view = app._node_content(ctrl.tree.current_node)
        check("中断节点视图：不再显示中断提示",
              "生成中断" not in node_view and "本轮内容已保存" not in node_view)
        check("中断节点视图：仍保留部分回答", "前半句" in node_view)
        check("状态条等控件关闭 markup",
              all(not app.query_one(sel, Static)._render_markup
                  for sel in ("#usage-left", "#usage-center", "#usage-right",
                              "#import-popup")))

        # 确认弹窗里是模型生成的命令/文件内容，同样必须按纯文本渲染
        from mincli.tui.confirm import ConfirmScreen

        app.push_screen(ConfirmScreen("确认", MARKUP_BAD_TEXT))
        for _ in range(10):
            await pilot.pause()
        conf_statics = list(app.screen.query(Static))
        check("确认弹窗按纯文本渲染（不抛 MarkupError）",
              bool(conf_statics)
              and all(not w._render_markup for w in conf_statics))
        app.pop_screen()
        for _ in range(5):
            await pilot.pause()
        await pilot.press("ctrl+c")


async def test_interrupt_binding():
    """Esc / 忙碌时 Ctrl+C：触发控制器打断，空闲后恢复。"""
    import time

    class SlowController(FakeController):
        def __init__(self):
            super().__init__()
            self.interrupted = False
            self.started = False

        def interrupt(self):
            self.interrupted = True
            return True

        def send_message(self, text, emit):
            node = self.tree.create_root(text, "", "", "慢回答", 0, 0)
            emit(ControllerEvent.node_created(node))
            self.started = True
            for _ in range(250):
                if self.interrupted:
                    break
                time.sleep(0.02)
            emit(ControllerEvent.stream("部分", ""))
            node.assistant_msg = "部分"
            emit(ControllerEvent.done(node))
            return node

    ctrl = SlowController()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#chat-input", ChatInput)
        check("打断：ChatInput 有 escape 优先绑定",
              any(b.key == "escape" for b in ChatInput.BINDINGS))
        check("打断：App 有 Ctrl+C 与 escape 绑定",
              {b.key for b in ChatApp.BINDINGS} >= {"ctrl+c", "escape"})

        await pilot.press("写", "个")
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause()
            if app._turn_active and ctrl.started:
                break
        check("打断：生成中处于忙碌状态", app._turn_active)
        check("打断：输入框提示如何打断",
              "Esc" in str(inp.placeholder))

        await pilot.press("escape")
        for _ in range(60):
            await pilot.pause()
            if ctrl.interrupted:
                break
        check("Esc 触发控制器打断", ctrl.interrupted)

        for _ in range(120):
            await pilot.pause()
            if not app._turn_active:
                break
        check("打断后回到空闲状态", not app._turn_active)
        await pilot.press("ctrl+c")


async def test_quit_forces_exit_after_interrupt():
    """Ctrl+C：忙碌时首次只打断；已请求打断仍未结束时再次强制退出。"""
    ctrl = FakeController()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(30):
            await pilot.pause()
            if app.ctrl is ctrl:
                break
        app._turn_active = True

        exits: list = []
        real_exit = app.exit
        app.exit = lambda *a, **k: exits.append(True)  # type: ignore[method-assign]
        try:
            await app.action_quit()
            check("退出：忙碌时首次 Ctrl+C 只打断不退出", not exits)
            check("退出：首次 Ctrl+C 置位打断标志", ctrl.interrupt_pending)

            await app.action_quit()
            check("退出：再次 Ctrl+C 强制退出", bool(exits))
        finally:
            app.exit = real_exit  # type: ignore[method-assign]


def test_stream_segment_sealable():
    """封段边界判定：只在不会破坏 Markdown 结构的位置封段。"""
    from mincli.tui.app import stream_segment_sealable

    check("封段判定：行尾 + 无围栏 → 可封",
          stream_segment_sealable("第一行\n第二行\n"))
    check("封段判定：未到行尾 → 不可封",
          not stream_segment_sealable("没有换行"))
    check("封段判定：未闭合围栏 → 不可封",
          not stream_segment_sealable("正文\n```python\ncode\n"))
    check("封段判定：闭合围栏后 → 可封",
          stream_segment_sealable("正文\n```python\ncode\n```\n"))
    check("封段判定：列表项后 → 不可封",
          not stream_segment_sealable("正文\n- 列表项\n"))
    check("封段判定：有序列表项后 → 不可封",
          not stream_segment_sealable("正文\n3. 第三项\n"))
    check("封段判定：表格行后 → 不可封",
          not stream_segment_sealable("正文\n| a | b |\n"))
    check("封段判定：思考引用前缀不影响判定",
          stream_segment_sealable("> 思考第一行\n> 思考第二行\n"))


async def test_stream_segment_sealing():
    """超长流式输出自动封段：单段有界、内容不丢、引用不嵌套。"""
    from mincli.tui import app as appmod
    from mincli.tui.app import REASONING_HEADER_MD

    ctrl = FakeController()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app.ctrl is ctrl:
                break
        await app._chat_reset("")
        app._stream_active = True
        app._answer_started = True
        app._reasoning_open = False
        # 带换行的长思考（换行 → 有安全边界），约 12000 字
        piece = "这是一段很长的思考内容，用于验证封段逻辑是否生效，并确认内容不会丢失。\n"
        n_batches = 150
        for _ in range(n_batches):
            app._stream_buf_reasoning = piece
            await app._flush_stream_buffer()
            await pilot.pause()
        segs = [b for b in app._chat_blocks if isinstance(b, Markdown)]
        check("封段：长思考被切成多段", len(segs) > 1)
        check("封段：单段长度有界",
              all(len(s.source) <= appmod.STREAM_SEG_MAX_CHARS + 400 for s in segs))
        check("封段：段计数复位（当前段未超限）",
              app._stream_seg_chars <= appmod.STREAM_SEG_MAX_CHARS + len(piece))
        source = app._chat_source()
        check("封段：内容不丢（末批仍在）", source.count("验证封段逻辑是否生效") == n_batches)
        check("封段：思考标题可重复",
              source.count(REASONING_HEADER_MD) == len(segs))
        check("封段：无嵌套引用（无 >>）", ">>" not in source)


async def test_adaptive_flush_interval():
    """自适应刷新间隔：随刷新耗时上升、有上下限、快时回落。"""
    from mincli.tui.app import (
        FLUSH_INTERVAL_MAX, FLUSH_INTERVAL_MIN, ChatApp as _App,
    )

    ctrl = FakeController()
    app = _App(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app.ctrl is ctrl:
                break
        app._flush_cpu_ema = 0.0
        app._adapt_flush_interval(0.001)
        check("自适应刷新：极快刷新回落到下限",
              app._flush_interval == FLUSH_INTERVAL_MIN)
        app._adapt_flush_interval(0.5)
        check("自适应刷新：慢刷新拉长间隔",
              app._flush_interval > FLUSH_INTERVAL_MIN)
        for _ in range(20):
            app._adapt_flush_interval(1.0)
        check("自适应刷新：间隔有上限",
              app._flush_interval <= FLUSH_INTERVAL_MAX)
        for _ in range(40):
            app._adapt_flush_interval(0.001)
        check("自适应刷新：刷新变快后回落",
              app._flush_interval == FLUSH_INTERVAL_MIN)


async def test_mcp_background_ui():
    """MCP 后台连接：状态条显示「连接中」，日志转通知（不往 stdout 写）。"""

    class FakeMcp:
        """最小 MCP 客户端替身：连接状态由测试显式放行（模拟慢连接）。"""

        def __init__(self):
            self.connecting = True
            self.ok = False
            self.release = threading.Event()

        def wait_ready(self, timeout=None):
            # 卡住「连接中」状态，等测试检查完状态条再放行
            self.release.wait(timeout=5)
            self.connecting = False
            self.ok = True
            return True

        def tools(self):
            return []

        def tool_names(self):
            return set()

        def close(self):
            pass

    class McpController(FakeController):
        def __init__(self):
            super().__init__()
            self._mcp = FakeMcp()

        @property
        def mcp_started(self):
            return True  # 已启动：App 不会再调 start_mcp，避免起真实连接

    ctrl = McpController()
    app = ChatApp(controller=ctrl)
    notified = []
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(30):
            await pilot.pause()
            if app.ctrl is ctrl:
                break
        left = str(app.query_one("#usage-left", Static).content)
        check("MCP：连接中时状态条有提示", "MCP 连接中" in left)

        app.notify = lambda message, **kwargs: notified.append(message)
        app._show_mcp_log("连接 MCP server「demo」失败: boom，已跳过")
        check("MCP：连接日志转成通知", bool(notified) and "boom" in notified[0])

        ctrl._mcp.release.set()  # 放行后台等待，模拟连接完成
        left = ""
        for _ in range(60):
            await pilot.pause()
            left = str(app.query_one("#usage-left", Static).content)
            if not ctrl.mcp_connecting and "MCP 连接中" not in left:
                break
        check("MCP：就绪后状态条提示消失", "MCP 连接中" not in left)


async def test_first_view_deferred():
    """首帧不等节点渲染：挂载立即返回，节点内容随后补上。"""
    ctrl = FakeController()
    ctrl.tree.create_root("问题", "答案", "思考", "标题", 1, 1)
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        check("启动：挂载阶段首帧已出（不阻塞）", app.is_running)
        for _ in range(40):
            await pilot.pause()
            if "答案" in app._chat_source():
                break
        check("启动：首帧之后恢复当前节点内容", "答案" in app._chat_source())


async def test_tree_sidebar_ui():
    """多对话树：侧栏列表 / 徽标 / 主题 / 切换 / 草稿。"""
    ctrl = FakeController()
    ctrl.create_tree(system_tools=False, mcp_tools=["tavily_search"])   # 树 2
    ctrl.tree.create_root("二问", "二答", "", "标题二", 1, 1)
    ctrl.create_tree()                                                  # 树 3
    ctrl.tree.create_root("三问", "三答", "", "标题三", 1, 1)
    ctrl.switch_tree(2)
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app._tree_rows:
                break
        rows = app._tree_rows
        check("侧栏：三棵树各一行", len(rows) == 3 and [r.number for r in rows] == [1, 2, 3])
        check("侧栏：当前树那一行高亮", [r.has_class("active") for r in rows] == [False, True, False])
        check("侧栏：行标签是「编号 对话」", str(rows[1]._name.content).strip() == "2 对话")
        check("侧栏：色块用该树的颜色",
              rows[1]._swatch.styles.background is not None)
        list_w = app.query_one("#tree-list", VerticalScroll)
        check("侧栏：那一栏不超过 3 行（加标题行共 4 行）", list_w.region.height <= 3)
        check("侧栏：三行都完整可见",
              all(r.region.height == 1 and list_w.region.contains_region(r.region)
                  for r in rows))
        check("顶部：徽标显示当前树", "树 2" in str(app.query_one("#tree-badge", Static).content))
        check("主题：随当前树切换", app.theme == "mincli-tree-2")

        # Ctrl+3 → 切到树 3
        await pilot.press("ctrl+3")
        for _ in range(20):
            await pilot.pause()
            if ctrl.active_number == 3 and "三答" in app._chat_source():
                break
        check("快捷键：Ctrl+3 切到树 3", ctrl.active_number == 3)
        check("切换后显示该树节点内容", "三答" in app._chat_source())
        check("切换后主题跟着换", app.theme == "mincli-tree-3")
        check("切换后徽标跟着换", "树 3" in str(app.query_one("#tree-badge", Static).content))

        # 草稿按树隔离
        inp = app.query_one("#chat-input", ChatInput)
        inp.load_text("树 3 写到一半")
        await pilot.pause()
        app._save_draft()
        await app._switch_tree(2)
        for _ in range(10):
            await pilot.pause()
        check("草稿：切走后再回来是空的（树 2 没写过）", inp.text == "")
        await app._switch_tree(3)
        for _ in range(10):
            await pilot.pause()
        check("草稿：切回树 3 恢复未发送内容", inp.text == "树 3 写到一半")

        # /tree 列表
        await app._handle_command("/tree")
        for _ in range(10):
            await pilot.pause()
        src = app._chat_source()
        check("命令：/tree 列出全部对话树", "树 1" in src and "树 2" in src and "树 3" in src)
        check("命令：/tree 标注当前树", "（当前）" in src)
        check("命令：/tree 显示挂载的能力", "无系统工具" in src or "外置工具" in src)

        # 超过 3 棵：这一栏高度不变，改为内部滚动
        ctrl.create_tree()
        ctrl.create_tree()
        app._refresh_tree_ui()
        for _ in range(20):
            await pilot.pause()
            if len(app._tree_rows) == 5:
                break
        check("侧栏：5 棵树都在列表里", len(app._tree_rows) == 5)
        list_w = app.query_one("#tree-list", VerticalScroll)
        check("侧栏：5 棵树时高度仍是 3 行", list_w.region.height == 3)
        check("侧栏：5 棵树时改为内部滚动", list_w.max_scroll_y > 0)

        # 树少时这一栏收缩（2 棵 = 2 行，1 棵 = 1 行）
        for expected in (2, 1):
            while len(ctrl.tree_numbers()) > expected:
                ctrl.delete_tree(ctrl.tree_numbers()[-1])
            app._refresh_tree_ui()
            for _ in range(20):
                await pilot.pause()
                if len(app._tree_rows) == expected:
                    break
            check(f"侧栏：{expected} 棵树时那一栏收缩到 {expected} 行",
                  app.query_one("#tree-list", VerticalScroll).region.height == expected)


async def test_tree_wizard_bootstrap():
    """没有树时启动：弹出建树向导，选完能力后建树并进入。"""
    from mincli.tui.tree_wizard import TreeWizardScreen

    ctrl = FakeController(with_tree=False)
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 35)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, TreeWizardScreen):
                break
        check("无树：启动即弹建树向导", isinstance(app.screen, TreeWizardScreen))
        check("无树：输入框提示先建树",
              "请先新建对话树" in str(app.query_one("#chat-input", ChatInput).placeholder))
        check("无树：徽标提示未创建", "未创建对话树" in str(app.query_one("#tree-badge", Static).content))
        check("无树：侧栏那一栏只剩标题 1 行",
              app.query_one("#tree-list", VerticalScroll).region.height == 0)

        # 取消勾选系统工具，然后确定
        screen = app.screen
        screen.query_one("#cap-system").value = False
        await pilot.pause()
        screen.query_one("#wizard-ok", Button).press()
        for _ in range(20):
            await pilot.pause()
            if ctrl.has_trees:
                break
        check("向导：确定后建出编号 1 的树", ctrl.tree_numbers() == [1])
        check("向导：能力按勾选结果挂载", ctrl.tree_caps(1)["system_tools"] is False)
        check("向导：关闭后回到主界面", not isinstance(app.screen, TreeWizardScreen))
        check("向导：建树后输入框提示恢复",
              "请先新建对话树" not in str(app.query_one("#chat-input", ChatInput).placeholder))


async def test_tree_wizard_cancel_exits():
    """没有树时取消建树向导 → 退出程序（不允许停在无法使用的界面）。"""
    from mincli.tui.tree_wizard import TreeWizardScreen

    ctrl = FakeController(with_tree=False)
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 35)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, TreeWizardScreen):
                break
        check("取消：向导已弹出", isinstance(app.screen, TreeWizardScreen))
        await pilot.press("escape")
        for _ in range(20):
            await pilot.pause()
            if not app.is_running:
                break
        check("取消：程序退出", not app.is_running)
        check("取消：没有建出任何树", not ctrl.has_trees)


async def test_tree_wizard_mcp_refresh():
    """建树向导开着时 MCP 连完：占位提示换成真实的工具复选框。"""
    from mincli.tui.tree_wizard import TreeWizardScreen

    class FakeMcp:
        connecting = True
        ready = False
        ok = False

        def tools(self):
            return []

        def tool_names(self):
            return set()

        def tool_owner(self, name):
            return None

        def tools_by_server(self):
            return {}

        def configured_servers(self):
            return ["tavily"]

        def server_status(self):
            return {}

        def reload(self):
            pass

        def cancel_running(self):
            return True

        def close(self):
            pass

    ctrl = FakeController(with_tree=False)
    ctrl._mcp = FakeMcp()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 35)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, TreeWizardScreen):
                break
        screen = app.screen
        check("向导：MCP 未就绪时显示占位提示",
              "正在连接" in str(screen.query_one("#wizard-hint", Static).content))

        # 模拟后台连接完成，并触发 UI 侧的就绪回调
        ctrl._mcp.connecting = False
        ctrl._mcp.ready = True
        ctrl._mcp.ok = True
        ctrl._mcp.tools_by_server = lambda: {
            "mincli": ["read_file"], "tavily": ["tavily_search"]
        }
        app._on_mcp_ready_ui()
        for _ in range(10):
            await pilot.pause()
        hint = str(screen.query_one("#wizard-hint", Static).content)
        check("向导：连完后占位提示消失", "正在连接" not in hint)
        check("向导：外置工具复选框自动补上", "tavily_search" in screen._boxes)
        check("向导：内置工具不单独列出（归入系统工具）", "read_file" not in screen._boxes)


async def test_tree_background_events():
    """后台那棵树的事件不渲染到当前消息区，只提示完成。"""
    ctrl = FakeController()
    ctrl.create_tree()          # 树 2
    ctrl.switch_tree(2)
    app = ChatApp(controller=ctrl)
    notified = []
    async with app.run_test(size=(100, 30)) as pilot:
        app.notify = lambda message, **kwargs: notified.append(message)
        for _ in range(10):
            await pilot.pause()
        before = app._chat_source()
        ev = ControllerEvent.stream("别的树的流式内容", "")
        ev.tree = 1
        await app._handle_event(ev)
        await pilot.pause()
        check("后台事件：流式内容不串进当前消息区",
              "别的树的流式内容" not in app._chat_source())
        done = ControllerEvent.done(None)
        done.tree = 1
        await app._handle_event(done)
        for _ in range(10):
            await pilot.pause()
        check("后台事件：完成时给出提示", any("对话树 1" in m and "完成" in m for m in notified))
        check("后台事件：当前视图未被改动", app._chat_source() == before)

        # 树 1 在后台生成期间，当前树（2）不允许发送
        notified.clear()
        app._turn_active = True
        ctrl._gen_tree = 1
        await app._send_user_text("趁它忙我也要发")
        await pilot.pause()
        check("后台生成期间：当前树发送被拒绝",
              any("正在生成" in m for m in notified))
        check("后台生成期间：消息没有进入会话",
              "趁它忙我也要发" not in app._chat_source())


async def main() -> int:
    print("== ChatApp headless 验证（2b） ==")
    test_command_spec()
    await test_command_completion_levels()
    test_markdown_safety()
    test_selection_safety()
    test_screen_forward_safety()
    test_tool_args_width()
    test_packaging()
    test_markup_safety()
    test_quote_marker_safety()
    fake = FakeController()
    app = ChatApp(controller=fake)
    async with app.run_test(size=(100, 30)) as pilot:
        # --- 1. 布局 ---
        check("布局：输入框存在且聚焦", app.query_one("#chat-input", ChatInput) is not None)
        check("布局：消息流存在", bool(app.query_one("#chat-log", VerticalScroll)))
        check("布局：会话树存在", bool(app.query_one("#tree", Tree)))
        check("布局：状态条存在", bool(app.query_one("#usage-bar", Horizontal)))
        check("布局：状态条三分栏", bool(app.query_one("#usage-left", Static))
              and bool(app.query_one("#usage-center", Static))
              and bool(app.query_one("#usage-right", Static)))
        for _ in range(10):
            await pilot.pause()
        usage_left = str(app.query_one("#usage-left", Static).content)
        check("状态条左栏显示缓存/余额", "缓存" in usage_left)
        check("启动：空会话显示欢迎页", "DeepSeek 树状对话 TUI" in app._chat_source())

        # --- 2. 发送消息 → 流式渲染 ---
        inp = app.query_one("#chat-input", ChatInput)
        await pilot.press("你", "好")
        check("输入中文正常", inp.text == "你好")
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
            if "世界！" in app._chat_source():
                break
        chat = app._chat_source()
        check("流式内容已追加", "你好，世界！" in chat)
        # 思考过程不折叠：完整内容以灰色块引用显示（无点击展开交互）
        check("思考以引用格式显示", "思考过程" in chat and "> 思考中" in chat)
        check("token 统计已显示", "tokens" in chat)

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
        chat = app.query_one("#chat-log", VerticalScroll)
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
        check("切换节点显示内容", "测试标题" in app._chat_source())

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
            if "**帮助**" in app._chat_source():
                break
        check("命令：/help 显示帮助", "**帮助**" in app._chat_source()
              and "/set thinking <on|off>" in app._chat_source())

        await type_command("/set model pro")
        await pilot.pause()
        check("命令：/set model pro", fake.current_model.endswith("pro"))

        await type_command("/tree")
        for _ in range(10):
            await pilot.pause()
            if "main: 测试标题" in app._chat_source():
                break
        check("命令：/tree 显示对话树", "main: 测试标题" in app._chat_source())

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
        check("补全：Tab 补全为 /delete（带参数命令补空格）", inp.text == "/delete ")
        check("补全：补全后转为命令提示", "/delete <节点ID> [...]" in str(body.content))

        await pilot.press("a", "1")
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
            if "当前配置" in app._chat_source():
                break
        check("命令：/set show 显示配置", "当前配置" in app._chat_source())

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
        hint = app.query_one("#usage-center", Static)
        check("导入提示显示在状态条中段", "已导入 1 个文件" in str(hint.content) and hint.has_class("visible"))

        # 一次导入多个文件（图片 + 文本）+ 悬停弹窗（完整文件名列表）
        await type_command("/import clear")
        for _ in range(10):
            await pilot.pause()
        await type_command(f"/import {_tpng} {_txt}")
        for _ in range(10):
            await pilot.pause()
        check("命令：/import 多文件", len(fake.pending_images) == 1 and len(fake.imported_files) == 1)
        check("状态条中段数量与前2文件名", "已导入 2 个文件" in str(hint.content) and "note.txt" in str(hint.content))
        # 悬停弹窗（固定显示在状态条上方，完整文件名列表）；移出中段自动消失
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
              and not app.query_one("#usage-center", Static).has_class("visible"))

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

        # --- 7.6c Windows 反斜杠路径（跨平台解析回归） ---
        # 在 POSIX 上创建文件名含「C:\...」的真实文件，模拟 Windows 拖入/粘贴路径：
        # POSIX 版 shlex 会把反斜杠当转义吃掉 → 导入失效；修复后必须原样识别并导入。
        if os.name != "nt":
            _win_txt = os.path.join(_tmp_img, r"C:\Users\me\notes.txt")
            with open(_win_txt, "w", encoding="utf-8") as f:
                f.write("windows style path")
            check("拖入：Windows 反斜杠路径被识别为可导入",
                  ChatInput._paths_from_paste(_win_txt) == [_win_txt])
            app.post_message(events.Paste(_win_txt))
            for _ in range(20):
                await pilot.pause()
            check("拖入：Windows 反斜杠路径成功导入文本",
                  any("notes.txt" in f["name"] for f in fake.imported_files))
            await type_command("/import clear")
            for _ in range(10):
                await pilot.pause()
            # /import 命令参数解析同样需保留反斜杠（此前会被 shlex 吃掉）
            await app._handle_command(f"/import {_win_txt}")
            for _ in range(10):
                await pilot.pause()
            check("命令：/import 支持 Windows 反斜杠路径",
                  any("notes.txt" in f["name"] for f in fake.imported_files))
            await type_command("/import clear")
            for _ in range(10):
                await pilot.pause()

        await type_command("/set detail low")
        await pilot.pause()
        check("命令：/set detail low", fake.image_detail == "low")

        await type_command("/set model vision")
        await pilot.pause()
        check("命令：/set model vision 映射为 flash", fake.current_model == "deepseek-flash")
        await type_command("/set model pro")
        await pilot.pause()
        check("命令：/set model pro", fake.current_model == "deepseek-v4-pro")
        await type_command("/set model flash")
        await pilot.pause()
        check("命令：/set model flash", fake.current_model == "deepseek-flash")

        await type_command("/files list")
        for _ in range(10):
            await pilot.pause()
        src_files = app._chat_source()
        check("命令：/files list 表格带序号与最新在前",
              "已上传图片文件" in src_files and "| 1 |" in src_files
              and src_files.index("tui.png") < src_files.index("old.jpg"))
        check("命令：/files list 显示容量与配额",
              "合计" in src_files and "配额 10000 个 / 25 GiB" in src_files)
        check("命令：/files list 提示按序号操作",
              "/files delete <ID|序号>" in src_files and "/files clean" in src_files)

        await type_command("/files list 1")
        for _ in range(10):
            await pilot.pause()
        src_one = app._chat_source()
        check("命令：/files list N 限制条数",
              "共 1 个（最新在前）" in src_one and "还有更早的文件未列出" in src_one)
        await type_command("/clear")
        await type_command("/files list")
        for _ in range(10):
            await pilot.pause()

        await type_command("/files info 1")
        for _ in range(10):
            await pilot.pause()
        src_info = app._chat_source()
        check("命令：/files info 按序号查询",
              "文件信息（Files API）" in src_info and "tui.png" in src_info
              and "2 KiB" in src_info)
        check("命令：/files info 显示永久有效", "永久有效" in src_info)

        await type_command("/files delete 2")
        for _ in range(10):
            await pilot.pause()
        check("命令：/files delete 按序号删除",
              [f["id"] for f in fake.client.files.items] == ["file-api-tui"])

        await type_command("/files clean")
        for _ in range(20):
            await pilot.pause()
            if app.screen.query("#confirm-yes"):
                break
        check("命令：/files clean 需确认", bool(app.screen.query("#confirm-yes")))
        app.screen.query_one("#confirm-yes", Button).press()
        for _ in range(10):
            await pilot.pause()
        check("命令：/files clean 删除未引用文件", fake.client.files.items == [])

        # 节点视图：带图片的节点渲染占位（直接驱动 node_created 事件）
        node = fake.tree.create_root("看图", "", "", "图题", 0, 0)
        node.user_images = [ImageAttachment(source=_tpng, name="t.png", width=800, height=600)]
        fake.tree.current_node = node
        await app._handle_event(ControllerEvent.node_created(node))
        for _ in range(10):
            await pilot.pause()
        check("节点视图含图片占位", "[图片: t.png (800x600)]" in app._chat_source())
        await type_command("/clear")

        # --- 7.6c 多轮工具调用：多个思考块穿插正文，各自按引用格式显示 ---
        node_m = fake.tree.create_root("多轮问题", "", "", "多轮标题", 0, 0)
        fake.tree.current_node = node_m
        await app._handle_event(ControllerEvent.node_created(node_m))
        await app._handle_event(ControllerEvent.stream("", "第一轮思考内容"))
        await app._handle_event(ControllerEvent.stream("第一轮正文", ""))
        await app._handle_event(ControllerEvent.tool("execute_command", '{"command":"ls"}', ""))
        await app._handle_event(ControllerEvent.tool("execute_command", '{"command":"ls"}', "（完成）"))
        await app._handle_event(ControllerEvent.stream("", "第二轮思考内容"))
        await app._handle_event(ControllerEvent.stream("第二轮正文", ""))
        await app._handle_event(ControllerEvent.done(node_m))
        for _ in range(30):
            await pilot.pause()
        src_m = app._chat_source()
        check("多轮：两个思考块各带标题", src_m.count("思考过程") == 2)
        check("多轮：第一轮思考按引用显示", "> 第一轮思考内容" in src_m)
        check("多轮：第二轮思考按引用显示", "> 第二轮思考内容" in src_m)
        check("多轮：两轮正文都显示", "第一轮正文" in src_m and "第二轮正文" in src_m)
        check("多轮：工具卡片显示工具名", "▸ 工具调用：execute_command" in src_m)
        check("多轮：工具卡片参数逐行", "command=\"ls\"" in src_m and "    command=\"ls\"" in src_m)
        check("多轮：工具卡片状态完成", "状态：完成" in src_m)
        check("多轮：工具卡片作为控件穿插在正文段之间",
              [type(b).__name__ for b in app._chat_blocks] == ["Markdown", "ToolCard", "Markdown"])
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
        chat = app._chat_source()
        check("命令：/compact 新建摘要节点", fake.tree.compaction is not None)
        node_id = fake.tree.compaction["boundary_id"]
        check(
            "命令：/compact 显示摘要",
            "上下文压缩摘要" in chat and "TUI 压缩测试内容" in chat,
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
        check("删除：当前节点被删后自动跳转", "main: 根" in app._chat_source())
        tree_w = app.query_one("#tree", Tree)
        cursor_node = getattr(tree_w, "cursor_node", None)
        check("删除：树光标跟随新的当前节点", cursor_node is not None and cursor_node.data == "main")

        # --- 7.9 文字选择 + 复制 ---
        chat = app._chat_source()
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

    # --- 9. 启动即显示上次会话的当前节点（而非欢迎页） ---
    fake2 = FakeController()
    fake2.reset()
    fake2.tree.create_root("上次的问题", "上次的回答", "", "上次标题", 1, 1)
    app2 = ChatApp(controller=fake2)
    async with app2.run_test(size=(100, 30)) as pilot2:
        for _ in range(10):
            await pilot2.pause()
        src2 = app2._chat_source()
        check("启动：直接显示当前节点（非欢迎页）", "上次标题" in src2 and "DeepSeek 树状对话 TUI" not in src2)
        tree2 = app2.query_one("#tree", Tree)
        cursor2 = getattr(tree2, "cursor_node", None)
        check("启动：树光标跟随当前节点", cursor2 is not None and cursor2.data == "main")
        await pilot2.press("ctrl+c")

    # --- 10. 工作流（/wf）：保存 / 挂载 / 执行 / 列表 / 修订 / 管理 ---
    fake3 = FakeController()
    app3 = ChatApp(controller=fake3)
    async with app3.run_test(size=(100, 30)) as pilot3:
        for _ in range(5):
            await pilot3.pause()
        inp3 = app3.query_one("#chat-input", ChatInput)

        async def type_wf_cmd(cmd: str) -> None:
            inp3.clear()
            await pilot3.press(*list(cmd))
            await pilot3.press("enter")

        # 先发一条消息，成为可提炼的当前节点
        await type_wf_cmd("总结一下项目变更")
        for _ in range(30):
            await pilot3.pause()
            if fake3.tree.current_node is not None:
                break
        check("工作流：准备当前节点成功", fake3.tree.current_node is not None
              and fake3.tree.current_node.user_msg == "总结一下项目变更")

        # /wf save（提炼被 stub）
        await type_wf_cmd("/wf save demo")
        for _ in range(40):
            await pilot3.pause()
            if fake3.wf_get("demo") is not None:
                break
        wf_demo = fake3.wf_get("demo")
        check("工作流：/wf save 提炼保存", wf_demo is not None
              and wf_demo.doc.startswith("目标：测试工作流"))
        check("工作流：来源节点与变量", wf_demo is not None
              and wf_demo.source_nodes == ["main"] and wf_demo.placeholders() == ["path"])

        # /wf list / show
        await type_wf_cmd("/wf list")
        for _ in range(12):
            await pilot3.pause()
            if "工作流列表" in app3._chat_source():
                break
        check("工作流：/wf list 显示列表", "工作流列表" in app3._chat_source()
              and "demo" in app3._chat_source())
        await type_wf_cmd("/wf show demo")
        for _ in range(12):
            await pilot3.pause()
            if "目标：测试工作流" in app3._chat_source():
                break
        check("工作流：/wf show 显示规范", "目标：测试工作流" in app3._chat_source())

        # /wf use：状态条中段提示 → 下一条消息按工作流合成执行 → 一次性解除
        await type_wf_cmd("/wf use demo")
        for _ in range(8):
            await pilot3.pause()
        center3 = app3.query_one("#usage-center", Static)
        check("工作流：挂载提示显示于状态条中段",
              center3.has_class("visible") and "工作流已挂载：demo" in str(center3.content))
        await type_wf_cmd("请针对 notes/a.txt 执行")
        for _ in range(30):
            await pilot3.pause()
            cur = fake3.tree.current_node
            if cur is not None and "请执行工作流「demo」" in cur.user_msg:
                break
        cur3 = fake3.tree.current_node
        check("工作流：下一条消息按工作流合成执行",
              cur3 is not None and "请执行工作流「demo」" in cur3.user_msg
              and "本次输入：请针对 notes/a.txt 执行" in cur3.user_msg)
        check("工作流：挂载一次性解除", app3._pending_wf is None
              and not center3.has_class("visible"))

        # /wf run：无需再输入，立即执行（位置参数按变量顺序填充）
        await type_wf_cmd("/wf run demo notes/b.txt")
        for _ in range(30):
            await pilot3.pause()
            cur = fake3.tree.current_node
            if cur is not None and "请执行工作流「demo」" in cur.user_msg:
                break
        cur3b = fake3.tree.current_node
        check("工作流：/wf run 立即执行", cur3b is not None
              and "请执行工作流「demo」" in cur3b.user_msg
              and "notes/b.txt" in cur3b.user_msg and "本次输入：" not in cur3b.user_msg)

        # /wf 参数里的 Windows 反斜杠路径不能被 POSIX shlex 吃掉（同类解析修复）
        await app3._handle_command(r"/wf run demo C:\work\b.txt")
        for _ in range(30):
            await pilot3.pause()
            cur = fake3.tree.current_node
            if cur is not None and "C:\\work\\b.txt" in cur.user_msg:
                break
        cur3c = fake3.tree.current_node
        check("工作流：/wf run 支持 Windows 反斜杠路径参数",
              cur3c is not None and r"C:\work\b.txt" in cur3c.user_msg
              and "C:workb.txt" not in cur3c.user_msg)

        # /wf edit 带修改要求 → 模型修订并落盘
        await type_wf_cmd("/wf edit demo 增加一条校验步骤")
        for _ in range(30):
            await pilot3.pause()
            wf = fake3.wf_get("demo")
            if wf is not None and wf.doc.startswith("目标：已按修改要求更新的工作流"):
                break
        check("工作流：/wf edit 修订保存",
              fake3.wf_get("demo").doc.startswith("目标：已按修改要求更新的工作流"))

        # /wf rename / delete（确认弹窗）
        await type_wf_cmd("/wf rename demo demo2")
        for _ in range(8):
            await pilot3.pause()
        check("工作流：/wf rename 生效", fake3.wf_get("demo") is None
              and fake3.wf_get("demo2") is not None)
        await type_wf_cmd("/wf delete demo2")
        for _ in range(20):
            await pilot3.pause()
            if app3.screen.query("#confirm-yes"):
                break
        check("工作流：删除需确认", bool(app3.screen.query("#confirm-yes")))
        app3.screen.query_one("#confirm-yes", Button).press()
        for _ in range(10):
            await pilot3.pause()
        check("工作流：确认后已删除", fake3.wf_get("demo2") is None)

        # 错误路径不崩溃
        await type_wf_cmd("/wf run 不存在的工作流")
        for _ in range(5):
            await pilot3.pause()
        check("工作流：运行不存在的工作流给出警告且不崩溃", app3.screen is not None
              and fake3.tree.current_node is not None)

        await pilot3.press("ctrl+c")

    await test_reasoning_quote_after_tool()
    await test_interrupted_node_kept()
    await test_interrupt_binding()
    await test_quit_forces_exit_after_interrupt()
    test_stream_segment_sealable()
    await test_stream_segment_sealing()
    await test_adaptive_flush_interval()
    await test_mcp_background_ui()
    await test_first_view_deferred()
    await test_tree_sidebar_ui()
    await test_tree_wizard_bootstrap()
    await test_tree_wizard_cancel_exits()
    await test_tree_wizard_mcp_refresh()
    await test_tree_background_events()

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


def test_tool_args_width():
    """工具参数格式化：长 JSON 逐行显示、单行宽度受限、不产生超宽单行。"""
    import json

    from mincli.tui.app import ChatApp

    args = json.dumps(
        {
            "query": "淘宝闪购 饿了么 关系 2025",
            "results": [{"url": "https://example.com/" + "x" * 60, "title": "标题"}],
        },
        ensure_ascii=False,
    )
    fmt = ChatApp._format_tool_args(args)
    check("工具参数换行（多行）", "\n" in fmt)
    check("工具参数逐键一行", fmt.split("\n")[0].startswith("query="))
    check("工具参数单行受限", all(len(line) <= 110 for line in fmt.split("\n")))
    # 非 dict（如普通字符串）不影响，直接透传不崩溃
    check("工具参数非 dict 不崩", isinstance(ChatApp._format_tool_args('"just a string"'), str))


def test_command_spec():
    """命令规格引擎：分级补全、别名、动态候选、用法与 /help 同源（纯逻辑，不起 App）。"""
    prov = {
        "trees": lambda: [("1", "对话树 1（3 个节点）（当前）"), ("2", "对话树 2（0 个节点）")],
        "workflows": lambda: [("demo", "示例工作流（2 步）")],
        "servers": lambda: [("tavily", "已配置的 MCP server")],
    }

    c = complete("/", prov)
    check("规格：/ 列出全部一级命令", c is not None and len(c.candidates) == len(CMD_SPEC))

    c = complete("/set", prov)
    check("规格：/set 列出二级命令", c is not None and len(c.candidates) == 10)
    check("规格：二级候选项带用法与说明",
          any(x.usage == "/set temp <值>" and "温度" in x.desc for x in c.candidates))
    check("规格：含子命令的候选项补空格", any(x.insert == "/set thinking " for x in c.candidates))
    check("规格：叶子候选项补空格（还要填参数）", any(x.insert == "/set show" for x in c.candidates))

    c = complete("/set thinking ", prov)
    check("规格：三级取值枚举", c is not None and {x.usage for x in c.candidates} ==
          {"/set thinking on", "/set thinking off"})
    check("规格：三级标题回显参数", c is not None and c.title == "/set thinking <on|off>")

    c = complete("/set thinking of", prov)
    check("规格：三级前缀过滤", c is not None and [x.insert for x in c.candidates] == ["/set thinking off"])

    c = complete("/set te", prov)
    check("规格：二级前缀过滤", c is not None and [x.insert for x in c.candidates] == ["/set temp "])

    c = complete("/tree ", prov)
    check("规格：动态候选（对话树编号）",
          c is not None and any(x.usage == "/tree 2" and "对话树 2" in x.desc for x in c.candidates))
    c = complete("/tree 2 ", prov)
    check("规格：动态候选自身还有下一级", c is not None and [x.usage for x in c.candidates] == ["/tree 2 tools"])
    c = complete("/tree delete ", prov)
    check("规格：子命令后的动态候选", c is not None and
          [x.insert for x in c.candidates] == ["/tree delete 1", "/tree delete 2"])
    c = complete("/wf show ", prov)
    check("规格：动态候选（工作流名）", c is not None and [x.insert for x in c.candidates] == ["/wf show demo"])
    c = complete("/mcp remove ", prov)
    check("规格：动态候选（MCP server 名）", c is not None and
          [x.insert for x in c.candidates] == ["/mcp remove tavily"])

    c = complete("/workflow list", prov)
    check("规格：别名按本体归位", c is not None and c.title == "/wf list")

    c = complete("/set temp", prov)
    check("规格：叶子命令转提示（无候选）", c is not None and not c.candidates and c.title == "/set temp <值>")
    check("规格：未知命令不弹窗", complete("/zzz", prov) is None)
    check("规格：非命令不弹窗", complete("你好", prov) is None)
    # 动态来源不可用时不崩
    def boom():
        raise RuntimeError("控制器未就绪")
    c = complete("/tree ", {"trees": boom})
    check("规格：动态来源异常不崩且退回静态候选",
          c is not None and [x.usage for x in c.candidates] == ["/tree new", "/tree delete <编号>"])

    check("用法：含子命令给名称清单", usage_line("/set").endswith("（system/temp/model/thinking/effort/audit/workspace/detail/file_confirm/show）"))
    check("用法：取值枚举直接写全", usage_line("/set detail") == "用法: /set detail <low|auto|high|original>")
    check("用法：叶子给完整参数", usage_line("/wf show") == "用法: /wf show <名>")

    help_text = help_markdown()
    check("帮助：一级命令全部出现", all(f"/{c0.name}" in help_text for c0 in CMD_SPEC))
    check("帮助：二级命令分行列出", "/set thinking <on|off>" in help_text and "/tree <编号> tools" in help_text)
    def _has_emoji(text: str) -> bool:
        # emoji / dingbat / 箭头符号等图形字符区间（AGENTS.md 禁止在交付物引入）
        return any(
            0x1F000 <= ord(ch) <= 0x1FAFF
            or 0x2600 <= ord(ch) <= 0x27BF
            or 0x2B00 <= ord(ch) <= 0x2BFF
            or 0x2190 <= ord(ch) <= 0x21FF
            or 0xFE00 <= ord(ch) <= 0xFE0F
            for ch in text
        )
    check("帮助：无 emoji", not _has_emoji(help_text))


async def test_command_completion_levels():
    """命令补全交互：Tab 循环时滚动跟随、Shift+Tab 反向、Enter 只补全不执行。"""
    fake = FakeController()
    fake.create_tree()  # 树 2，供动态候选使用
    app = ChatApp(controller=fake)
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#chat-input", ChatInput)
        popup = app.query_one("#cmd-popup")
        body = app.query_one("#cmd-popup-body", Static)

        async def type_text(text: str) -> None:
            inp.clear()
            for _ in range(4):
                await pilot.pause()
            await pilot.press(*list(text))
            for _ in range(6):
                await pilot.pause()

        def popup_text() -> str:
            return str(body.content)

        await type_text("/")
        text = popup_text()
        check("补全：/ 每行一个候选（用法 + 灰色说明）", "/exit" in text and "/wf" in text)
        check("补全：灰字统一为说明（不再混入用法行）", "用法:" not in text)
        check("补全：输入 / 时无高亮项也能滚到第一项", popup.scroll_y == 0)
        check("补全：标题给出候选序号", "第 1/18 项" in text)

        # 18 个一级命令在 30 行终端里超出弹窗可视高度：Tab 必须把高亮项滚进视野
        indices = []
        for _ in range(len(app._completion.candidates)):
            await pilot.press("tab")
            for _ in range(2):
                await pilot.pause()
            index = app._completion_index
            row = 1 + index
            visible = popup.scroll_y <= row < popup.scroll_y + popup.content_size.height
            indices.append((index, popup.scroll_y, visible))
        # 首次 Tab 从第 0 项移到第 1 项，转一圈后回到第 0 项
        expected = list(range(1, len(CMD_SPEC))) + [0]
        check("补全：Tab 循环覆盖全部候选", [i for i, _, _ in indices] == expected)
        check("补全：Tab 时高亮项滚入可视区", all(vis for _, _, vis in indices))
        check("补全：确实发生了滚动（不再固定不动）", max(sy for _, sy, _ in indices) > 0)

        await pilot.press("shift+tab")
        for _ in range(2):
            await pilot.pause()
        # 循环结束时高亮在第 0 项，反向一次应回到最后一项
        check("补全：Shift+Tab 反向切换", app._completion_index == len(CMD_SPEC) - 1)
        check("补全：序号随高亮更新", f"第 {len(CMD_SPEC)}/{len(CMD_SPEC)} 项" in popup_text())

        await type_text("/set")
        text = popup_text()
        check("补全：/set 分行列出二级命令", text.count("\n") == 10)
        check("补全：二级候选带用法与说明", "→ /set system <提示词>" in text and "修改系统提示词" in text)

        await pilot.press("tab")
        for _ in range(3):
            await pilot.pause()
        check("补全：Tab 选中第二项", app._completion.candidates[app._completion_index].insert == "/set temp ")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        check("补全：Enter 只补全不执行 /set", inp.text == "/set temp ")
        check("补全：Enter 后弹窗转为该命令提示", "/set temp <值>" in popup_text())

        await type_text("/set thinking ")
        check("补全：三级取值枚举候选", "→ /set thinking on" in popup_text()
              and "/set thinking off" in popup_text())

        await type_text("/tree ")
        text = popup_text()
        check("补全：动态列出对话树编号", "/tree 1" in text and "/tree 2" in text
              and "树 2" in text.replace("对话树 2", "树 2"))
        await type_text("/tree 2 ")
        check("补全：编号后还有三级命令", "→ /tree 2 tools" in popup_text())

        await type_text("/zzz")
        check("补全：未知命令不弹窗", not popup.has_class("visible"))
        await pilot.press("ctrl+c")


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
