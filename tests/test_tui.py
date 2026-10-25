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

from mincli.config import Config
from mincli.controller import ChatController, ControllerEvent
from mincli.tui.app import AttachmentViewScreen, ChatApp
from mincli.tui.commands import CMD_SPEC, complete, help_markdown, usage_line
from mincli.tui.widgets import ChatInput
from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Markdown, Static, TextArea, Tree
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
    """记录并打印一条断言；在 pytest 下失败即抛出，避免测试静默通过。"""
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
        return
    FAIL += 1
    print(f"  FAIL  {name}")
    if "PYTEST_CURRENT_TEST" in os.environ:
        raise AssertionError(name)


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
    """真实 ChatController + 固定事件流（不联网）。

    配置与提示词都落在临时目录，并同步设置 MINCLI_CONFIG_PATH /
    MINCLI_PROMPT_PATH，避免界面开关写回用户主目录。
    """

    def __init__(
        self,
        with_tree: bool = True,
        trees_dir: str | None = None,
        files_mode: str = "off",
        web_enabled: bool = True,
    ):
        cfg_dir = tempfile.mkdtemp(prefix="mincli_tui_cfg_")
        self.CONFIG_PATH = os.path.join(cfg_dir, "config.toml")
        self.PROMPT_PATH = os.path.join(cfg_dir, "prompt.md")
        self.ATTACHMENT_DIR = os.path.join(cfg_dir, "attachments")
        os.environ["MINCLI_CONFIG_PATH"] = self.CONFIG_PATH
        os.environ["MINCLI_PROMPT_PATH"] = self.PROMPT_PATH
        os.environ["MINCLI_ATTACHMENTS_PATH"] = self.ATTACHMENT_DIR
        # 每个控制器一个独立树目录：树会落盘，共享目录会让用例互相污染
        self.TREES_DIR = trees_dir or tempfile.mkdtemp(prefix="mincli_tui_trees_")
        self.config = Config(
            files_mode=files_mode,
            web_enabled=web_enabled,
            attachment_dir=self.ATTACHMENT_DIR,
            prompt_file=self.PROMPT_PATH,
        )
        super().__init__(
            client=_FakeClient([]),
            config=self.config,
            default_system="sys",
            trees_dir=self.TREES_DIR,
        )
        # 默认先建一棵树：没有树时 App 启动会显示欢迎页，绝大多数用例不测这个
        if with_tree and not self.has_trees:
            self.create_tree()
        self.saved = False
        self.closed = False

    def send_message(self, text, emit, attachments=None):
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
        def send_message(self, text, emit, attachments=None):
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
        head, rest = app._node_view_parts(ctrl.tree.current_node)
        node_view = head + rest
        check("中断节点视图：不再显示中断提示",
              "生成中断" not in node_view and "本轮内容已保存" not in node_view)
        check("中断节点视图：仍保留部分回答", "前半句" in node_view)
        check("状态条等控件关闭 markup",
              all(not app.query_one(sel, Static)._render_markup
                  for sel in ("#usage-left", "#usage-right",
                              "#footer-hint", "#footer-file", "#footer-web")))

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

        def send_message(self, text, emit, attachments=None):
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


async def test_welcome_page():
    """没有树时启动：聊天区显示欢迎页，点击新建按钮后创建树并消失。"""
    ctrl = FakeController(with_tree=False)
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 40)) as pilot:
        for _ in range(60):
            await pilot.pause()
            if app.query("#welcome"):
                break
        check("无树：聊天区显示欢迎页", bool(app.query("#welcome")))
        check("无树：欢迎页有新建按钮", bool(app.query("#welcome-new")))
        check("无树：输入框提示先建树",
              "请先新建对话树" in str(app.query_one("#chat-input", ChatInput).placeholder))
        check("无树：徽标提示未创建",
              "未创建对话树" in str(app.query_one("#tree-badge", Static).content))
        check("无树：此时没有任何对话树", not ctrl.has_trees)

        app.query_one("#welcome-new", Button).press()
        for _ in range(60):
            await pilot.pause()
            if ctrl.has_trees and not app.query("#welcome"):
                break
        check("欢迎页：按钮创建出编号 1 的树", ctrl.tree_numbers() == [1])
        check("欢迎页：创建后欢迎页消失", not app.query("#welcome"))
        check("欢迎页：输入框提示恢复",
              "请先新建对话树" not in str(app.query_one("#chat-input", ChatInput).placeholder))
        await pilot.press("ctrl+c")


async def test_ctrl_n_new_tree():
    """Ctrl+N：新建一棵对话树并切过去。"""
    ctrl = FakeController()
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 40)) as pilot:
        for _ in range(30):
            await pilot.pause()
            if app.ctrl is ctrl:
                break
        before = list(ctrl.tree_numbers())
        await pilot.press("ctrl+n")
        for _ in range(40):
            await pilot.pause()
            if len(ctrl.tree_numbers()) == len(before) + 1:
                break
        numbers = ctrl.tree_numbers()
        check("Ctrl+N：新建一棵对话树", len(numbers) == len(before) + 1)
        check("Ctrl+N：新树编号递增", numbers[-1] == before[-1] + 1)
        check("Ctrl+N：自动切到新树", ctrl.active_number == numbers[-1])
        await pilot.press("ctrl+c")


async def test_footer_reflects_config():
    """底部状态栏：显示当前文件四态/联网开关，点击与快捷键都能切换。"""
    ctrl = FakeController(files_mode="off", web_enabled=True)
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if "关闭" in str(app.query_one("#footer-file", Static).content):
                break
        footer_file = app.query_one("#footer-file", Static)
        footer_web = app.query_one("#footer-web", Static)
        footer_think = app.query_one("#footer-think", Static)
        check("状态栏：文件模式显示为关闭", "文件 关闭" in str(footer_file.content))
        check("状态栏：联网显示为开", "联网 开" in str(footer_web.content))
        check("状态栏：思考默认关闭", "思考 关" in str(footer_think.content))

        await pilot.click("#footer-file")
        for _ in range(30):
            await pilot.pause()
            if ctrl.config.files_mode == "read":
                break
        check("状态栏：点击文件开关切到下一态（关闭→只读）",
              ctrl.config.files_mode == "read")
        check("状态栏：文件文本跟随配置", "文件 只读" in str(footer_file.content))

        await pilot.click("#footer-web")
        for _ in range(30):
            await pilot.pause()
            if not ctrl.config.web_enabled:
                break
        check("状态栏：点击联网开关翻转", ctrl.config.web_enabled is False)
        check("状态栏：联网文本跟随配置", "联网 关" in str(footer_web.content))

        # 快捷键与点击共用同一套状态
        await pilot.press("ctrl+r")
        for _ in range(30):
            await pilot.pause()
            if ctrl.config.files_mode == "confirm":
                break
        check("状态栏：Ctrl+R 继续循环文件模式", ctrl.config.files_mode == "confirm")
        check("状态栏：写需确认文案", "文件 写需确认" in str(footer_file.content))
        await pilot.press("ctrl+w")
        for _ in range(30):
            await pilot.pause()
            if ctrl.config.web_enabled:
                break
        check("状态栏：Ctrl+W 翻转联网", ctrl.config.web_enabled is True)
        check("状态栏：联网重新显示为开", "联网 开" in str(footer_web.content))

        # 思考：快捷键、点击、命令三路一致
        await pilot.press("ctrl+t")
        for _ in range(30):
            await pilot.pause()
            if ctrl.config.thinking:
                break
        check("状态栏：Ctrl+T 开启思考", ctrl.config.thinking is True)
        check("状态栏：思考文本跟随配置", "思考 开" in str(footer_think.content))
        try:
            saved = open(ctrl.CONFIG_PATH, encoding="utf-8").read()
        except OSError:
            saved = ""
        check("状态栏：思考开关写回 config.toml", "thinking = true" in saved)
        await pilot.click("#footer-think")
        for _ in range(30):
            await pilot.pause()
            if not ctrl.config.thinking:
                break
        check("状态栏：点击思考开关关闭", ctrl.config.thinking is False)
        check("状态栏：思考重新显示为关", "思考 关" in str(footer_think.content))

        inp = app.query_one("#chat-input", ChatInput)
        inp.clear()
        for _ in range(4):
            await pilot.pause()
        await pilot.press(*list("/thinking on"))
        for _ in range(8):
            await pilot.pause()
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
            if ctrl.config.thinking:
                break
        check("命令：/thinking on 开启思考", ctrl.config.thinking is True)
        check("命令：底部状态栏同步", "思考 开" in str(footer_think.content))
        await pilot.press("ctrl+c")


def test_chip_display_is_inline_code():
    """附件引用块在聊天区回显为行内代码（不被当链接解析、不丢字）。"""
    out = ChatApp._chips_as_code("[DOCX a.docx] 看看")
    check("引用块：转换为反引号包裹的行内代码", "`[DOCX a.docx]`" in out)
    check("引用块：其余文字原样保留", out.endswith("看看") and out.startswith("`["))
    check("引用块：空输入不崩溃", ChatApp._chips_as_code("") == "")
    check("引用块：IMAGE/WEB 同样适用",
          "`[WEB example.com]`" in ChatApp._chips_as_code("[WEB example.com] 读一下"))


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
    ctrl.create_tree()                                                  # 树 2
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
        check("命令：/tree 显示每棵树的节点数", src.count("个节点") >= 3)
        check("命令：/tree 列出子命令", "/tree delete <编号>" in src)

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


async def test_delete_node_key():
    """侧栏选中节点后按 Delete：确认弹窗（自动确认）后删除该节点。"""
    ctrl = FakeController()
    root = ctrl.tree.create_root("根问题", "根回答", "", "根标题", 1, 1)
    child = ctrl.tree.add_child(root, "子问题", "子回答", "", "子标题", 1, 1)
    ctrl.tree.current_node = child
    app = ChatApp(controller=ctrl)

    def auto_confirm(title, text, on_result):
        # 不弹真实弹窗：直接以「确认」结果回调（_on_delete_confirmed 是协程）
        asyncio.ensure_future(on_result(True))

    async with app.run_test(size=(100, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._tree_rows:
                break
        app._ask_confirm = auto_confirm
        app._select_tree_node(child.id)
        for _ in range(10):
            await pilot.pause()
        tree_w = app.query_one("#tree", Tree)
        tree_w.focus()
        for _ in range(5):
            await pilot.pause()
        check("删除键：侧栏已选中子节点",
              tree_w.cursor_node is not None and tree_w.cursor_node.data == child.id)

        await pilot.press("delete")
        for _ in range(60):
            await pilot.pause()
            if child.id not in ctrl.tree.nodes:
                break
        check("删除键：按 Delete 后节点被删除", child.id not in ctrl.tree.nodes)
        check("删除键：根节点保留",
              ctrl.tree.root is not None and root.id in ctrl.tree.nodes)
        await pilot.press("ctrl+c")


async def test_attachment_buttons_and_view():
    """节点下方渲染附件按钮；点击后打开只读全文弹窗。"""
    ctrl = FakeController()
    node = ctrl.tree.create_root("看看附件", "好的", "", "附件节点", 1, 1)
    att_dir = tempfile.mkdtemp(prefix="mincli_tui_att_")
    stored = os.path.join(att_dir, "a.txt")
    with open(stored, "w", encoding="utf-8") as f:
        f.write("这是附件的完整正文。\n第二行。")
    node.attachments = [
        {"name": "a.txt", "kind": "document", "stored_path": stored, "chars": 15}
    ]
    ctrl.tree.current_node = node
    app = ChatApp(controller=ctrl)
    async with app.run_test(size=(100, 40)) as pilot:
        buttons: list[Button] = []
        for _ in range(60):
            await pilot.pause()
            buttons = [
                b for b in app.query(Button) if (b.id or "").startswith("att-")
            ]
            if buttons:
                break
        check("附件：节点下方出现附件按钮", len(buttons) == 1)
        if buttons:
            check("附件：按钮 id 以 att- 开头", buttons[0].id.startswith("att-"))
            check("附件：按钮文字是附件名", str(buttons[0].label) == "a.txt")
            await pilot.click(f"#{buttons[0].id}")
            for _ in range(40):
                await pilot.pause()
                if isinstance(app.screen, AttachmentViewScreen):
                    break
            check("附件：点击后打开 AttachmentViewScreen",
                  isinstance(app.screen, AttachmentViewScreen))
            if isinstance(app.screen, AttachmentViewScreen):
                text = app.screen.query_one("#attachment-view-text", TextArea).text
                check("附件：弹窗显示附件全文", "这是附件的完整正文" in text)
                await pilot.press("escape")
                for _ in range(20):
                    await pilot.pause()
                check("附件：Escape 关闭弹窗",
                      not isinstance(app.screen, AttachmentViewScreen))
        await pilot.press("ctrl+c")


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
        "trees": lambda: [("1", "对话树 1（当前）"), ("2", "对话树 2")],
    }

    c = complete("/", prov)
    check("规格：/ 列出全部一级命令", c is not None and len(c.candidates) == len(CMD_SPEC) == 16)

    names = {x.insert.strip() for x in c.candidates}
    check(
        "规格：一级命令清单",
        names
        == {
            "/help", "/clear", "/compact", "/exit", "/tree", "/up", "/home",
            "/delete", "/recall", "/file", "/files", "/save", "/web", "/model",
            "/thinking", "/a1",
        },
    )
    check(
        "规格：已删除的命令不再出现",
        not any(x in names for x in ("/set", "/wf", "/mcp", "/import", "/view", "/info", "/full")),
    )

    c = complete("/file", prov)
    check(
        "规格：/file 列出四态",
        c is not None and [x.insert for x in c.candidates]
        == ["/file off", "/file read", "/file confirm", "/file direct"],
    )
    check("规格：二级候选项带用法与说明", any(
        x.usage == "/file read" and "只读" in x.desc for x in c.candidates
    ))

    c = complete("/file re", prov)
    check("规格：二级前缀过滤", c is not None and [x.insert for x in c.candidates] == ["/file read"])

    c = complete("/web ", prov)
    check("规格：/web 取值枚举", c is not None and {x.insert for x in c.candidates} == {"/web on", "/web off"})

    c = complete("/recall ", prov)
    check("规格：/recall 取值枚举", c is not None and {x.insert for x in c.candidates} == {"/recall on", "/recall off"})

    c = complete("/model ", prov)
    check("规格：/model 只列 flash/pro", c is not None and {x.insert for x in c.candidates} == {"/model flash", "/model pro"})

    c = complete("/thinking ", prov)
    check("规格：/thinking 取值枚举",
          c is not None and {x.insert for x in c.candidates} == {"/thinking on", "/thinking off"})

    c = complete("/tree ", prov)
    check("规格：动态候选（对话树编号）",
          c is not None and any(x.usage == "/tree 1" and "当前" in x.desc for x in c.candidates))
    check("规格：静态子命令优先列出",
          c is not None and [x.insert for x in c.candidates][:3] == ["/tree list", "/tree new", "/tree delete "])

    c = complete("/tree delete ", prov)
    check("规格：子命令后的动态候选", c is not None and
          [x.insert for x in c.candidates] == ["/tree delete 1", "/tree delete 2"])

    c = complete("/h", prov)
    check("规格：别名按本体归位", c is not None and c.title == "/help")

    c = complete("/file read", prov)
    check("规格：叶子命令转提示（无候选）", c is not None and not c.candidates and c.title == "/file read")

    check("规格：未知命令不弹窗", complete("/zzz", prov) is None)
    check("规格：非命令不弹窗", complete("你好", prov) is None)

    def boom():
        raise RuntimeError("控制器未就绪")

    c = complete("/tree ", {"trees": boom})
    check("规格：动态来源异常不崩且退回静态候选",
          c is not None and [x.insert for x in c.candidates] == ["/tree list", "/tree new", "/tree delete "])

    check("用法：取值枚举直接写全",
          usage_line("/file") == "用法: /file <off|read|confirm|direct>")
    check("用法：含子命令给名称清单",
          usage_line("/tree") == "用法: /tree <子命令>（list/new/delete/<编号>）")
    check("用法：叶子给完整参数", usage_line("/delete") == "用法: /delete <节点ID> [...]")

    help_text = help_markdown()
    check("帮助：一级命令全部出现", all(f"/{c0.name}" in help_text for c0 in CMD_SPEC))
    check("帮助：二级命令分行列出",
          "/file read" in help_text and "/tree delete <编号>" in help_text)
    check("帮助：不再提到已删功能", "/wf" not in help_text and "/set" not in help_text
          and "/mcp" not in help_text)

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
        check("补全：/ 每行一个候选（用法 + 灰色说明）", "/exit" in text and "/file" in text)
        check("补全：灰字统一为说明（不再混入用法行）", "用法:" not in text)
        check("补全：输入 / 时无高亮项也能滚到第一项", popup.scroll_y == 0)
        check("补全：标题给出候选序号", "第 1/16 项" in text)

        # 15 个一级命令在 30 行终端里可能超出弹窗可视高度：Tab 必须把高亮项滚进视野
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

        await pilot.press("shift+tab")
        for _ in range(2):
            await pilot.pause()
        # 循环结束时高亮在第 0 项，反向一次应回到最后一项
        check("补全：Shift+Tab 反向切换", app._completion_index == len(CMD_SPEC) - 1)
        check("补全：序号随高亮更新", f"第 {len(CMD_SPEC)}/{len(CMD_SPEC)} 项" in popup_text())

        await type_text("/file")
        text = popup_text()
        check("补全：/file 分行列出四态", text.count("\n") == 4)
        check("补全：二级候选带用法与说明", "/file read" in text and "只读" in text)

        await pilot.press("tab")
        for _ in range(3):
            await pilot.pause()
        check("补全：Tab 选中第二项",
              app._completion.candidates[app._completion_index].insert == "/file read")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        check("补全：Enter 只补全不执行", inp.text == "/file read")

        await type_text("/tree ")
        text = popup_text()
        check("补全：动态列出对话树编号", "/tree 1" in text and "/tree 2" in text)
        await type_text("/tree delete ")
        check("补全：子命令后的动态候选",
              "/tree delete 1" in popup_text() and "/tree delete 2" in popup_text())

        await type_text("/web ")
        check("补全：/web 取值枚举", "→ /web on" in popup_text() and "/web off" in popup_text())

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

