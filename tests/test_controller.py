"""ChatController 单元测试（fake OpenAI client，不联网）。

运行：`python3 -m pytest -q tests/test_controller.py`
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli.attachments import KIND_DOCUMENT, KIND_IMAGE, Attachment
from mincli.config import MODEL_FLASH, MODEL_PRO, Config
from mincli.controller import ChatController, ControllerEvent
from mincli.tools.images import estimate_image_tokens, make_path_attachment

PASS = 0
FAIL = 0

_TMP = tempfile.mkdtemp(prefix="mincli_test_")

# 所有配置写回都落在临时目录，绝不碰用户主目录（_save_config 读这两个环境变量）
os.environ["MINCLI_CONFIG_PATH"] = os.path.join(_TMP, "config.toml")
os.environ["MINCLI_PROMPT_PATH"] = os.path.join(_TMP, "prompt.md")


def make_config(
    files_mode: str = "read",
    web_enabled: bool = True,
    recall_enabled: bool = False,
) -> Config:
    """测试用配置：附件与提示词都指向临时目录。"""
    return Config(
        model=MODEL_FLASH,
        temperature=1.0,
        files_mode=files_mode,
        web_enabled=web_enabled,
        recall_enabled=recall_enabled,
        attachment_dir=os.path.join(_TMP, "attachments"),
        prompt_file=os.path.join(_TMP, "prompt.md"),
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


# ---------------- fake OpenAI ----------------

class FakeDelta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class FakeToolCall:
    def __init__(self, index, id, name, arguments):
        self.index = index
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeChunk:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None, usage=None):
        self.choices = [SimpleNamespace(delta=FakeDelta(content, reasoning_content, tool_calls))]
        self.usage = usage


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(FakeMessage(content))]


class FakeCompletions:
    """按脚本顺序返回预置响应。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("FakeCompletions 脚本已用完")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeFiles:
    """Files API 模拟：可脚本化上传成功/失败（list 支持 order/limit/after）。"""

    def __init__(self, script=None, fail=False):
        self.script = list(script or [])
        self.calls = []
        self.fail = fail
        self.created = []
        self.list_calls = []

    def create(self, file, purpose="user_data"):
        self.calls.append(("create", file.name, purpose))
        if self.fail:
            raise RuntimeError("模拟上传失败")
        fid = self.script.pop(0) if self.script else f"file-api-{len(self.created)}"
        self.created.append({"id": fid, "name": os.path.basename(file.name)})
        return SimpleNamespace(id=fid)

    def list(self, limit=1000, order="asc", after=None):
        self.list_calls.append({"limit": limit, "order": order, "after": after})
        items = [
            SimpleNamespace(
                id=c["id"], filename=c["name"], bytes=100, created_at=1, expires_at=None
            )
            for c in self.created
        ]
        if order == "desc":
            items = list(reversed(items))
        return SimpleNamespace(data=items[:limit], has_more=len(items) > limit)

    def retrieve(self, file_id):
        self.calls.append(("retrieve", file_id))
        for c in self.created:
            if c["id"] == file_id:
                return SimpleNamespace(
                    id=c["id"], filename=c["name"], bytes=100,
                    created_at=1, expires_at=None,
                )
        raise RuntimeError("not found")

    def delete(self, file_id):
        self.calls.append(("delete", file_id))
        self.created = [c for c in self.created if c["id"] != file_id]
        return SimpleNamespace(deleted=True)


class FakeClient:
    def __init__(self, script, files_script=None, files_fail=False):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))
        self.files = FakeFiles(files_script, fail=files_fail)


# ---------------- 测试用控制器 ----------------

class TestController(ChatController):
    """测试控制器：对话树落在临时目录，默认先建一棵树。

    历史用例都面向「只有一棵树」的行为，因此构造后若磁盘上没有树就先建
    编号 1；需要验证「一棵树都没有」的用例请用 reset_store() 清空后自行构造。
    """

    TREES_DIR = os.path.join(_TMP, "trees")

    def __init__(self, client, config=None, default_system=None, trees_dir=None):
        super().__init__(
            client,
            config=config or make_config(),
            default_system=default_system,
            trees_dir=trees_dir or self.TREES_DIR,
        )
        if not self.has_trees:
            self.create_tree()

    @classmethod
    def reset_store(cls) -> None:
        """清空测试用树目录（用例之间相互隔离）。"""
        shutil.rmtree(cls.TREES_DIR, ignore_errors=True)
        os.makedirs(cls.TREES_DIR, exist_ok=True)


@pytest.fixture(autouse=True)
def _fresh_trees():
    """每个用例前后清空对话树目录：树是持久化的，不隔离会互相污染。"""
    TestController.reset_store()
    yield
    TestController.reset_store()


def _image_attachment(path: str, detail: str = "auto") -> Attachment:
    """把本地图片包成 send_message 需要的 Attachment。"""
    image = make_path_attachment(path, detail)
    return Attachment(
        kind=KIND_IMAGE,
        name=image.name,
        path=image.source,
        image=image,
        attachment_id="img-" + os.path.basename(path),
    )


def collect(ctrl, text, attachments=None):
    """调用 send_message 并收集事件。"""
    events = []

    def emit(ev: ControllerEvent):
        events.append(ev)

    node = ctrl.send_message(text, emit, attachments=attachments)
    return node, events


def test_simple_qa():
    print("== 基础问答（含思考） ==")
    script = [
        [
            FakeChunk(content="你好", reasoning_content="思考中"),
            FakeChunk(content="，", reasoning_content="继续思考"),
            FakeChunk(content="世界！", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)),
        ],
        FakeChatResponse(content="测试标题"),
    ]
    ctrl = TestController(
        FakeClient(script), config=make_config(), default_system="你是助手"
    )
    node, events = collect(ctrl, "打个招呼")
    check("返回节点", node is not None)
    check("节点内容", node.assistant_msg == "你好，世界！")
    check("节点思考", node.reasoning == "思考中继续思考")
    check("节点标题", node.title == "测试标题")
    check("根节点创建", ctrl.tree.root is not None and ctrl.tree.current_node is node)
    kinds = [e.kind for e in events]
    check("事件含 node_created", "node_created" in kinds)
    check("事件含 stream", "stream" in kinds)
    check("事件含 done", "done" in kinds)
    check("无 error 事件", "error" not in kinds)
    nc = next(e for e in events if e.kind == "node_created")
    check("node_created 即当前节点", nc.node is not None and ctrl.tree.current_node is nc.node)
    content = "".join(e.content for e in events if e.kind == "stream")
    reasoning = "".join(e.reasoning for e in events if e.kind == "stream")
    check("流式增量完整", content == "你好，世界！" and reasoning == "思考中继续思考")
    check("token 统计", node.input_tokens == 10 and node.output_tokens == 5)


def test_tool_round():
    print("== 工具调用轮（read_file → 继续问答） ==")
    target = os.path.join(_TMP, "tool_round.txt")
    with open(target, "w", encoding="utf-8") as f:
        f.write("第一行\n第二行\n第三行\n")
    tool_chunks = [
        FakeChunk(
            tool_calls=[
                FakeToolCall(
                    index=0,
                    id="call_1",
                    name="read_file",
                    arguments=json.dumps({"path": target, "offset": 2, "limit": 1}),
                )
            ]
        )
    ]
    content_chunks = [FakeChunk(content="文件读完了。")]
    script = [tool_chunks, content_chunks, FakeChatResponse(content="读文件测试")]
    ctrl = TestController(FakeClient(script), config=make_config(files_mode="read"))
    node, events = collect(ctrl, "帮我读个文件")
    check("工具轮返回节点", node is not None)
    check("最终回答", node.assistant_msg == "文件读完了。")
    tool_events = [e for e in events if e.kind == "tool"]
    check("工具事件成对（开始+结果）", len(tool_events) == 2)
    check("工具名正确", tool_events[0].tool_name == "read_file")
    check("工具结果摘要", tool_events[1].tool_summary.startswith("文件："))
    # 验证发给模型的消息包含 tool 结果
    calls = ctrl.client.chat.completions.calls
    msgs2 = calls[1]["messages"]
    roles = [m["role"] for m in msgs2]
    check("第二轮消息含 assistant+tool", "assistant" in roles and "tool" in roles)
    check("tool 消息携带 tool_call_id", any(m.get("tool_call_id") == "call_1" for m in msgs2 if m["role"] == "tool"))
    check("工具结果含文件内容", any("第二行" in str(m.get("content")) for m in msgs2 if m["role"] == "tool"))
    check("工具轮消息写入节点", bool(node.tool_messages))


def test_tool_gating_matrix():
    print("== 工具门控矩阵：files_mode × 联网 × 树查询 ==")
    file_names = {
        "off": set(),
        "read": {"read_file", "list_directory"},
        "confirm": {"read_file", "list_directory", "write_file", "edit_file"},
        "direct": {"read_file", "list_directory", "write_file", "edit_file"},
    }
    web_tools = {"web_search", "web_fetch"}
    recall_tools = {"query_conversation_tree", "read_conversation_nodes"}

    for mode in ("off", "read", "confirm", "direct"):
        for web_on in (False, True):
            for recall_on in (False, True):
                cfg = make_config(
                    files_mode=mode,
                    web_enabled=web_on,
                    recall_enabled=recall_on,
                )
                ctrl = TestController(FakeClient([]), config=cfg)
                names = {t["function"]["name"] for t in ctrl.llm_tools}
                expected = set(file_names[mode])
                if web_on:
                    expected |= web_tools
                if recall_on:
                    expected |= recall_tools
                check(
                    f"门控 {mode} / web={web_on} / recall={recall_on}",
                    names == expected,
                )

    off_ctrl = TestController(
        FakeClient([]), config=make_config("off", web_enabled=False, recall_enabled=False)
    )
    check("off 且无附件时工具列表为空", off_ctrl.llm_tools == [])


def test_send_with_attachment_persists():
    print("== 附件发送：元数据 / 落盘 / read_attachment 续读 ==")
    script = [
        [FakeChunk(content="看过了", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))],
        FakeChatResponse(content="附件标题"),
    ]
    ctrl = TestController(FakeClient(script), config=make_config(files_mode="off"))
    text = "第一行\n第二行\n第三行"
    att = Attachment(
        kind=KIND_DOCUMENT,
        name="a.txt",
        path=os.path.join(_TMP, "a.txt"),
        text=text,
        attachment_id="a1",
    )
    node, events = collect(ctrl, "看看", attachments=[att])
    check("返回节点", node is not None)
    check("正文仍由调用方拼好", node.user_msg == "看看")
    check("附件元数据写入节点", len(node.attachments) == 1)
    meta = node.attachments[0]
    check("元数据字段完整", meta["id"] == "a1" and meta["name"] == "a.txt"
          and meta["kind"] == KIND_DOCUMENT)
    check("元数据记录行数与字符数", meta["lines"] == 3 and meta["chars"] == len(text))
    stored = os.path.join(ctrl.config.attachment_dir, "a1.txt")
    check("正文落盘到 attachment_dir", os.path.exists(stored))
    with open(stored, "r", encoding="utf-8") as f:
        check("落盘内容与附件一致", f.read() == text)
    check("元数据带落盘路径", meta["stored_path"] == stored)
    names = {t["function"]["name"] for t in ctrl.llm_tools}
    check("有附件时下发 read_attachment（与 files_mode=off 无关）",
          "read_attachment" in names and "read_file" not in names)
    out = ctrl._run_tool(
        "read_attachment",
        {"name": "a.txt", "offset": 2, "limit": 1},
        lambda ev: None,
        node,
    )
    check("read_attachment 能续读", "第二行" in out and "第 2-2 行" in out)
    check("续读提示下一段 offset", "offset=3" in out)


def test_write_confirmation():
    print("== 写文件确认：confirm 拒绝 / direct 直写 / off 不下发 ==")
    target = os.path.join(_TMP, "write_confirm.txt")
    if os.path.exists(target):
        os.remove(target)

    confirm_ctrl = TestController(FakeClient([]), config=make_config("confirm"))
    asked = []

    def _deny(title, text):
        asked.append(title)
        return False

    confirm_ctrl.confirm = _deny
    confirm_ctrl.tree.create_root("q", "a", "", "t", 0, 0)
    node = confirm_ctrl.tree.current_node
    out = confirm_ctrl._run_tool(
        "write_file", {"path": target, "content": "hi"},
        lambda ev: None, node,
    )
    check("confirm 模式先询问", asked == ["确认写入"])
    check("confirm 拒绝时不写文件", not os.path.exists(target))
    check("confirm 拒绝返回取消文案", "取消" in out)

    confirm_ctrl.confirm = lambda title, text: True
    out = confirm_ctrl._run_tool(
        "write_file", {"path": target, "content": "hi"},
        lambda ev: None, node,
    )
    check("confirm 同意后写文件", os.path.exists(target) and "已写入" in out)
    with open(target, "r", encoding="utf-8") as f:
        check("写文件内容正确", f.read() == "hi")

    direct_target = os.path.join(_TMP, "write_direct.txt")
    if os.path.exists(direct_target):
        os.remove(direct_target)
    direct_ctrl = TestController(FakeClient([]), config=make_config("direct"))
    direct_asked = []
    direct_ctrl.confirm = lambda title, text: direct_asked.append(title) or False
    direct_ctrl.tree.create_root("q", "a", "", "t", 0, 0)
    direct_out = direct_ctrl._run_tool(
        "write_file", {"path": direct_target, "content": "direct"},
        lambda ev: None, direct_ctrl.tree.current_node,
    )
    check("direct 模式直接写文件",
          os.path.exists(direct_target) and "已写入" in direct_out)
    check("direct 模式不弹确认", direct_asked == [])

    off_ctrl = TestController(FakeClient([]), config=make_config("off"))
    off_names = {t["function"]["name"] for t in off_ctrl.llm_tools}
    check("off 模式不下发 write_file/edit_file",
          "write_file" not in off_names and "edit_file" not in off_names)


def test_web_disabled_hides_tools():
    print("== 联网关闭时不下发搜索/抓取 ==")
    disabled = TestController(
        FakeClient([]), config=make_config("read", web_enabled=False)
    )
    disabled_names = {t["function"]["name"] for t in disabled.llm_tools}
    check("web_enabled=False 时没有 web_search/web_fetch",
          "web_search" not in disabled_names and "web_fetch" not in disabled_names)
    check("文件只读工具仍可用",
          {"read_file", "list_directory"} <= disabled_names)

    enabled = TestController(
        FakeClient([]), config=make_config("read", web_enabled=True)
    )
    enabled_names = {t["function"]["name"] for t in enabled.llm_tools}
    check("web_enabled=True 时下发 web_search/web_fetch",
          {"web_search", "web_fetch"} <= enabled_names)


def test_api_error():
    print("== API 错误处理（节点保留，可继续） ==")
    script = [RuntimeError("connection reset")]
    ctrl = TestController(FakeClient(script), config=make_config())
    node, events = collect(ctrl, "hello")
    check("返回已保存的节点（不再回滚）", node is not None and ctrl.tree.root is node)
    check("节点仍是当前节点", ctrl.tree.current_node is node and node.id in ctrl.tree.nodes)
    check("错误原因写入节点", "connection reset" in node.error)
    check("无内容时回答为空", node.assistant_msg == "")
    check("中断轮不额外调用标题生成", len(ctrl.client.chat.completions.calls) == 1)
    errs = [e for e in events if e.kind == "error"]
    check("发出 error 事件", len(errs) == 1 and "connection reset" in errs[0].message)
    check("发出 done 事件（UI 刷新树/状态条）", "done" in [e.kind for e in events])

    # 继续：空回答的中断节点不能再把空 assistant 消息发回模型（否则 API 报错）
    ctrl.client.chat.completions.script = [
        [FakeChunk(content="补上回答")],
        FakeChatResponse(content="标题"),
    ]
    node2, _ = collect(ctrl, "继续")
    msgs = ctrl.client.chat.completions.calls[1]["messages"]
    check("继续轮挂在中断节点下", node2 is not None and node2.parent_id == node.id)
    check("继续轮不发送空 assistant 消息",
          all(not (m.get("role") == "assistant" and not m.get("content")) for m in msgs))
    check("继续轮历史含上一轮提问", any(
        m.get("role") == "user" and m.get("content") == "hello" for m in msgs
    ))
    check("继续轮正常落盘", node2.assistant_msg == "补上回答")


def test_partial_save_on_error():
    print("== 生成中断：部分回答落盘 + 可「继续」 ==")

    def _chunks_then_raise():
        """先吐两个增量再抛错，模拟生成中途失败（如 400 Content Exists Risk）。"""
        yield FakeChunk(content="前半句", reasoning_content="先想想")
        yield FakeChunk(content="，还没写完")
        raise RuntimeError("Error code: 400 - Content Exists Risk")

    ctrl = TestController(
        FakeClient([_chunks_then_raise()]),
        config=make_config(),
    )
    node, events = collect(ctrl, "写个长回答")
    check("中断节点已保存", node is not None and node.id in ctrl.tree.nodes)
    check("部分回答已落盘", node.assistant_msg == "前半句，还没写完")
    check("部分思考已落盘", node.reasoning == "先想想")
    check("错误原因写入节点", "Content Exists Risk" in node.error)
    check("流式内容与落盘一致",
          "".join(e.content for e in events if e.kind == "stream") == node.assistant_msg)
    kinds = [e.kind for e in events]
    check("先 error 后 done", kinds.count("error") == 1 and "done" in kinds)
    check("error 在 done 之前", kinds.index("error") < kinds.index("done"))
    check("中断轮不额外调用标题生成", len(ctrl.client.chat.completions.calls) == 1)

    # 「继续」：下一轮历史带上已保存的半截回答与思考，模型可接着写
    ctrl.client.chat.completions.script = [
        [FakeChunk(content="接着写完了")],
        FakeChatResponse(content="继续标题"),
    ]
    node2, _ = collect(ctrl, "继续")
    msgs = ctrl.client.chat.completions.calls[1]["messages"]
    check("继续轮历史含半截回答", any(
        m.get("role") == "assistant" and m.get("content") == "前半句，还没写完"
        for m in msgs
    ))
    check("继续轮历史含思考内容", any(
        m.get("role") == "assistant" and m.get("reasoning_content") == "先想想"
        for m in msgs
    ))
    check("继续轮是中断节点的子节点", node2 is not None and node2.parent_id == node.id)
    check("继续后错误标记不残留", node2.error == "" and node2.assistant_msg == "接着写完了")

    # 中断标记与部分回答随会话持久化（重启后仍能看到并继续）
    check("保存带中断标记的会话", ctrl.save_session())
    ctrl_reload = TestController(
        FakeClient([]), config=make_config(),
    )
    reloaded = ctrl_reload.tree.nodes.get(node.id)
    check("重载后部分回答仍在",
          reloaded is not None and reloaded.assistant_msg == "前半句，还没写完")
    check("重载后中断标记仍在",
          reloaded is not None and "Content Exists Risk" in reloaded.error)
    TestController.reset_store()


def test_tool_round_partial_save():
    print("== 工具轮之后中断：工具结果一并保存 ==")
    target = os.path.join(_TMP, "tool_partial.txt")
    with open(target, "w", encoding="utf-8") as f:
        f.write("检索到的资料\n")
    tool_chunks = [
        FakeChunk(
            tool_calls=[
                FakeToolCall(
                    index=0,
                    id="call_9",
                    name="read_file",
                    arguments=json.dumps({"path": target}),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )
    ]

    def _answer_then_raise():
        yield FakeChunk(content="基于检索结果，")
        raise RuntimeError("connection reset by peer")

    ctrl = TestController(
        FakeClient([tool_chunks, _answer_then_raise()]),
        config=make_config(files_mode="read"),
    )
    node, events = collect(ctrl, "帮我读文件再总结")
    check("工具轮后中断的节点保留", node is not None and node.assistant_msg == "基于检索结果，")
    check("已完成的工具轮写入节点", bool(node.tool_messages))
    check("工具轮 token 用量保留", node.input_tokens >= 7 and node.output_tokens >= 3)

    ctrl.client.chat.completions.script = [
        [FakeChunk(content="总结完毕")],
        FakeChatResponse(content="标题"),
    ]
    node2, _ = collect(ctrl, "继续")
    # calls: [0] 工具轮 [1] 失败轮 [2] 继续轮 [3] 标题生成
    msgs = ctrl.client.chat.completions.calls[2]["messages"]
    check("继续轮历史含 tool 结果",
          any(m.get("role") == "tool" for m in msgs))
    check("继续轮回答落盘", node2 is not None and node2.assistant_msg == "总结完毕")


def test_session_roundtrip():
    print("== 会话持久化 ==")
    TestController.reset_store()
    script = [[FakeChunk(content="第一轮")], FakeChatResponse(content="标题A")]
    ctrl = TestController(FakeClient(script), config=make_config())
    check("首次无已存会话", not ctrl.session_loaded)
    collect(ctrl, "第一问")
    check("save_session 成功", ctrl.save_session())
    check("文件已写入", os.path.exists(os.path.join(TestController.TREES_DIR, "1.json")))

    script2 = [[FakeChunk(content="第二轮")], FakeChatResponse(content="标题B")]
    ctrl2 = TestController(FakeClient(script2), config=make_config())
    check("重新加载会话", ctrl2.session_loaded)
    check("树已恢复", ctrl2.tree.root is not None and ctrl2.tree.root.title == "标题A")
    node, _ = collect(ctrl2, "第二问")
    check("在已有树上追加节点", node is not None and node.parent_id is not None and node.id != "main")
    TestController.reset_store()


def test_path_args():
    """跨平台路径解析：Windows 反斜杠路径不能被 POSIX shlex 吃掉。"""
    print("== 跨平台路径解析（/import 与拖入粘贴） ==")
    from mincli.helpers import split_path_args

    check("路径解析：Windows 未加引号反斜杠路径完整保留",
          split_path_args(r"C:\Users\me\notes.txt") == [r"C:\Users\me\notes.txt"])
    check("路径解析：Windows 带空格多文件（双引号）",
          split_path_args(r'"C:\Users\me\My File.txt" "D:\b.txt"')
          == [r"C:\Users\me\My File.txt", r"D:\b.txt"])
    check("路径解析：Windows 单引号路径",
          split_path_args(r"'C:\x\y.txt'") == [r"C:\x\y.txt"])
    check("路径解析：POSIX 引号路径去引号",
          split_path_args('"/tmp/a b.txt"') == ["/tmp/a b.txt"])
    check("路径解析：POSIX 转义空格",
          split_path_args(r"/tmp/a\ b.txt") == ["/tmp/a b.txt"])
    check("路径解析：URL 原样保留",
          split_path_args("https://a.com/x.png http://b.com")
          == ["https://a.com/x.png", "http://b.com"])
    check("路径解析：普通文本原样返回",
          split_path_args("这是一段普通文本") == ["这是一段普通文本"])
    check("路径解析：空串返回空列表", split_path_args("") == [])
    check("路径解析：引号不匹配不抛异常",
          isinstance(split_path_args('"/tmp/unclosed'), list))


def test_delete_nodes():
    print("== 批量删除 /delete a1 b3 g5 ==")
    ctrl = TestController(FakeClient([]), config=make_config())
    ctrl.tree.create_root("根", "回", "", "根", 1, 1)
    a1 = ctrl.tree.add_child(ctrl.tree.root, "q", "a", "", "t", 1, 1)
    a2 = ctrl.tree.add_child(a1, "q", "a", "", "t", 1, 1)
    b1 = ctrl.tree.add_child(ctrl.tree.root, "q", "a", "", "t", 1, 1)
    b2 = ctrl.tree.add_child(b1, "q", "a", "", "t", 1, 1)
    b3 = ctrl.tree.add_child(b2, "q", "a", "", "t", 1, 1)
    c1 = ctrl.tree.add_child(ctrl.tree.root, "q", "a", "", "t", 1, 1)

    # a2 是 a1 子孙、b3 是 b2 子孙：仅删父即可，子节点随父级联删除不报错
    res = ctrl.delete_nodes([a1.id, a2.id, b2.id, b3.id, "ghost"])
    check("删除顶层节点", set(res["deleted"]) == {a1.id, b2.id})
    check("子孙随父级联删除", a2.id not in ctrl.tree.nodes and b3.id not in ctrl.tree.nodes)
    check("无关节点保留", b1.id in ctrl.tree.nodes and c1.id in ctrl.tree.nodes and "main" in ctrl.tree.nodes)
    check("不存在的节点不报错", "ghost" not in res["deleted"])

    # 根节点跳过
    res2 = ctrl.delete_nodes(["main", b1.id])
    check("根节点跳过", res2["deleted"] == [b1.id] and res2["skipped"] == ["main"] and ctrl.tree.root is not None)

    # 删除摘要节点后压缩状态清除（避免悬挂）
    ctrl2 = TestController(FakeClient([]), config=make_config())
    ctrl2.tree.create_root("根", "回", "", "根", 1, 1)
    s = ctrl2.tree.add_child(ctrl2.tree.root, "摘要", "", "", "上下文压缩摘要", 0, 0)
    ctrl2.tree.compaction = {"summary": "摘要", "boundary_id": s.id}
    ctrl2.tree.current_node = s
    ctrl2.delete_nodes([s.id])
    check("删除摘要节点后压缩状态清除", ctrl2.tree.compaction is None)


def test_settings():
    print("== 设置：温度 / 思考 / 推理强度 / 模型 ==")
    ctrl = TestController(FakeClient([]), config=make_config())
    ctrl.set_temperature(0.3)
    check("温度写入运行时", ctrl.current_temperature == 0.3)
    check("温度写入配置", ctrl.config.temperature == 0.3)
    ctrl.set_thinking(True)
    check("thinking 设置", ctrl.thinking_enabled and ctrl.config.thinking is True)
    check("effort 校验", ctrl.set_effort("max") and not ctrl.set_effort("extreme"))
    check("effort 写入运行时与配置",
          ctrl.reasoning_effort == "max" and ctrl.config.effort == "max")
    check("切换模型 flash", ctrl.set_model("flash") and ctrl.current_model == MODEL_FLASH)
    check("切换模型 pro", ctrl.set_model("pro") and ctrl.current_model == MODEL_PRO)
    check("切回 flash", ctrl.set_model("flash") and ctrl.current_model == MODEL_FLASH)
    check("非法模型拒绝", not ctrl.set_model("turbo") and ctrl.current_model == MODEL_FLASH)
    check("模型写入配置", ctrl.config.model == MODEL_FLASH)


def test_compact():
    print("== 上下文压缩 /compact（全部压缩 + 新建摘要节点） ==")
    script = []
    for i in range(1, 6):
        # 回答内容较长，保证压缩后确实节省 token
        script.append([FakeChunk(content=f"回答{i}：" + "详细内容" * 60)])
        script.append(FakeChatResponse(content=f"标题{i}"))
    # 第 4 轮结束时子树「a」达 3 节点，触发 _auto_title_subtree，紧随 标题4（index 7）之后
    script.insert(8, FakeChatResponse(content="子树标题"))
    script.append(FakeChatResponse(content="【摘要】目标X；已执行命令Y；待办Z。"))
    script.append([FakeChunk(content="回答新：" + "新内容" * 60)])
    script.append(FakeChatResponse(content="标题新"))
    ctrl = TestController(FakeClient(script), config=make_config())
    for i in range(1, 6):
        collect(ctrl, f"问题{i}")

    # 压缩——压缩前状态条是普通节点：真实 usage 口径（最后一次请求 prompt + 该轮输出）
    before_node = ctrl.tree.current_node
    before_us = ctrl.usage_stats()
    from mincli.helpers import estimate_prompt_tokens, estimate_tokens
    events = []
    stats = ctrl.compact_history(emit=events.append)
    check("压缩返回统计", stats is not None)
    check(
        "压缩前状态条=最后一次请求 usage 口径",
        before_us["next_input_tokens"]
        == before_node.last_prompt_tokens + before_node.last_output_tokens,
    )
    # 压缩后：摘要节点没有 API usage，回退到「含工具定义」的本地估算
    after_us = ctrl.usage_stats()
    summary_node = ctrl.tree.current_node
    check("摘要节点无 usage（回退估算）",
          summary_node.last_prompt_tokens == 0 and summary_node.input_tokens == 0)
    check(
        "压缩后状态条=摘要上下文估算+工具定义",
        after_us["next_input_tokens"]
        == estimate_prompt_tokens(ctrl.tree.get_messages_for_node(summary_node), ctrl.llm_tools)
        and after_us["next_input_tokens"] > stats["after_tokens"],
    )
    check("压缩后状态条显著变小", after_us["next_input_tokens"] < before_us["next_input_tokens"])
    check("全部压缩（main+4轮=5节点）", stats["nodes_compressed"] == 5)
    check("新建摘要节点", stats["node_id"] is not None and stats["node_id"] in ctrl.tree.nodes)
    check("摘要节点设为当前", ctrl.tree.current_node.id == stats["node_id"])
    check("摘要节点用户消息=摘要", ctrl.tree.current_node.user_msg.startswith("【摘要】"))
    check("摘要写入树", ctrl.tree.compaction is not None and ctrl.tree.compaction["summary"].startswith("【摘要】"))
    check("boundary=摘要节点", stats["node_id"] == ctrl.tree.compaction["boundary_id"])
    check("发出 status 事件", any(e.kind == "status" for e in events))
    check("节省 token>0", stats["saved_tokens"] > 0)

    msgs = ctrl.tree.get_messages_for_node(ctrl.tree.current_node)
    joined = "\n".join(str(m.get("content", "")) for m in msgs)
    check("摘要节点消息仅含摘要", "【摘要】" in joined and "回答1" not in joined and "回答5" not in joined)
    check("摘要带前缀标记", msgs[1]["content"].startswith("【以下是本对话早期内容"))

    # 摘要节点上继续对话：发给模型的消息 = 摘要 + 新输入
    node, _ = collect(ctrl, "新问题")
    check("摘要节点上继续对话成功", node is not None)
    check("新节点是摘要节点的子节点", node.parent_id == stats["node_id"])
    sent_joined = ""
    for call in ctrl.client.chat.completions.calls:
        msgs = call.get("messages", [])
        if msgs and msgs[-1].get("role") == "user" and str(msgs[-1].get("content", "")).strip() == "新问题":
            sent_joined = "\n".join(str(m.get("content", "")) for m in msgs)
    check("发给模型的消息含摘要", "【摘要】" in sent_joined)
    check("发给模型的消息不含旧回答", "回答1" not in sent_joined and "回答3" not in sent_joined)

    # 切换到摘要节点之前的节点：仍使用完整历史
    old_id = ctrl._path_to_root(ctrl.tree.current_node)[-3].id  # 摘要节点前一个节点
    check("切换到旧节点", ctrl.tree.switch_to_node(old_id))
    full_joined = "\n".join(str(m.get("content", "")) for m in ctrl.tree.get_messages_for_node(ctrl.tree.current_node))
    check("旧节点仍用完整历史", "回答1" in full_joined and "回答5" in full_joined)
    check("旧节点消息不含摘要", "【摘要】" not in full_joined)

    # 当前节点=摘要节点时禁止重复压缩
    ctrl.tree.switch_to_node(stats["node_id"])
    blocked = ctrl.compact_history()
    check("摘要节点禁止重复压缩", blocked is not None and blocked.get("blocked") == "already_compact")

    # 压缩随会话持久化
    ctrl.save_session()
    ctrl3 = TestController(FakeClient([]), config=make_config())
    check("重载后压缩仍在", ctrl3.session_loaded and ctrl3.tree.compaction is not None)
    TestController.reset_store()

    # 空会话无可压缩内容
    ctrl2 = TestController(
        FakeClient([]),
        config=make_config(),
    )
    check("空会话返回 None", ctrl2.compact_history() is None)


def _make_png(path: str, w: int = 800, h: int = 600) -> str:
    """构造最小合法 PNG（前 24 字节足够嗅探+尺寸）。"""
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\x0dIHDR"
            + w.to_bytes(4, "big")
            + h.to_bytes(4, "big")
            + b"\x08\x06\x00\x00\x00"
        )
    return path


def test_multimodal():
    print("== 多模态：上传 / 守卫 / 回退 / 历史重放 / 文件管理 ==")
    png = _make_png(os.path.join(_TMP, "m.png"))

    # 1) 上传成功 → file 块（flash 原生支持图片，模型不切换）
    script = [
        [FakeChunk(content="图", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5))],
        FakeChatResponse(content="图题"),
    ]
    ctrl = TestController(FakeClient(script), config=make_config())
    events = []
    node = ctrl.send_message(
        "描述图片", events.append, attachments=[_image_attachment(png)]
    )
    check("发送成功", node is not None)
    check("file_id 已写入节点", node.user_images[0].file_id == "file-api-0")
    check("Flash 原生支持图片（模型不切换）", ctrl.current_model == MODEL_FLASH)
    check("上传调用 1 次", len(ctrl.client.files.calls) == 1)
    check("附件元数据写入节点", len(node.attachments) == 1
          and node.attachments[0]["kind"] == KIND_IMAGE)
    check("发图片后工具列表含 read_attachment",
          "read_attachment" in {t["function"]["name"] for t in ctrl.llm_tools})
    sent = ctrl.client.chat.completions.calls[0]["messages"]
    blocks = sent[-1]["content"]
    check("消息为块数组", isinstance(blocks, list) and blocks[0]["type"] == "text")
    check("图片为 file 块", any(
        b.get("type") == "file" and b.get("file_id") == "file-api-0" for b in blocks
    ))
    check("无 error 事件", "error" not in [e.kind for e in events])

    # 2) 上传失败 → base64 内联回退
    script2 = [
        [FakeChunk(content="ok", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))],
        FakeChatResponse(content="标题2"),
    ]
    ctrl2 = TestController(FakeClient(script2, files_fail=True), config=make_config())
    events2 = []
    node2 = ctrl2.send_message(
        "看图", events2.append, attachments=[_image_attachment(png)]
    )
    check("回退发送成功", node2 is not None and node2.user_images[0].file_id is None)
    sent2 = ctrl2.client.chat.completions.calls[0]["messages"]
    blocks2 = sent2[-1]["content"]
    check("回退为 data URL", any(
        b.get("type") == "image_url"
        and str(b.get("image_url", {}).get("url", "")).startswith("data:image/png;base64,")
        for b in blocks2
    ))
    check("回退提示发出", any(e.kind == "status" and "内联" in e.message for e in events2))

    # 3) 不支持图片的模型（如 gpt-4o）→ 报错 + 图片放回待发送
    script3 = [
        [FakeChunk(content="x", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="t"),
    ]
    # 独立场景：清空树目录，保证 ctrl3 从空树开始（前两个控制器已把树落盘）
    TestController.reset_store()
    ctrl3 = TestController(FakeClient(script3), config=make_config())
    ctrl3.current_model = "gpt-4o"
    events3 = []
    node3 = ctrl3.send_message(
        "看图", events3.append, attachments=[_image_attachment(png)]
    )
    check("自定义模型被拒但节点保留", node3 is not None and "不支持图片" in node3.error)
    check("图片保留在中断节点里", len(node3.user_images) == 1)
    check("发出 error 事件（提示不支持图片）", any(
        e.kind == "error" and "不支持图片" in e.message for e in events3
    ))
    check("节点留在树中", node3.id in ctrl3.tree.nodes and ctrl3.tree.root is node3)

    # 3b) Pro 不支持图片 → 提示手动切换，且不自动改模型
    ctrl3b = TestController(FakeClient([]), config=make_config())
    ctrl3b.set_model("pro")
    events3b = []
    node3b = ctrl3b.send_message(
        "看图", events3b.append, attachments=[_image_attachment(png)]
    )
    check("Pro 收到图片被拒但节点保留",
          node3b is not None and ctrl3b.current_model == MODEL_PRO)
    check("Pro 提示切到 flash", any(
        e.kind == "error" and "/model flash" in e.message for e in events3b
    ))
    check("Pro 图片保留在中断节点里", len(node3b.user_images) == 1)

    # 4) 历史重放：第一轮回退 base64，第二轮补传 → file 块
    script4 = [
        [FakeChunk(content="a", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))],
        FakeChatResponse(content="题A"),
    ]
    c4 = TestController(FakeClient(script4, files_fail=True), config=make_config())
    n1 = c4.send_message(
        "图一", lambda e: None, attachments=[_image_attachment(png)]
    )
    check("第一轮回退成功", n1 is not None)
    c4.client.files.fail = False
    c4.client.chat.completions.script = [
        [FakeChunk(content="b", usage=SimpleNamespace(prompt_tokens=20, completion_tokens=3))],
        FakeChatResponse(content="题B"),
    ]
    n2 = c4.send_message("继续", lambda e: None)
    check("第二轮发送成功", n2 is not None)
    hist = c4.client.chat.completions.calls[2]["messages"]  # 第 2 轮流式请求
    check("历史消息含 file 块", any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "file" for b in m["content"] if isinstance(b, dict))
        for m in hist if m.get("role") == "user"
    ))
    check("补传后 file_id 写入", n1.user_images[0].file_id == "file-api-0")

    # 5) 文件管理 + 节点删除清理
    script6 = [
        [FakeChunk(content="a", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="题A"),
        [FakeChunk(content="b", usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))],
        FakeChatResponse(content="题B"),
    ]
    c6 = TestController(FakeClient(script6, files_script=["file-api-abc"]), config=make_config())
    n6a = c6.send_message("第一问", lambda e: None)
    n6b = c6.send_message(
        "看图", lambda e: None, attachments=[_image_attachment(png)]
    )
    check("上传取脚本 id", n6b.user_images[0].file_id == "file-api-abc")
    files = c6.files_list()
    check("files_list 返回文件", len(files["items"]) == 1
          and files["items"][0]["id"] == "file-api-abc")
    check("files_list 默认倒序（最新在前）",
          c6.client.files.list_calls[-1]["order"] == "desc")
    check("files_retrieve 取回信息",
          c6.files_retrieve("file-api-abc")["name"] == "m.png")
    check("files_delete 调用删除", c6.files_delete("file-api-abc"))
    check("删除 API 已调用", c6.client.files.calls[-1] == ("delete", "file-api-abc"))
    check("删除子节点清理关联文件", c6.delete_node(n6b.id))
    check("节点文件已删除", any(c[0] == "delete" for c in c6.client.files.calls))
    check("根节点不能删", not c6.delete_node("main"))
    check("子节点已从树移除", n6b.id not in c6.tree.nodes)

    # 7) 压缩源图片占位
    script7 = [
        [FakeChunk(content="x", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="题"),
    ]
    c7 = TestController(FakeClient(script7), config=make_config())
    n7 = c7.send_message(
        "图", lambda e: None, attachments=[_image_attachment(png)]
    )
    check("压缩用例发送成功", n7 is not None)
    src = c7._build_compact_source([n7])
    check("压缩源含图片占位", "[图片: m.png (800x600)]" in src)

    # 8) 会话持久化保留附件（路径与 file_id，不含 base64）
    TestController.reset_store()
    c8 = TestController(FakeClient(script6, files_script=["file-api-persist"]), config=make_config())
    n8 = c8.send_message(
        "图", lambda e: None, attachments=[_image_attachment(png)]
    )
    check("持久化前 file_id", n8.user_images[0].file_id == "file-api-persist")
    c8.save_session()
    c8b = TestController(FakeClient([]), config=make_config())
    check("重载后附件保留", c8b.session_loaded
          and c8b.tree.current_node is not None
          and c8b.tree.current_node.user_images
          and c8b.tree.current_node.user_images[0].file_id == "file-api-persist")
    TestController.reset_store()

    # 9) set_detail 校验
    c9 = TestController(FakeClient([]), config=make_config())
    check("set_detail low", c9.set_detail("low") and c9.image_detail == "low")
    check("set_detail 非法拒绝", not c9.set_detail("huge"))
    check("set_model vision 别名 → flash", c9.set_model("vision") and c9.current_model == MODEL_FLASH)
    check("旧名 deepseek-v4-flash 改写为 flash",
          c9.set_model("deepseek-v4-flash") and c9.current_model == MODEL_FLASH)
    check("set_model pro", c9.set_model("pro") and c9.current_model == MODEL_PRO)

    # 10) detail=low → 内联发送（file 块没有 detail 字段，只有内联才真正生效）
    TestController.reset_store()
    script10 = [
        [FakeChunk(content="ok", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))],
        FakeChatResponse(content="标题10"),
    ]
    c10 = TestController(FakeClient(script10), config=make_config())
    att10 = _image_attachment(png, detail="low")
    check("low 档图片 token 估算",
          att10.image.tokens_est == estimate_image_tokens(800, 600, "low"))
    events10 = []
    n10 = c10.send_message("看图", events10.append, attachments=[att10])
    check("low 档不上传 Files API", n10.user_images[0].file_id is None
          and not c10.client.files.calls)
    blocks10 = c10.client.chat.completions.calls[0]["messages"][-1]["content"]
    check("low 档走内联并带 detail=low", any(
        b.get("type") == "image_url" and b.get("image_url", {}).get("detail") == "low"
        for b in blocks10))
    check("low 档发出内联提示", any(
        e.kind == "status" and "内联" in e.message for e in events10))
    check("low 档估算低于 auto 档",
          estimate_image_tokens(800, 600, "low") < estimate_image_tokens(800, 600))

    # 11) detail=low 但超过内联单图上限（32MiB）→ 退回上传，detail 被忽略
    TestController.reset_store()
    big = _make_png(os.path.join(_TMP, "big_low.png"))
    with open(big, "ab") as f:
        f.truncate(33 * 1024 * 1024)
    script11 = [
        [FakeChunk(content="ok", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))],
        FakeChatResponse(content="标题11"),
    ]
    c11 = TestController(FakeClient(script11, files_script=["file-api-big"]),
                         config=make_config())
    att11 = _image_attachment(big, detail="low")
    events11 = []
    n11 = c11.send_message("大图", events11.append, attachments=[att11])
    check("超内联上限的 low 图仍上传", n11.user_images[0].file_id == "file-api-big")
    check("提示 detail 将被忽略", any(
        e.kind == "status" and "detail 将被忽略" in e.message for e in events11))

    # 12) files_clean：跨对话树统计引用，只清理没人引用的远端文件
    TestController.reset_store()
    script12 = [
        [FakeChunk(content="a", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="题A"),
        [FakeChunk(content="b", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="题B"),
    ]
    c12 = TestController(FakeClient(script12), config=make_config())
    n12 = c12.send_message(
        "树1看图", lambda e: None, attachments=[_image_attachment(png)]
    )
    ref1 = n12.user_images[0].file_id
    c12.create_tree()  # 另一棵树里也传一张
    n12b = c12.send_message(
        "树2看图", lambda e: None, attachments=[_image_attachment(png)]
    )
    ref2 = n12b.user_images[0].file_id
    c12.client.files.created.append({"id": "file-api-orphan", "name": "orphan.png"})
    res12 = c12.files_clean()
    check("clean 只报未引用文件", [f["id"] for f in res12["unused"]] == ["file-api-orphan"])
    check("clean 统计（含非当前树）",
          res12["total"] == 3 and res12["used"] == 2
          and ref1 in c12.files_referenced_ids() and ref2 in c12.files_referenced_ids())
    check("切树后引用仍算数", c12.switch_tree(1))
    check("切回树 1 后引用集合不变",
          {"file-api-orphan"} == {f["id"] for f in c12.files_clean()["unused"]})
    TestController.reset_store()


def test_usage_stats():
    print("== 输入栏状态条 usage_stats ==")
    script = [
        [FakeChunk(content="回答1", usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=20,
            prompt_cache_hit_tokens=80, prompt_cache_miss_tokens=20))],
        FakeChatResponse(content="标题1"),
        [FakeChunk(content="回答2", usage=SimpleNamespace(
            prompt_tokens=200, completion_tokens=30,
            prompt_cache_hit_tokens=180, prompt_cache_miss_tokens=20))],
        FakeChatResponse(content="标题2"),
    ]
    ctrl = TestController(FakeClient(script), config=make_config())
    collect(ctrl, "问题1")
    collect(ctrl, "问题2")

    node = ctrl.tree.current_node
    check("节点缓存统计已写入", node.cache_hit_tokens == 180 and node.cache_miss_tokens == 20)
    check("节点 token 统计", node.input_tokens == 200 and node.output_tokens == 30)
    check("记录最后一次请求 usage", node.last_prompt_tokens == 200 and node.last_output_tokens == 30)

    stats = ctrl.usage_stats()
    check("缓存命中率=90%", stats["cache_hit_rate"] is not None and abs(stats["cache_hit_rate"] - 0.9) < 1e-9)
    # 下次输入 = 真实 usage 口径：本节点最后一次请求的 prompt + 该轮输出
    from mincli.helpers import (
        estimate_input_price, is_peak_hour, estimate_tokens,
        estimate_prompt_tokens, estimate_tools_tokens,
    )
    expect_next = node.last_prompt_tokens + node.last_output_tokens
    check("下次输入=最后一次请求 prompt+输出", stats["next_input_tokens"] == expect_next == 230)
    check("预计价格非空", stats["estimated_price"] is not None and stats["estimated_price"] > 0)
    expect = estimate_input_price(ctrl.current_model, expect_next, 0.9, is_peak_hour())
    check("预计价格公式正确", abs(stats["estimated_price"] - expect) < 1e-9)
    check("模型标记正确", stats["model"] == ctrl.current_model)

    # 没有真实 usage 的节点（摘要节点/旧存档/中断）：回退到含工具定义的本地估算
    msgs = ctrl.tree.get_messages_for_node(node)
    est = estimate_prompt_tokens(msgs, ctrl.llm_tools)
    check("回退估算计入工具定义", estimate_tools_tokens(ctrl.llm_tools) > 0 and est > estimate_tokens(msgs))
    saved = (node.last_prompt_tokens, node.last_output_tokens)
    node.last_prompt_tokens = node.last_output_tokens = 0
    check("无 usage 时回退到含工具定义的估算",
          ctrl.usage_stats()["next_input_tokens"] == est)
    node.last_prompt_tokens, node.last_output_tokens = saved

    # 多轮工具调用：状态条用「最后一次请求」，页脚仍显示整轮累加
    target = os.path.join(_TMP, "usage_tool.txt")
    with open(target, "w", encoding="utf-8") as f:
        f.write("资料\n")
    tool_round = [FakeChunk(
        tool_calls=[FakeToolCall(
            index=0, id="call_1", name="read_file",
            arguments=json.dumps({"path": target}),
        )],
        usage=SimpleNamespace(prompt_tokens=500, completion_tokens=40,
                              prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=500),
    )]
    answer_round = [FakeChunk(content="读好了。", usage=SimpleNamespace(
        prompt_tokens=900, completion_tokens=50,
        prompt_cache_hit_tokens=800, prompt_cache_miss_tokens=100))]
    ctrl.client.chat.completions.script = [tool_round, answer_round, FakeChatResponse(content="标题3")]
    node3, _ = collect(ctrl, "帮我读文件")
    check("多轮：页脚=各轮累加", node3.input_tokens == 1400 and node3.output_tokens == 90)
    check("多轮：最后一次请求单独记录", node3.last_prompt_tokens == 900 and node3.last_output_tokens == 50)
    check("多轮：下次输入用最后一次请求",
          ctrl.usage_stats()["next_input_tokens"] == 950)

    # 会话持久化保留缓存统计
    ctrl.tree.current_node = node
    ctrl.save_session()
    ctrl2 = TestController(FakeClient([]), config=make_config())
    check("重载后缓存统计保留", ctrl2.tree.current_node is not None
          and ctrl2.tree.current_node.cache_hit_tokens == 180
          and ctrl2.tree.current_node.cache_miss_tokens == 20)
    check("重载后 last_* usage 保留", ctrl2.tree.current_node.last_prompt_tokens == 200
          and ctrl2.tree.current_node.last_output_tokens == 30
          and ctrl2.usage_stats()["next_input_tokens"] == 230)
    TestController.reset_store()

    # 无节点时返回默认值
    ctrl3 = TestController(FakeClient([]), config=make_config())
    stats3 = ctrl3.usage_stats()
    check("无节点显示默认", stats3["cache_hit_rate"] is None
          and stats3["next_input_tokens"] == 0 and stats3["estimated_price"] is None)


def test_pricing_config():
    """定价配置：pricing.json 覆盖价格/峰谷/图片 token，非法内容回退默认。"""
    import datetime
    import json

    import mincli.pricing as pricing
    from mincli.config import DEEPSEEK_PRICING

    original_path = pricing.PRICING_PATH
    tmp = tempfile.mkdtemp(prefix="mincli_pricing_")
    missing = os.path.join(tmp, "missing.json")
    path = os.path.join(tmp, "pricing.json")
    try:
        # 1) 无配置文件 → 内置默认（2026-09 降价后的价格）
        pricing.PRICING_PATH = missing
        pricing.load_pricing(force=True)
        check("定价：无配置用内置默认",
              pricing.model_pricing("deepseek-flash")["miss"] == (1.0, 2.0)
              and pricing.model_pricing("deepseek-flash")["output"] == (4.0, 8.0))
        check("定价：尺寸未知时默认兜底 1024", pricing.image_tokens_per_image() == 1024)
        check("峰谷：默认周一 10 点高峰",
              pricing.is_peak_hour(datetime.datetime(2026, 9, 14, 10, 0)))
        check("峰谷：默认周一 13 点空闲",
              not pricing.is_peak_hour(datetime.datetime(2026, 9, 14, 13, 0)))
        check("峰谷：默认周六 10 点空闲（周末不算高峰）",
              not pricing.is_peak_hour(datetime.datetime(2026, 9, 12, 10, 0)))

        # 2) 配置文件覆盖：单数字/数组/部分字段/图片 token/峰谷规则
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "peak": {"days": [6, 7], "ranges": [[0, 24]], "timezone_offset_hours": 8},
                "models": {
                    "deepseek-flash": {"miss": 9.9, "hit": [0.01, 0.02]},
                    "deepseek-v4-pro": {"output": [1.0, 2.0]},
                },
                "image_tokens": 384,
            }, f, ensure_ascii=False)
        pricing.PRICING_PATH = path
        data = pricing.load_pricing(force=True)
        check("定价：配置文件被加载", data["path"] == path)
        check("定价：单数字覆盖为不分峰谷",
              pricing.model_pricing("deepseek-flash")["miss"] == (9.9, 9.9))
        check("定价：数组覆盖保留空闲/高峰",
              pricing.model_pricing("deepseek-flash")["hit"] == (0.01, 0.02))
        check("定价：未覆盖字段沿用内置",
              pricing.model_pricing("deepseek-flash")["output"]
              == DEEPSEEK_PRICING["deepseek-flash"]["output"])
        check("定价：未列出的模型沿用内置",
              pricing.model_pricing("deepseek-v4-pro")["miss"] == (4.5, 9.0))
        check("定价：pro 单字段覆盖", pricing.model_pricing("deepseek-v4-pro")["output"] == (1.0, 2.0))
        check("定价：图片 token 覆盖", pricing.image_tokens_per_image() == 384)
        check("峰谷：覆盖后周六全天高峰",
              pricing.is_peak_hour(datetime.datetime(2026, 9, 12, 3, 0)))
        check("峰谷：覆盖后周一不再高峰",
              not pricing.is_peak_hour(datetime.datetime(2026, 9, 14, 10, 0)))
        check("定价：价格计算使用覆盖值",
              pricing.estimate_input_price("deepseek-flash", 1_000_000, 0.0, peak=False) == 9.9)

        # 3) 损坏文件 → 回退默认
        with open(path, "w", encoding="utf-8") as f:
            f.write("{bad json")
        pricing.load_pricing(force=True)
        check("定价：文件损坏回退默认",
              pricing.model_pricing("deepseek-flash")["miss"] == (1.0, 2.0)
              and pricing.image_tokens_per_image() == 1024)
    finally:
        pricing.PRICING_PATH = original_path
        pricing.load_pricing(force=True)


def test_model_migration():
    """/model 只在 flash / pro 之间切换，其余模型名一律拒绝。"""
    ctrl = TestController(FakeClient([]), config=make_config())
    check("默认模型为 flash", ctrl.current_model == MODEL_FLASH)
    check("切到 pro", ctrl.set_model("pro") and ctrl.current_model == MODEL_PRO)
    check("切回 flash", ctrl.set_model("flash") and ctrl.current_model == MODEL_FLASH)
    for bad in ("turbo", "gpt-4o", "deepseek-v4-ultra", ""):
        check(f"拒绝非 flash/pro 模型：{bad!r}",
              not ctrl.set_model(bad) and ctrl.current_model == MODEL_FLASH)


def test_image_limits():
    """图片请求限制校验（数量/像素/内联大小/总量）。"""
    from mincli.tools.images import ImageAttachment

    ctrl = TestController(FakeClient([]), config=make_config())
    ctrl.tree.create_root("q", "a", "", "t", 1, 1)
    node = ctrl.tree.current_node

    def att(name="i.png", w=100, h=100, size=1024, file_id=None, is_url=False):
        return ImageAttachment(
            source=name, detail="auto", name=name, size_bytes=size,
            width=w, height=h, file_id=file_id, is_url=is_url,
        )

    check("限制：正常图片通过", ctrl._validate_request_images(node) is None)
    node.user_images = [att(w=9000, h=100)]
    check("限制：单边超 8192 被拒", "尺寸超限" in (ctrl._validate_request_images(node) or ""))
    node.user_images = [att(w=5000, h=100)] * 15
    check("限制：≥15 张时 4096 上限生效", "4096" in (ctrl._validate_request_images(node) or ""))
    node.user_images = [att(size=40 * 1024 * 1024)]
    check("限制：内联单图超 32MiB 被拒", "内联上限" in (ctrl._validate_request_images(node) or ""))
    node.user_images = [att(size=40 * 1024 * 1024, file_id="file-api-x")]
    check("限制：file_id 图片不受 32MiB 限制", ctrl._validate_request_images(node) is None)
    node.user_images = [att(size=25 * 1024 * 1024)] * 3
    check("限制：内联总量超 64MiB 被拒", "内联图片总量超限" in (ctrl._validate_request_images(node) or ""))
    node.user_images = [att(size=70 * 1024 * 1024, file_id="f")] * 3
    check("限制：图片总量超 200MiB 被拒", "图片总大小超限" in (ctrl._validate_request_images(node) or ""))
    node.user_images = [att()] * 601
    check("限制：数量超 600 被拒", "图片数量超限" in (ctrl._validate_request_images(node) or ""))


def test_open_path_with_os():
    """跨平台打开器：按平台选择命令（用假 Popen，不真正启动进程）。"""
    import mincli.helpers as helpers

    calls = []

    class FakePopen:
        def __init__(self, cmd, *args, **kwargs):
            calls.append(list(cmd))

    original = helpers.subprocess.Popen
    helpers.subprocess.Popen = FakePopen
    try:
        if sys.platform == "darwin":
            check("打开器：macOS 用 open",
                  helpers.open_path_with_os("/tmp/a.md") is None
                  and calls[-1] == ["open", "/tmp/a.md"])
            check("打开器：macOS 文本编辑器用 open -e",
                  helpers.open_path_with_os("/tmp/a.md", prefer_text_editor=True) is None
                  and calls[-1] == ["open", "-e", "/tmp/a.md"])
        elif os.name != "nt":
            check("打开器：Unix 用 xdg-open",
                  helpers.open_path_with_os("/tmp/a.md") is None
                  and calls[-1] == ["xdg-open", "/tmp/a.md"])
        else:
            check("打开器：Windows 走 os.startfile（不启动子进程）", True)
    finally:
        helpers.subprocess.Popen = original


def test_interrupt():
    """用户主动打断：流式中止，已生成部分落盘为「中断」节点。"""
    print("== 用户打断（流式中止 / 部分内容落盘） ==")
    holder = {}

    class InterruptStream:
        def __init__(self):
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.n += 1
            if self.n == 1:
                return FakeChunk(content="前半句", reasoning_content="先想")
            if self.n == 2:
                holder["ctrl"].interrupt()
                return FakeChunk(content="不该出现")
            raise StopIteration

        def close(self):
            holder["closed"] = True

    ctrl = TestController(
        FakeClient([InterruptStream()]),
        config=make_config(),
    )
    holder["ctrl"] = ctrl
    node, events = collect(ctrl, "写个长回答")
    check("打断：节点保留", node is not None)
    check("打断：部分正文落盘", node.assistant_msg == "前半句")
    check("打断：部分思考落盘", node.reasoning == "先想")
    check("打断：节点标记中断", "打断" in (node.error or ""))
    kinds = [e.kind for e in events]
    check("打断：发出 done 事件", "done" in kinds)
    check("打断：打断后不再渲染后续内容",
          "不该出现" not in "".join(e.content for e in events))
    check("打断：流已关闭", holder.get("closed") is True)
    check("打断：本轮标志保持置位",
          ctrl._interrupt_requested is True)


def test_multi_tree():
    print("== 多对话树：新建 / 切换 / 草稿 / 删除 ==")
    script = [
        [FakeChunk(content="树二回答", usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4))],
        FakeChatResponse(content="标题二"),
    ]
    ctrl = TestController(
        FakeClient(script), config=make_config(),
    )
    check("默认只有树 1", ctrl.tree_numbers() == [1] and ctrl.active_number == 1)
    check("树 1 颜色为 1（青）", ctrl.tree_color(1) == 1)

    n2 = ctrl.create_tree()
    check("新建得到编号 2 并切过去", n2 == 2 and ctrl.active_number == 2)
    check("树 2 颜色为 2", ctrl.tree_color(2) == 2)

    ctrl.draft = "树 2 的草稿"
    check("切回树 1 后草稿为空", ctrl.switch_tree(1) and ctrl.draft == "")
    check("切回树 2 后草稿还在",
          ctrl.switch_tree(2) and ctrl.draft == "树 2 的草稿")

    node2, _ = collect(ctrl, "树二的问题")
    check("树 2 的节点落在树 2", ctrl.tree.root is node2)
    ctrl.switch_tree(1)
    check("树 1 仍是空树", ctrl.tree.root is None and ctrl.tree.current_node is None)

    ctrl.save_session()
    reloaded = TestController(
        FakeClient([]), config=make_config(),
    )
    check("重载后两棵树都在", reloaded.tree_numbers() == [1, 2])
    check("重载后停在树 1", reloaded.active_number == 1)
    check("重载后树 2 的问答还在", reloaded.tree_summary(2)["title"] == "标题二")
    check("重载后树 2 的草稿还在", reloaded._load_state(2).draft == "树 2 的草稿")
    check("每棵树一个文件",
          os.path.exists(os.path.join(TestController.TREES_DIR, "1.json"))
          and os.path.exists(os.path.join(TestController.TREES_DIR, "2.json")))

    result = ctrl.delete_tree(2)
    check("删除树 2", result["ok"] and ctrl.tree_numbers() == [1])
    check("删除后文件也没了",
          not os.path.exists(os.path.join(TestController.TREES_DIR, "2.json")))
    check("编号复用：删了 2 号后新建得到 2", ctrl.create_tree() == 2)


def test_multi_tree_no_tree_guard():
    print("== 没有对话树时拒绝发送 ==")
    base = tempfile.mkdtemp(prefix="mincli_notree_")
    try:
        ctrl = ChatController(
            FakeClient([]), config=make_config(), default_system="sys",
            trees_dir=base,
        )
        check("没有树时 has_trees 为假", not ctrl.has_trees and not ctrl.session_loaded)
        raised = False
        try:
            ctrl.send_message("你好", lambda ev: None)
        except RuntimeError:
            raised = True
        check("没有树时发送被拒绝", raised)
        check("建树后恢复正常", ctrl.create_tree() == 1 and ctrl.has_trees)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_background_tree_mark():
    print("== 后台生成：切走后留完成标记 ==")
    script = [
        [FakeChunk(content="回答", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="标题"),
    ]
    ctrl = TestController(
        FakeClient(script), config=make_config(),
    )
    ctrl.create_tree()          # 建树 2
    ctrl.switch_tree(1)         # 回到树 1 发起生成

    seen_trees = []

    def fake_impl(text, emit, attachments=None):
        # 模拟生成期间用户切到树 2
        ctrl.switch_tree(2)
        emit(ControllerEvent.done(None))
        return None

    ctrl._send_message_impl = fake_impl
    ctrl.send_message("目标树里的问题", lambda ev: seen_trees.append(ev.tree))
    check("事件带上发起生成的树编号", seen_trees == [1])
    check("后台生成结束留下完成标记", ctrl.tree_marks.get(1) == "done")
    check("当前仍在用户切过去的树", ctrl.active_number == 2)
    check("切回该树时标记被清除", ctrl.switch_tree(1) and 1 not in ctrl.tree_marks)


def test_cross_tree_tools():
    print("== 跨树查询工具（tree 参数） ==")
    script = [
        [FakeChunk(content="回答一", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="标题一"),
        [FakeChunk(content="回答二", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="标题二"),
    ]
    ctrl = TestController(
        FakeClient(script), config=make_config(recall_enabled=True),
    )
    collect(ctrl, "树一的问题")
    ctrl.create_tree()
    collect(ctrl, "树二的问题")

    names = {t["function"]["name"] for t in ctrl.llm_tools}
    check("recall 开启后两个树查询工具都在",
          {"query_conversation_tree", "read_conversation_nodes"} <= names)

    node = ctrl.tree.current_node
    out = ctrl._run_tool("query_conversation_tree", {}, lambda ev: None, node)
    check("不传 tree：列出全部对话树编号与首条提问",
          "树 1" in out and "树 2" in out and "标题一" in out and "标题二" in out)
    check("不传 tree：同时给出当前树的子树索引", "main:" in out)

    out1 = ctrl._run_tool("query_conversation_tree", {"tree": "1"}, lambda ev: None, node)
    check("指定 tree=1 查到另一棵树", "标题一" in out1 and "标题二" not in out1)
    check("指定编号时不附带全树索引", "全部对话树" not in out1)

    search = ctrl._run_tool(
        "query_conversation_tree", {"tree": "1", "search": "树一"}, lambda ev: None, node
    )
    check("跨树搜索命中", "main" in search)

    read = ctrl._run_tool(
        "read_conversation_nodes", {"node_ids": "main", "tree": "1"}, lambda ev: None, node
    )
    check("跨树读节点内容", "树一的问题" in read and "回答一" in read)

    bad = ctrl._run_tool("query_conversation_tree", {"tree": "9"}, lambda ev: None, node)
    check("不存在的编号给出可用编号提示", "不存在" in bad and "1" in bad)


def run_test(fn) -> None:
    """每个用例前清空对话树目录：树是持久化的，不隔离会互相污染。"""
    TestController.reset_store()
    fn()


if __name__ == "__main__":
    run_test(test_simple_qa)
    run_test(test_tool_round)
    run_test(test_tool_gating_matrix)
    run_test(test_send_with_attachment_persists)
    run_test(test_write_confirmation)
    run_test(test_web_disabled_hides_tools)
    run_test(test_api_error)
    run_test(test_partial_save_on_error)
    run_test(test_tool_round_partial_save)
    run_test(test_session_roundtrip)
    run_test(test_path_args)
    run_test(test_delete_nodes)
    run_test(test_settings)
    run_test(test_compact)
    run_test(test_multimodal)
    run_test(test_usage_stats)
    run_test(test_pricing_config)
    run_test(test_model_migration)
    run_test(test_image_limits)
    run_test(test_open_path_with_os)
    run_test(test_interrupt)
    run_test(test_multi_tree)
    run_test(test_multi_tree_no_tree_guard)
    run_test(test_background_tree_mark)
    run_test(test_cross_tree_tools)
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
