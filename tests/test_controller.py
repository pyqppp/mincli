"""ChatController 单元测试（fake OpenAI client，不联网）。

运行：`venv/bin/python -m tests.test_controller`
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mincli.controller import ChatController, ControllerEvent

PASS = 0
FAIL = 0

_TMP = tempfile.mkdtemp(prefix="mincli_test_")


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


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


class FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))


# ---------------- 测试用控制器 ----------------

class TestController(ChatController):
    SAVE_FILE = os.path.join(_TMP, "session.json")


def collect(ctrl, text):
    """调用 send_message 并收集事件。"""
    events = []

    def emit(ev: ControllerEvent):
        events.append(ev)

    node = ctrl.send_message(text, emit)
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
    ctrl = TestController(FakeClient(script), default_system="你是助手", default_temperature=1.0, auto_start_mcp=False)
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
    print("== 工具调用轮（write_file → 继续问答） ==")
    tool_chunks = [
        FakeChunk(
            tool_calls=[
                FakeToolCall(
                    index=0,
                    id="call_1",
                    name="write_file",
                    arguments='{"filepath": "/tmp/mincli_test.txt", "content": "hi"}',
                )
            ]
        )
    ]
    content_chunks = [FakeChunk(content="文件已写入。")]
    script = [tool_chunks, content_chunks, FakeChatResponse(content="写文件测试")]
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl.confirm = lambda title, text: True  # 允许确认
    node, events = collect(ctrl, "帮我写个文件")
    check("工具轮返回节点", node is not None)
    check("最终回答", node.assistant_msg == "文件已写入。")
    tool_events = [e for e in events if e.kind == "tool"]
    check("工具事件成对（开始+结果）", len(tool_events) == 2)
    check("工具名正确", tool_events[0].tool_name == "write_file")
    check("工具结果摘要", tool_events[1].tool_summary == "写文件工具不可用")
    # 验证发给模型的消息包含 tool 结果
    calls = ctrl.client.chat.completions.calls
    msgs2 = calls[1]["messages"]
    roles = [m["role"] for m in msgs2]
    check("第二轮消息含 assistant+tool", "assistant" in roles and "tool" in roles)
    check("tool 消息携带 tool_call_id", any(m.get("tool_call_id") == "call_1" for m in msgs2 if m["role"] == "tool"))
    check("工具轮消息写入节点", bool(node.tool_messages))


def test_api_error():
    print("== API 错误处理 ==")
    script = [RuntimeError("connection reset")]
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    node, events = collect(ctrl, "hello")
    check("返回 None", node is None)
    check("出错回滚空节点", ctrl.tree.root is None and ctrl.tree.current_node is None)
    errs = [e for e in events if e.kind == "error"]
    check("发出 error 事件", len(errs) == 1 and "connection reset" in errs[0].message)


def test_session_roundtrip():
    print("== 会话持久化 ==")
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)
    script = [[FakeChunk(content="第一轮")], FakeChatResponse(content="标题A")]
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("首次无已存会话", not ctrl.session_loaded)
    collect(ctrl, "第一问")
    check("save_session 成功", ctrl.save_session())
    check("文件已写入", os.path.exists(TestController.SAVE_FILE))

    script2 = [[FakeChunk(content="第二轮")], FakeChatResponse(content="标题B")]
    ctrl2 = TestController(FakeClient(script2), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("重新加载会话", ctrl2.session_loaded)
    check("树已恢复", ctrl2.tree.root is not None and ctrl2.tree.root.title == "标题A")
    node, _ = collect(ctrl2, "第二问")
    check("在已有树上追加节点", node is not None and node.parent_id is not None and node.id != "main")
    os.remove(TestController.SAVE_FILE)


def test_import_target():
    print("== /import 导入 ==")
    src = os.path.join(_TMP, "sample.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("导入的上下文内容")
    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("导入文件成功", ctrl.import_target(src) is None)
    check("imported_content 已设置", ctrl.imported_content is not None and "导入的上下文内容" in ctrl.imported_content)
    check("导入不存在文件返回错误", ctrl.import_target("/nonexistent/xxx.md") is not None)


def test_settings():
    print("== 设置 ==")
    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("切换模型 flash", ctrl.set_model("flash") and ctrl.current_model.endswith("flash"))
    check("切换模型 pro", ctrl.set_model("pro") and ctrl.current_model.endswith("pro"))
    check("非法模型拒绝", not ctrl.set_model("turbo"))
    check("effort 校验", ctrl.set_effort("max") and not ctrl.set_effort("extreme"))
    check("audit 校验", ctrl.set_audit(4) and not ctrl.set_audit(9))
    ctrl.set_thinking(True)
    check("thinking 设置", ctrl.thinking_enabled)
    ctrl.set_system("新系统提示词")
    check("system 同步到树", ctrl.current_system == "新系统提示词" and ctrl.tree.system_prompt == "新系统提示词")


if __name__ == "__main__":
    test_simple_qa()
    test_tool_round()
    test_api_error()
    test_session_roundtrip()
    test_import_target()
    test_settings()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
