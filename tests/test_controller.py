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


def test_compact():
    print("== 上下文压缩 /compact ==")
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
    script.append(FakeChatResponse(content="【摘要2】keep=0 全量压缩"))
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    for i in range(1, 6):
        collect(ctrl, f"问题{i}")

    # 压缩（保留最近 2 轮）——压缩前状态条应等于「上次输入+输出」（API 口径）
    before_us = ctrl.usage_stats()
    events = []
    stats = ctrl.compact_history(keep=2, emit=events.append)
    check("压缩返回统计", stats is not None)
    check(
        "压缩前状态条=上次输入+输出",
        before_us["next_input_tokens"]
        == ctrl.tree.current_node.input_tokens + ctrl.tree.current_node.output_tokens,
    )
    # 压缩后：状态条「下次输入」= 压缩报告 after_tokens（三者自洽）
    after_us = ctrl.usage_stats()
    check("压缩后状态条=压缩报告 after", after_us["next_input_tokens"] == stats["after_tokens"])
    check("压缩后状态条显著变小", after_us["next_input_tokens"] < before_us["next_input_tokens"])
    check("压缩 3 轮", stats["nodes_compressed"] == 3)
    check("保留 2 轮", stats["nodes_kept"] == 2)
    check("摘要写入树", ctrl.tree.compaction is not None and ctrl.tree.compaction["summary"].startswith("【摘要】"))
    check("发出 status 事件", any(e.kind == "status" for e in events))
    check("节省 token>0", stats["saved_tokens"] > 0)

    path = ctrl._path_to_root(ctrl.tree.current_node)
    check("boundary=倒数第 3 个节点", stats["boundary_id"] == path[-3].id)

    msgs = ctrl.tree.get_messages_for_node(ctrl.tree.current_node)
    joined = "\n".join(str(m.get("content", "")) for m in msgs)
    check("消息含摘要", "【摘要】" in joined)
    check("旧内容已压缩", "回答1" not in joined and "回答2" not in joined and "回答3" not in joined)
    check("新内容保留", "回答4" in joined and "回答5" in joined)
    check("摘要带前缀标记", msgs[1]["content"].startswith("【以下是本对话早期内容"))

    # 压缩后继续对话：发给模型的消息应使用摘要
    node, _ = collect(ctrl, "新问题")
    check("压缩后继续对话成功", node is not None)
    sent_joined = ""
    for call in ctrl.client.chat.completions.calls:
        msgs = call.get("messages", [])
        if msgs and msgs[-1].get("role") == "user" and str(msgs[-1].get("content", "")).strip() == "新问题":
            sent_joined = "\n".join(str(m.get("content", "")) for m in msgs)
    check("发给模型的消息含摘要", "【摘要】" in sent_joined)
    check("发给模型的消息不含旧回答", "回答1" not in sent_joined and "回答3" not in sent_joined)

    # /compact off 恢复原文
    check("清除压缩", ctrl.clear_compaction())
    msgs2 = ctrl.tree.get_messages_for_node(ctrl.tree.current_node)
    joined2 = "\n".join(str(m.get("content", "")) for m in msgs2)
    check("恢复后含完整原文", "回答1" in joined2 and "回答新" in joined2)
    check("恢复后无摘要", "【摘要】" not in joined2)

    # keep=0 全部压缩
    stats0 = ctrl.compact_history(keep=0)
    check("keep=0 压缩全部 6 轮", stats0 is not None and stats0["nodes_compressed"] == 6 and stats0["nodes_kept"] == 0)

    # 压缩随会话持久化
    ctrl.save_session()
    ctrl3 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("重载后压缩仍在", ctrl3.session_loaded and ctrl3.tree.compaction is not None)
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)

    # 对话太短
    ctrl2 = TestController(
        FakeClient([[FakeChunk(content="仅一轮")], FakeChatResponse(content="标题")]),
        default_system="sys", default_temperature=1.0, auto_start_mcp=False,
    )
    collect(ctrl2, "问题")
    check("太短返回 None", ctrl2.compact_history() is None)


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
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    collect(ctrl, "问题1")
    collect(ctrl, "问题2")

    node = ctrl.tree.current_node
    check("节点缓存统计已写入", node.cache_hit_tokens == 180 and node.cache_miss_tokens == 20)
    check("节点 token 统计", node.input_tokens == 200 and node.output_tokens == 30)

    stats = ctrl.usage_stats()
    check("缓存命中率=90%", stats["cache_hit_rate"] is not None and abs(stats["cache_hit_rate"] - 0.9) < 1e-9)
    # 未压缩时：下次输入 = 上次完整输入 + 本节点输出（API 口径，与 done 显示对应）
    from mincli.helpers import estimate_input_price, is_peak_hour
    node = ctrl.tree.current_node
    expect_next = node.input_tokens + node.output_tokens
    check("下次输入=上次输入+本节点输出", stats["next_input_tokens"] == expect_next)
    check("预计价格非空", stats["estimated_price"] is not None and stats["estimated_price"] > 0)
    expect = estimate_input_price(ctrl.current_model, expect_next, 0.9, is_peak_hour())
    check("预计价格公式正确", abs(stats["estimated_price"] - expect) < 1e-9)
    check("模型标记正确", stats["model"] == ctrl.current_model)

    # 会话持久化保留缓存统计
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)
    ctrl.save_session()
    ctrl2 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("重载后缓存统计保留", ctrl2.tree.current_node is not None
          and ctrl2.tree.current_node.cache_hit_tokens == 180
          and ctrl2.tree.current_node.cache_miss_tokens == 20)
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)

    # 无节点时返回默认值
    ctrl3 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    stats3 = ctrl3.usage_stats()
    check("无节点显示默认", stats3["cache_hit_rate"] is None
          and stats3["next_input_tokens"] == 0 and stats3["estimated_price"] is None)


if __name__ == "__main__":
    test_simple_qa()
    test_tool_round()
    test_api_error()
    test_session_roundtrip()
    test_import_target()
    test_settings()
    test_compact()
    test_usage_stats()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
