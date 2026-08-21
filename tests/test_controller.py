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


class FakeFiles:
    """Files API 模拟：可脚本化上传成功/失败。"""

    def __init__(self, script=None, fail=False):
        self.script = list(script or [])
        self.calls = []
        self.fail = fail
        self.created = []

    def create(self, file, purpose="user_data"):
        self.calls.append(("create", file.name, purpose))
        if self.fail:
            raise RuntimeError("模拟上传失败")
        fid = self.script.pop(0) if self.script else f"file-api-{len(self.created)}"
        self.created.append({"id": fid, "name": os.path.basename(file.name)})
        return SimpleNamespace(id=fid)

    def list(self):
        return SimpleNamespace(data=[
            SimpleNamespace(
                id=c["id"], filename=c["name"], bytes=100, created_at=1, expires_at=None
            )
            for c in self.created
        ])

    def delete(self, file_id):
        self.calls.append(("delete", file_id))
        return SimpleNamespace(deleted=True)


class FakeClient:
    def __init__(self, script, files_script=None, files_fail=False):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))
        self.files = FakeFiles(files_script, fail=files_fail)


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

    # 1) 上传成功 → file 块 + 模型自动切换
    script = [
        [FakeChunk(content="图", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5))],
        FakeChatResponse(content="图题"),
    ]
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    events = []
    added, errors = ctrl.add_pending_images([png])
    check("添加图片成功", added == 1 and not errors)
    node = ctrl.send_message("描述图片", events.append)
    check("发送成功", node is not None)
    check("file_id 已写入节点", node.user_images[0].file_id == "file-api-0")
    check("模型自动切换 vision", ctrl.current_model == "deepseek-v4-flash-vision-exp")
    check("上传调用 1 次", len(ctrl.client.files.calls) == 1)
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
    ctrl2 = TestController(FakeClient(script2, files_fail=True), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    events2 = []
    ctrl2.add_pending_images([png])
    node2 = ctrl2.send_message("看图", events2.append)
    check("回退发送成功", node2 is not None and node2.user_images[0].file_id is None)
    sent2 = ctrl2.client.chat.completions.calls[0]["messages"]
    blocks2 = sent2[-1]["content"]
    check("回退为 data URL", any(
        b.get("type") == "image_url"
        and str(b.get("image_url", {}).get("url", "")).startswith("data:image/png;base64,")
        for b in blocks2
    ))
    check("回退提示发出", any(e.kind == "status" and "内联" in e.message for e in events2))

    # 3) 自定义模型（非 flash/pro）→ 报错 + 图片放回待发送
    script3 = [
        [FakeChunk(content="x", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="t"),
    ]
    ctrl3 = TestController(FakeClient(script3), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl3.current_model = "gpt-4o"
    events3 = []
    ctrl3.add_pending_images([png])
    node3 = ctrl3.send_message("看图", events3.append)
    check("自定义模型被拒", node3 is None)
    check("图片放回待发送", len(ctrl3.pending_images) == 1)
    check("发出 error 事件", any(e.kind == "error" and "vision" in e.message for e in events3))
    check("节点已回滚", ctrl3.tree.root is None)

    # 4) 历史重放：第一轮回退 base64，第二轮补传 → file 块
    script4 = [
        [FakeChunk(content="a", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))],
        FakeChatResponse(content="题A"),
    ]
    c4 = TestController(FakeClient(script4, files_fail=True), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    c4.add_pending_images([png])
    n1 = c4.send_message("图一", lambda e: None)
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

    # 5) import_target 图片路由
    c5 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("import 图片 → 待发送", c5.import_target(png) is None
          and len(c5.pending_images) == 1 and c5.imported_content is None)
    check("import 缺失文件报错", c5.import_target("/nonexistent/x.md") is not None)

    # 6) 文件管理 + 节点删除清理
    script6 = [
        [FakeChunk(content="a", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))],
        FakeChatResponse(content="题A"),
        [FakeChunk(content="b", usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))],
        FakeChatResponse(content="题B"),
    ]
    c6 = TestController(FakeClient(script6, files_script=["file-api-abc"]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    n6a = c6.send_message("第一问", lambda e: None)
    c6.add_pending_images([png])
    n6b = c6.send_message("看图", lambda e: None)
    check("上传取脚本 id", n6b.user_images[0].file_id == "file-api-abc")
    files = c6.files_list()
    check("files_list 返回文件", len(files) == 1 and files[0]["id"] == "file-api-abc")
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
    c7 = TestController(FakeClient(script7), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    c7.add_pending_images([png])
    n7 = c7.send_message("图", lambda e: None)
    check("压缩用例发送成功", n7 is not None)
    src = c7._build_compact_source([n7])
    check("压缩源含图片占位", "[图片: m.png (800x600)]" in src)

    # 8) 会话持久化保留附件（路径与 file_id，不含 base64）
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)
    c8 = TestController(FakeClient(script6, files_script=["file-api-persist"]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    c8.add_pending_images([png])
    n8 = c8.send_message("图", lambda e: None)
    check("持久化前 file_id", n8.user_images[0].file_id == "file-api-persist")
    c8.save_session()
    c8b = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("重载后附件保留", c8b.session_loaded
          and c8b.tree.current_node is not None
          and c8b.tree.current_node.user_images
          and c8b.tree.current_node.user_images[0].file_id == "file-api-persist")
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)

    # 9) set_detail 校验
    c9 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("set_detail low", c9.set_detail("low") and c9.image_detail == "low")
    check("set_detail 非法拒绝", not c9.set_detail("huge"))
    check("set_model vision 别名", c9.set_model("vision") and c9.current_model == "deepseek-v4-flash-vision-exp")


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
    test_multimodal()
    test_usage_stats()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
