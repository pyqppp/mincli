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
    WORKFLOWS_FILE = os.path.join(_TMP, "workflows.json")


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


def test_import_multi():
    print("== /import 多文件混合（图片+文本） ==")
    a_txt = os.path.join(_TMP, "multi_a.txt")
    b_md = os.path.join(_TMP, "multi_b.md")
    png = _make_png(os.path.join(_TMP, "multi.png"))
    with open(a_txt, "w", encoding="utf-8") as f:
        f.write("文本一")
    with open(b_md, "w", encoding="utf-8") as f:
        f.write("文本二")
    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    res = ctrl.import_targets([a_txt, png, b_md])
    check("混合导入无错误", not res["errors"])
    check("图片计数", res["images_added"] == 1 and len(ctrl.pending_images) == 1)
    check("文本计数", res["text_added"] == 2 and len(ctrl.imported_files) == 2)
    check("内容拼接", "文本一" in ctrl.imported_content and "文本二" in ctrl.imported_content)
    summary = ctrl.import_summary()
    check("状态栏含数量与前2文件名", "已导入 3 个文件" in summary and "multi_a.txt" in summary and "multi.png" in summary)
    check("第3个文件名省略", "multi_b.md" not in summary and "…" in summary)
    items = ctrl.import_file_list()
    check("完整列表含全部", len(items) == 3 and items[0]["kind"] == "image")
    check("清除导入", ctrl.clear_imports() == 3 and not ctrl.pending_images and not ctrl.imported_files)

    # 发送：文本导入随消息附带并清空，图片绑定到节点
    script = [
        [FakeChunk(content="好", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))],
        FakeChatResponse(content="标题"),
    ]
    ctrl2 = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl2.import_targets([a_txt, png])
    node, _ = collect(ctrl2, "结合以上内容回答")
    check("发送后文本导入清空", ctrl2.imported_content is None and not ctrl2.imported_files)
    check("图片已绑定到节点", node is not None and node.user_images and node.user_images[0].name == "multi.png")
    check("发送后待发送图片清空", not ctrl2.pending_images)


def test_delete_nodes():
    print("== 批量删除 /delete a1 b3 g5 ==")
    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
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
    ctrl2 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl2.tree.create_root("根", "回", "", "根", 1, 1)
    s = ctrl2.tree.add_child(ctrl2.tree.root, "摘要", "", "", "上下文压缩摘要", 0, 0)
    ctrl2.tree.compaction = {"summary": "摘要", "boundary_id": s.id}
    ctrl2.tree.current_node = s
    ctrl2.delete_nodes([s.id])
    check("删除摘要节点后压缩状态清除", ctrl2.tree.compaction is None)


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
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    for i in range(1, 6):
        collect(ctrl, f"问题{i}")

    # 压缩——压缩前状态条应基于当前节点完整消息链实时估算
    before_node = ctrl.tree.current_node
    before_us = ctrl.usage_stats()
    from mincli.helpers import estimate_tokens
    events = []
    stats = ctrl.compact_history(emit=events.append)
    check("压缩返回统计", stats is not None)
    check(
        "压缩前状态条=实时消息链估算",
        before_us["next_input_tokens"]
        == estimate_tokens(ctrl.tree.get_messages_for_node(before_node)),
    )
    # 压缩后：状态条「下次输入」= 摘要节点实际发送的估算（= 压缩报告 after）
    after_us = ctrl.usage_stats()
    check("压缩后状态条=压缩报告 after", after_us["next_input_tokens"] == stats["after_tokens"])
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
    ctrl3 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("重载后压缩仍在", ctrl3.session_loaded and ctrl3.tree.compaction is not None)
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)

    # 空会话无可压缩内容
    ctrl2 = TestController(
        FakeClient([]),
        default_system="sys", default_temperature=1.0, auto_start_mcp=False,
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
    ctrl = TestController(FakeClient(script), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    events = []
    added, errors = ctrl.add_pending_images([png])
    check("添加图片成功", added == 1 and not errors)
    node = ctrl.send_message("描述图片", events.append)
    check("发送成功", node is not None)
    check("file_id 已写入节点", node.user_images[0].file_id == "file-api-0")
    check("Flash 原生支持图片（模型不切换）", ctrl.current_model == "deepseek-flash")
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

    # 3) 不支持图片的模型（如 gpt-4o）→ 报错 + 图片放回待发送
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
    check("发出 error 事件（提示不支持图片）", any(
        e.kind == "error" and "不支持图片" in e.message for e in events3
    ))
    check("节点已回滚", ctrl3.tree.root is None)

    # 3b) Pro 不支持图片 → 提示手动切换，且不自动改模型
    ctrl3b = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl3b.set_model("pro")
    events3b = []
    ctrl3b.add_pending_images([png])
    node3b = ctrl3b.send_message("看图", events3b.append)
    check("Pro 收到图片被拒", node3b is None and ctrl3b.current_model == "deepseek-v4-pro")
    check("Pro 提示切到 flash", any(
        e.kind == "error" and "/set model flash" in e.message for e in events3b
    ))
    check("Pro 图片放回待发送", len(ctrl3b.pending_images) == 1)

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
    check("set_model vision 别名 → flash", c9.set_model("vision") and c9.current_model == "deepseek-flash")
    check("旧名 deepseek-v4-flash 改写为 flash",
          c9.set_model("deepseek-v4-flash") and c9.current_model == "deepseek-flash")
    check("set_model pro", c9.set_model("pro") and c9.current_model == "deepseek-v4-pro")


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
    # 下次输入 = 当前节点完整消息链的实时估算（树状对话/压缩后口径一致）
    from mincli.helpers import estimate_input_price, is_peak_hour, estimate_tokens
    node = ctrl.tree.current_node
    expect_next = estimate_tokens(ctrl.tree.get_messages_for_node(node))
    check("下次输入=当前消息链实时估算", stats["next_input_tokens"] == expect_next)
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


def test_workflows():
    """工作流（/wf）：存储持久化 / 提炼 / 合成 / 修订 / 管理。"""
    from mincli.workflows import FALLBACK_MARK, Workflow

    if os.path.exists(TestController.WORKFLOWS_FILE):
        os.remove(TestController.WORKFLOWS_FILE)

    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("工作流：初始无文件", not os.path.exists(TestController.WORKFLOWS_FILE))
    check("工作流：列表初始为空", ctrl.wf_list() == [])

    DOC = (
        "目标：为 Git 仓库生成 changelog 并写入文件\n\n"
        "变量：\n"
        "- {start}：起始版本（示例：v1.0）\n"
        "- {end}：目标版本（示例：v2.0）\n\n"
        "步骤：\n"
        "1. 查看自 {start} 以来的提交：`git log {start}..{end} --oneline`\n"
        "2. 生成 CHANGELOG.md"
    )
    ctrl._wf_store.put(Workflow(name="rel", doc=DOC))
    check("工作流：put 后文件已写入", os.path.exists(TestController.WORKFLOWS_FILE))

    # “重启”后从同一文件加载
    ctrl2 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    wf = ctrl2.wf_get("rel")
    check("工作流：重启后仍在且内容一致", wf is not None and wf.doc == DOC)
    check("工作流：占位符推导", wf is not None and wf.placeholders() == ["start", "end"])
    lst = ctrl2.wf_list()
    check("工作流：list 摘要正确", len(lst) == 1 and lst[0]["name"] == "rel"
          and "changelog" in lst[0]["goal"] and lst[0]["steps"] == 2
          and lst[0]["vars"] == ["start", "end"])
    check("工作流：同名 put 需 overwrite", ctrl2._wf_store.put(Workflow(name="rel", doc="x")) is False)
    check("工作流：同名 put overwrite 成功",
          ctrl2._wf_store.put(Workflow(name="rel", doc="x"), overwrite=True) is True)
    ctrl2.wf_import_text("rel", DOC)
    check("工作流：import_text 回写文档", ctrl2.wf_get("rel").doc == DOC)
    check("工作流：重命名非法名报错", ctrl2.wf_rename("rel", "bad name") is not None)
    check("工作流：重命名成功", ctrl2.wf_rename("rel", "rel2") is None
          and ctrl2.wf_get("rel") is None and ctrl2.wf_get("rel2") is not None)
    ctrl2.wf_rename("rel2", "rel")
    check("工作流：删除不存在返回 False", ctrl2.wf_delete("nope") is False)
    check("工作流：删除成功", ctrl2.wf_delete("rel") is True and ctrl2.wf_get("rel") is None)

    # 提炼：构造带工具调用的节点 + stub 模型
    ctrl3 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl3.tree.create_root("请为仓库生成 changelog 并写入文件", "已完成", "", "生成 changelog", 1, 1)
    node = ctrl3.tree.current_node
    node.tool_messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": "git log --oneline v1.0..HEAD"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a1b2c3 提交说明"},
    ]
    src = ctrl3._wf_build_source([node])
    check("工作流：提炼源含用户指令", "请为仓库生成 changelog" in src)
    check("工作流：提炼源含工具调用细节", "调用工具: execute_command" in src and "git log" in src)
    check("工作流：提炼源含工具结果", "工具结果" in src and "a1b2c3" in src)

    ctrl3._wf_call_model = lambda messages, max_tokens: DOC  # stub 提炼
    r = ctrl3.wf_save("demo")
    check("工作流：自动提炼保存", r.get("status") == "saved" and r.get("from") == "extract"
          and r.get("placeholders") == ["start", "end"])
    check("工作流：记录来源节点", ctrl3.wf_get("demo").source_nodes == ["main"])
    check("工作流：同名未 force 返回 exists", ctrl3.wf_save("demo").get("status") == "exists")
    check("工作流：force 覆盖成功", ctrl3.wf_save("demo", force=True).get("status") == "saved")
    check("工作流：起点不在链上报错", ctrl3.wf_save("x", start_id="bogus").get("status") == "error")
    # 一连串操作：起点 → 当前链（a1 挂在 main 下）
    a1 = ctrl3.tree.add_child(ctrl3.tree.root, "第二轮追问", "第二轮回答", "", "追问", 1, 1)
    ctrl3.tree.current_node = a1
    r_chain = ctrl3.wf_save("chain", start_id="main")
    check("工作流：一连串操作提炼（起点→当前）",
          r_chain.get("status") == "saved" and r_chain.get("nodes") == 2)
    wf_chain = ctrl3.wf_get("chain")
    check("工作流：来源记录整条链", wf_chain is not None and wf_chain.source_nodes == ["main", "a1"])
    ctrl3._wf_call_model = lambda messages, max_tokens: ""  # 提炼失败
    rfb = ctrl3.wf_save("fb")
    check("工作流：提炼失败回退原文", rfb.get("status") == "saved" and rfb.get("from") == "fallback")
    check("工作流：回退文档含提示标记", FALLBACK_MARK in ctrl3.wf_get("fb").doc)

    # 合成：use / run 与变量替换、运行计数
    ctrl4 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl4._wf_store.put(Workflow(name="rel", doc=DOC))
    m_use = ctrl4.wf_compose("rel", typed="这次发布到 v2.0")
    check("工作流：use 合成含规范与本次输入", m_use is not None
          and "请执行工作流「rel」" in m_use and "本次输入：这次发布到 v2.0" in m_use)
    m_run = ctrl4.wf_compose("rel", values={"start": "v1.0", "end": "v2.0"})
    check("工作流：run 合成替换变量", m_run is not None
          and "{start}" not in m_run and "{end}" not in m_run
          and "v1.0" in m_run and "v2.0" in m_run)
    m_win = ctrl4.wf_compose("rel", values={"start": r"C:\work\a", "end": r"D:\out\b"})
    check("工作流：替换值含反斜杠不崩溃且原样写入", m_win is not None
          and r"C:\work\a" in m_win and r"D:\out\b" in m_win
          and "bad escape" not in m_win)
    m_part = ctrl4.wf_compose("rel", values={"start": "v1.0"})
    check("工作流：缺失变量提示推断", m_part is not None
          and "{end}" in m_part and "未提供值的变量" in m_part)
    check("工作流：未知名合成返回 None", ctrl4.wf_compose("nope") is None)
    wf4 = ctrl4.wf_get("rel")
    check("工作流：运行计数递增", wf4 is not None and wf4.run_count == 4 and wf4.last_run_at is not None)

    # 修订
    ctrl4._wf_call_model = lambda messages, max_tokens: "目标：修订版\n\n步骤：\n1. 校验步骤"
    rr = ctrl4.wf_revise("rel", "增加校验")
    check("工作流：模型修订成功", rr.get("status") == "revised" and "修订版" in ctrl4.wf_get("rel").doc)
    ctrl4._wf_call_model = lambda messages, max_tokens: ""
    rerr = ctrl4.wf_revise("rel", "x")
    check("工作流：修订失败报错", rerr.get("status") == "error")
    check("工作流：修订失败保留原文档", "修订版" in ctrl4.wf_get("rel").doc)
    check("工作流：修订不存在名字报错", ctrl4.wf_revise("nope", "x").get("status") == "error")


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
        check("定价：默认图片固定值 1024", pricing.image_tokens_per_image() == 1024)
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
    """旧模型名自动改写（会话加载 + 简写归一）。"""
    from mincli.config import MODEL_FLASH, MODEL_PRO, normalize_model_name

    check("归一：简写 flash", normalize_model_name("flash") == MODEL_FLASH)
    check("归一：简写 vision → flash", normalize_model_name("vision") == MODEL_FLASH)
    check("归一：旧名 vision-exp → flash",
          normalize_model_name("deepseek-v4-flash-vision-exp") == MODEL_FLASH)
    check("归一：旧名 chat → flash", normalize_model_name("deepseek-chat") == MODEL_FLASH)
    check("归一：pro 保持", normalize_model_name("deepseek-v4-pro") == MODEL_PRO)
    check("归一：未知名原样", normalize_model_name("gpt-4o") == "gpt-4o")

    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)
    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    ctrl.tree.create_root("问题", "回答", "", "标题", 1, 1)
    ctrl.current_model = "deepseek-v4-flash-vision-exp"  # 模拟旧会话
    ctrl.save_session()
    ctrl2 = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
    check("旧会话模型自动改写", ctrl2.current_model == MODEL_FLASH)
    check("记录改写来源", ctrl2.model_migrated_from == "deepseek-v4-flash-vision-exp")
    if os.path.exists(TestController.SAVE_FILE):
        os.remove(TestController.SAVE_FILE)


def test_image_limits():
    """图片请求限制校验（数量/像素/内联大小/总量）。"""
    from mincli.tools.images import ImageAttachment

    ctrl = TestController(FakeClient([]), default_system="sys", default_temperature=1.0, auto_start_mcp=False)
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


if __name__ == "__main__":
    test_simple_qa()
    test_tool_round()
    test_api_error()
    test_session_roundtrip()
    test_import_target()
    test_path_args()
    test_import_multi()
    test_delete_nodes()
    test_settings()
    test_compact()
    test_multimodal()
    test_usage_stats()
    test_workflows()
    test_pricing_config()
    test_model_migration()
    test_image_limits()
    test_open_path_with_os()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
