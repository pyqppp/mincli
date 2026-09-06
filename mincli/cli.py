"""mincli CLI 入口（Typer）。

默认启动 Textual TUI；`--no-tui` 时退化为极简纯文本对话（无 Rich / 无
prompt_toolkit / 无流式渲染依赖）。
"""

import os
import shlex
import sys
from typing import Optional

import typer
from openai import OpenAI

from mincli.tools.files import FilesAPIError

from mincli.config import (
    MODEL_V4_FLASH,
    MODEL_V4_PRO,
    MODEL_V4_VISION,
    SAVE_BASE_DIR,
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SOURCE,
    MODELS_AVAILABLE,
    API_PROVIDERS,
    load_models,
    register_model,
    get_model_base_url,
    get_model_key_var,
)

app = typer.Typer(help="mincli - 树状对话 AI 助手")


def resolve_api_key(provider: str, model: str) -> str:
    """按 provider/模型解析 API Key：优先 provider 对应环境变量，回退 DEEPSEEK_API_KEY。"""
    key_var = get_model_key_var(provider, model)
    api_key = os.getenv(key_var) or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print(f"错误: 未设置 {key_var}（或 DEEPSEEK_API_KEY），请在 .env 或环境变量中配置")
        raise typer.Exit(1)
    return api_key


def resolve_model_name(model: str) -> str:
    """把简写 flash/pro/vision 映射为完整模型名。"""
    arg = (model or "").lower()
    if arg in ("flash", "v4-flash", "f"):
        return MODEL_V4_FLASH
    if arg in ("pro", "v4-pro", "p"):
        return MODEL_V4_PRO
    if arg in ("vision", "v-flash-vision", "v4-vision"):
        return MODEL_V4_VISION
    return model


def build_controller(
    provider: str,
    model: str,
    temperature: float,
    thinking: bool,
    effort: str,
) -> "ChatController":
    """按 CLI 参数构造 ChatController（支持多 Provider/多模型，惰性导入）。"""
    from mincli.controller import ChatController

    effective_model = resolve_model_name(model)
    base_url = get_model_base_url(provider, effective_model)
    api_key = resolve_api_key(provider, effective_model)

    return ChatController(
        client=OpenAI(api_key=api_key, base_url=base_url),
        default_system=DEFAULT_SYSTEM_PROMPT,
        default_temperature=temperature,
        default_model=effective_model,
        thinking_enabled=thinking,
        reasoning_effort=effort,
    )


@app.command()
def chat(
    provider: str = typer.Option("deepseek", "-p", "--provider", help="API Provider（deepseek/openai 或已注册的自定义 provider）"),
    model: str = typer.Option("flash", "-m", "--model", help="模型: flash / pro / 完整模型名（如 gpt-4o）"),
    temperature: float = typer.Option(1.0, "--temp", "-temp-opt", help="温度参数"),
    thinking: bool = typer.Option(False, "--thinking", "-r", help="开启思考模式（默认 high）"),
    effort: str = typer.Option("high", "--effort", help="推理强度: low, high 或 max"),
    no_tui: bool = typer.Option(False, "--no-tui", help="不使用 TUI，用极简纯文本对话"),
) -> None:
    """启动树状对话（默认 Textual TUI），支持多 Provider/多模型。"""
    if effort not in ("low", "high", "max"):
        print(f"无效推理强度: {effort}，可选 low / high / max")
        raise SystemExit(2)

    if no_tui:
        _chat_plain(provider, model, temperature, thinking, effort)
        return

    from mincli.tui.app import ChatApp

    ChatApp(controller=build_controller(provider, model, temperature, thinking, effort)).run()


@app.command("register")
def register(
    model: str = typer.Argument(..., help="模型名（如 gpt-4o、claude-3-5-sonnet），注册后可用 -m 调用"),
    url: str = typer.Argument(..., help="API base URL（OpenAI 兼容端点，如 https://api.openai.com/v1）"),
    provider: str = typer.Option("deepseek", "-p", "--provider", help="Provider 名（决定默认 API Key 环境变量）"),
    api_key_var: str = typer.Option(None, "-k", "--key-var", help="API Key 环境变量名（默认取 provider 映射）"),
) -> None:
    """注册一个新的模型配置到 ~/.mincli/models.json。"""
    if register_model(provider, model, url, api_key_var):
        print(f"✅ 已注册模型「{model}」")
        print(f"   URL: {url}")
        print(f"   API Key 环境变量: {api_key_var or API_PROVIDERS.get(provider, 'DEEPSEEK_API_KEY')}")
        print(f"   使用: mincli chat -m {model}")
    else:
        print("❌ 注册失败（请检查 ~/.mincli 目录写权限）")
        raise SystemExit(1)


@app.command("models")
def list_models() -> None:
    """列出所有可用模型（内置 + 已注册）。"""
    registered = load_models()
    print("内置模型:")
    for name, url in MODELS_AVAILABLE.items():
        print(f"  - {name}  ({url})")
    if registered:
        print("\n已注册模型:")
        for name, cfg in registered.items():
            print(f"  - {name}  ({cfg.get('url')})  [Key: {cfg.get('key_var')}]")
    else:
        print("\n（无已注册模型，可用 `mincli register <模型名> <URL>` 添加）")


_WF_PLAIN_USAGE = (
    "用法: /wf list | /wf show <名> | /wf save <名> [起点节点ID] | "
    "/wf use <名> | /wf stop | /wf run <名> [键=值...] | "
    "/wf edit <名> <修改要求> | /wf rename <旧> <新> | /wf delete <名>"
)


def _plain_wf(ctrl, text: str) -> Optional[str]:
    """纯文本模式 /wf 分发（use/stop 由调用方就地处理）。

    返回需要发送执行的消息文本（/wf run），其余命令打印结果并返回 None。
    """
    try:
        parts = shlex.split(text)
    except ValueError:
        print("参数解析失败（引号不匹配）")
        return None
    if len(parts) < 2:
        print(_WF_PLAIN_USAGE)
        return None
    sub = parts[1].lower()
    name = parts[2] if len(parts) > 2 else ""

    if sub in ("list", "ls"):
        data = ctrl.wf_list()
        if not data:
            print("（暂无工作流）用法: /wf save <名> 把当前操作保存为工作流")
            return None
        for item in data:
            goal = (item.get("goal") or "—")[:40]
            print(
                f"{item['name']}  |  {goal}  |  步骤 {item.get('steps', 0)} / "
                f"变量 {len(item.get('vars') or [])} / 运行 {item.get('run_count', 0)}"
            )
        print("使用: /wf use <名> 挂载下次输入 | /wf run <名> 立即执行 | /wf show <名> 查看")
        return None
    if sub == "show":
        wf = ctrl.wf_get(name)
        if wf is None:
            print(f"⚠️ 工作流「{name}」不存在")
        else:
            print(f"工作流 {wf.name}（运行 {wf.run_count} 次）\n{wf.doc}")
        return None
    if sub == "save":
        if not name:
            print("用法: /wf save <名> [起点节点ID]")
            return None
        start_id = parts[3] if len(parts) > 3 else None
        if ctrl.wf_get(name) is not None and not ctrl.confirm(
            "覆盖工作流", f"工作流「{name}」已存在，覆盖旧版本？"
        ):
            print("已取消")
            return None
        print("正在从对话提炼工作流…")
        res = ctrl.wf_save(name, start_id, True)
        if res.get("status") == "error":
            print(f"⚠️ {res.get('message')}")
        elif res.get("from") == "fallback":
            print(f"已保存（未能自动提炼，原始记录已存档，可 /wf edit {name} <修改要求> 修正）")
        else:
            print(
                f"✅ 已提炼保存工作流「{name}」（{res.get('nodes', 0)} 轮 / "
                f"{len(res.get('placeholders') or [])} 个变量）"
            )
        return None
    if sub == "run":
        if not name:
            print("用法: /wf run <名> [键=值...]（位置参数按变量顺序填充）")
            return None
        wf = ctrl.wf_get(name)
        if wf is None:
            print(f"⚠️ 工作流「{name}」不存在")
            return None
        values: dict = {}
        positionals: list = []
        for tok in parts[3:]:
            if "=" in tok:
                k, _, v = tok.partition("=")
                values[k.strip()] = v
            else:
                positionals.append(tok)
        if positionals:
            phs = wf.placeholders()
            for i, pv in enumerate(positionals):
                if i >= len(phs):
                    break
                key = phs[i]
                if key not in values:
                    values[key] = pv
            if len(positionals) > len(phs):
                print("⚠️ 多余的位置参数已忽略")
        composed = ctrl.wf_compose(name, values=values)
        if composed is None:
            print(f"⚠️ 工作流「{name}」不存在")
            return None
        print(f"▶ 已开始执行工作流「{name}」")
        return composed
    if sub == "delete":
        if not name:
            print("用法: /wf delete <名>")
            return None
        if not ctrl.confirm("删除工作流", f"确定删除工作流「{name}」吗？此操作不可恢复。"):
            print("已取消")
        elif ctrl.wf_delete(name):
            print(f"已删除工作流「{name}」")
        else:
            print(f"⚠️ 删除失败：工作流「{name}」不存在")
        return None
    if sub == "rename":
        new_name = parts[3] if len(parts) > 3 else ""
        if not name or not new_name:
            print("用法: /wf rename <旧名> <新名>")
            return None
        err = ctrl.wf_rename(name, new_name)
        if err:
            print(f"⚠️ {err}")
        else:
            print(f"已重命名：{name} → {new_name}")
        return None
    if sub == "edit":
        if not name:
            print("用法: /wf edit <名> <修改要求>")
            return None
        if len(parts) < 4:
            print("纯文本模式不支持打开编辑器；请用 /wf edit <名> <修改要求> 由模型修订")
            return None
        request = " ".join(parts[3:])
        print("正在按你的要求修订工作流…")
        res = ctrl.wf_revise(name, request)
        if res.get("status") == "error":
            print(f"⚠️ {res.get('message')}")
        else:
            print(f"✅ 工作流「{name}」已按你的要求更新")
        return None
    print(_WF_PLAIN_USAGE)
    return None


def _chat_plain(provider: str, model: str, temperature: float, thinking: bool, effort: str) -> None:
    """极简纯文本对话：input() 逐行输入，无 Rich / prompt_toolkit 依赖。"""
    from mincli.controller import ControllerEvent

    ctrl = build_controller(provider, model, temperature, thinking, effort)

    def emit(ev: ControllerEvent) -> None:
        if ev.kind == "stream":
            if ev.reasoning:
                print(f"\n🧠 {ev.reasoning}")
            if ev.content:
                print(ev.content, end="", flush=True)
        elif ev.kind == "status":
            print(f"\n[{ev.message}]")
        elif ev.kind == "tool":
            print(f"\n[工具: {ev.tool_name}]")
        elif ev.kind == "error":
            print(f"\n⚠️ {ev.message}")
        elif ev.kind == "done":
            print()

    ctrl.confirm = lambda title, text: input(f"{title}: {text} (y/N): ").strip().lower() in ("y", "yes")

    print("mincli 纯文本模式（输入 /exit 退出，/help 查看命令）")
    pending_wf: Optional[str] = None  # /wf use 挂载到下一次输入的工作流名
    try:
        while True:
            try:
                line = input("你> ")
            except EOFError:
                break
            text = line.strip()
            if not text:
                continue
            low = text.lower()
            if low in ("/exit", "/quit", "/q"):
                break
            if low in ("/help", "/h"):
                print("命令: /exit 退出 | /clear 清空 | /compact 压缩上下文（新建摘要节点） | /tree 显示对话树 | /info 节点详情 | /import 导入文件/图片 | /files 管理图片文件 | /wf 工作流（list/save/use/run 等）")
                continue
            if low.startswith("/import"):
                try:
                    parts = shlex.split(text)
                except ValueError:
                    print("参数解析失败（引号不匹配）")
                    continue
                targets = parts[1:]
                if not targets:
                    print("用法: /import <路径或URL> [...] | /import clear")
                elif targets[0].lower() in ("clear", "c"):
                    n = ctrl.clear_imports()
                    print(f"已清除 {n} 个待导入文件")
                else:
                    res = ctrl.import_targets(targets)
                    bits = []
                    if res["images_added"]:
                        bits.append(f"{res['images_added']} 张图片")
                    if res["text_added"]:
                        bits.append(f"{res['text_added']} 个文本/网页")
                    print(f"✅ 已导入 {'、'.join(bits)}（发送时自动附带）" if bits else "未导入任何内容")
                    for err in res["errors"]:
                        print(f"⚠️ {err}")
                continue
            if low.startswith("/files"):
                parts = text.split(maxsplit=2)
                sub = parts[1].lower() if len(parts) > 1 else "list"
                try:
                    if sub in ("list", "ls", ""):
                        files = ctrl.files_list()
                        if not files:
                            print("（无已上传图片文件）")
                        else:
                            for f in files:
                                print(f"{f['id']}  {f['name']}  {f['bytes'] / 1024 / 1024:.2f} MiB")
                    elif sub in ("delete", "rm", "del") and len(parts) == 3:
                        ctrl.files_delete(parts[2])
                        print(f"✅ 已删除文件 {parts[2]}")
                    else:
                        print("用法: /files list | /files delete <ID>")
                except FilesAPIError as e:
                    print(f"⚠️ {e}")
                continue
            if low == "/clear":
                ctrl.reset()
                print("已清空当前会话")
                continue
            if low.startswith("/compact"):
                if not ctrl.tree or ctrl.tree.current_node is None:
                    print("当前没有对话可压缩")
                    continue
                if ctrl.tree.compaction and ctrl.tree.compaction.get("boundary_id") == ctrl.tree.current_node.id:
                    print("当前节点已是压缩摘要节点（切换回其他节点仍使用完整历史）")
                    continue
                print("正在压缩上下文…")
                stats = ctrl.compact_history()
                if stats is None:
                    print("无可压缩的对话（或压缩失败）")
                elif stats.get("blocked"):
                    print("当前节点已是压缩摘要节点")
                else:
                    print(
                        f"✅ 已压缩 {stats['nodes_compressed']} 轮 → 新节点 {stats['node_id']}（摘要）；"
                        f"Token {stats['before_tokens']} → {stats['after_tokens']}（节省 {stats['saved_tokens']}）"
                    )
                continue
            if low.startswith("/set"):
                parts = text.split(maxsplit=2)
                sub = parts[1].lower() if len(parts) > 1 else ""
                if sub == "file_confirm" and len(parts) == 3:
                    arg = parts[2].lower()
                    if arg in ("on", "1", "true"):
                        ctrl.set_file_confirm(True)
                        print("✅ 写文件/编辑文件确认已开启")
                    elif arg in ("off", "0", "false"):
                        ctrl.set_file_confirm(False)
                        print("⚠️ 写文件/编辑文件确认已关闭（AI 可直接写入/修改文件）")
                    else:
                        print("用法: /set file_confirm <on|off>")
                elif sub == "show":
                    print(
                        f"模型: {ctrl.current_model} | 温度: {ctrl.current_temperature} | "
                        f"思考: {'开' if ctrl.thinking_enabled else '关'} | "
                        f"审核: {ctrl.audit_level} | "
                        f"文件确认: {'开' if ctrl.file_confirm else '关'}"
                    )
                else:
                    print("用法: /set file_confirm <on|off> | /set show")
                continue
            if low.startswith(("/wf", "/workflow")):
                try:
                    wparts = shlex.split(text)
                except ValueError:
                    print("参数解析失败（引号不匹配）")
                    continue
                wsub = wparts[1].lower() if len(wparts) > 1 else ""
                wname = wparts[2] if len(wparts) > 2 else ""
                if wsub == "use":
                    if not wname:
                        print("用法: /wf use <名>（/wf stop 取消）")
                        continue
                    if ctrl.wf_get(wname) is None:
                        print(f"⚠️ 工作流「{wname}」不存在（/wf list 查看）")
                        continue
                    pending_wf = wname
                    print(f"▶ 已挂载工作流「{wname}」：下一条消息发送即按工作流执行（/wf stop 取消）")
                    continue
                if wsub in ("stop", "unuse"):
                    if pending_wf:
                        print(f"已解除工作流「{pending_wf}」的挂载")
                    else:
                        print("当前没有已挂载的工作流")
                    pending_wf = None
                    continue
                to_send = _plain_wf(ctrl, text)
                if to_send is not None:
                    try:
                        ctrl.send_message(to_send, emit)
                    except Exception as e:
                        print(f"\n⚠️ {e}")
                continue
            if low == "/tree":
                print(ctrl.tree.render_tree(
                    ctrl.tree.current_node.id if ctrl.tree.current_node else None
                ))
                continue
            if text.startswith("/"):
                print(f"未知命令: {text}")
                continue
            if pending_wf:
                wf_name = pending_wf
                pending_wf = None
                composed = ctrl.wf_compose(wf_name, typed=text)
                if composed is None:
                    print(f"⚠️ 工作流「{wf_name}」不存在，已解除挂载")
                    continue
                text = composed
                print(f"▶ 已按工作流「{wf_name}」执行")
            try:
                ctrl.send_message(text, emit)
            except Exception as e:
                print(f"\n⚠️ {e}")
    finally:
        ctrl.save_session()
        ctrl.close()


@app.command()
def info() -> None:
    """显示当前配置信息。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    registered = load_models()
    print("mincli 配置")
    print(f"  API Key: {'已配置 ✓' if api_key else '未配置 ✗'} (DEEPSEEK_API_KEY)")
    print(f"  模型: {MODEL_V4_FLASH} / {MODEL_V4_PRO} / {MODEL_V4_VISION}")
    if registered:
        print(f"  已注册模型: {', '.join(registered.keys())}")
    print(f"  保存路径: {SAVE_BASE_DIR}")
    print(f"  系统提示词: {SYSTEM_PROMPT_SOURCE or '内置兜底'}（{len(DEFAULT_SYSTEM_PROMPT)} 字符）")
    print("  模式: 树状对话 (Textual TUI)")
    print("  多模型: `mincli register <模型名> <URL>` 注册 / `mincli models` 查看")


def main() -> None:
    if "--mcp-server" in sys.argv:
        from mincli.mcp_server import main as mcp_server_main
        mcp_server_main()
        return
    app()


if __name__ == "__main__":
    main()
