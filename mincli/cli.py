"""mincli CLI 入口（Typer）。

默认启动 Textual TUI；`--no-tui` 时退化为极简纯文本对话（无 Rich / 无
prompt_toolkit / 无流式渲染依赖）。
"""

import os
import sys

import typer
from openai import OpenAI

from mincli.tools.files import FilesAPIError

from mincli.config import (
    MODEL_V4_FLASH,
    MODEL_V4_PRO,
    MODEL_V4_VISION,
    COMPACT_DEFAULT_KEEP,
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
                print("命令: /exit 退出 | /clear 清空 | /compact 压缩上下文 | /tree 显示对话树 | /info 节点详情 | /img 添加图片 | /files 管理图片文件")
                continue
            if low in ("/img",) or low.startswith("/img "):
                parts = text.split()
                if len(parts) < 2:
                    print("用法: /img <路径或URL> [...] | /img clear")
                elif parts[1].lower() in ("clear", "c"):
                    n = ctrl.clear_pending_images()
                    print(f"已清除 {n} 张待发送图片")
                else:
                    added, errors = ctrl.add_pending_images(parts[1:])
                    print(f"✅ 已添加 {added} 张图片（发送时自动附带）" if added else "未添加图片")
                    for err in errors:
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
                parts = text.split()
                sub = parts[1].lower() if len(parts) > 1 else ""
                if sub in ("off", "reset", "clear", "undo"):
                    if ctrl.clear_compaction():
                        print("已清除上下文压缩摘要，恢复发送完整原始消息")
                    else:
                        print("当前没有压缩摘要")
                    continue
                keep = COMPACT_DEFAULT_KEEP
                if sub:
                    try:
                        keep = max(0, int(sub))
                    except ValueError:
                        print("用法: /compact [保留轮数] | /compact off")
                        continue
                print("正在压缩上下文…")
                stats = ctrl.compact_history(keep)
                if stats is None:
                    print("无可压缩的对话（对话太短，或压缩失败）")
                else:
                    print(
                        f"✅ 已压缩 {stats['nodes_compressed']} 轮，保留最近 {stats['nodes_kept']} 轮原文；"
                        f"Token {stats['before_tokens']} → {stats['after_tokens']}（节省 {stats['saved_tokens']}）"
                    )
                    print(f"📦 摘要（{stats['summary_chars']} 字）：\n{stats['summary']}")
                continue
            if low == "/tree":
                print(ctrl.tree.render_tree(
                    ctrl.tree.current_node.id if ctrl.tree.current_node else None
                ))
                continue
            if text.startswith("/"):
                print(f"未知命令: {text}")
                continue
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
