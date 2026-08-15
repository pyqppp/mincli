"""mincli CLI 入口（Typer）。

默认启动 Textual TUI；`--no-tui` 时退化为极简纯文本对话（无 Rich / 无
prompt_toolkit / 无流式渲染依赖）。
"""

import os
import sys

import typer
from openai import OpenAI

from mincli.config import (
    MODEL_V4_FLASH,
    MODEL_V4_PRO,
    SAVE_BASE_DIR,
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SOURCE,
)

app = typer.Typer(help="mincli - 树状对话 AI 助手")


def get_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 未设置 DEEPSEEK_API_KEY（请在 .env 或环境变量中配置）")
        raise typer.Exit(1)
    return api_key


def build_controller(
    model: str,
    temperature: float,
    thinking: bool,
    effort: str,
) -> "ChatController":
    """按 CLI 参数构造 ChatController（惰性导入，避免拖慢非聊天命令）。"""
    from mincli.controller import ChatController

    return ChatController(
        client=OpenAI(api_key=get_api_key(), base_url="https://api.deepseek.com"),
        default_system=DEFAULT_SYSTEM_PROMPT,
        default_temperature=temperature,
        default_model=MODEL_V4_PRO if model.lower() == "pro" else MODEL_V4_FLASH,
        thinking_enabled=thinking,
        reasoning_effort=effort,
    )


@app.command()
def chat(
    model: str = typer.Option("flash", "--model", "-m", help="模型: flash 或 pro"),
    temperature: float = typer.Option(1.0, "--temp", "-temp-opt", help="温度参数"),
    thinking: bool = typer.Option(False, "--thinking", "-r", help="开启思考模式（默认 high）"),
    effort: str = typer.Option("high", "--effort", help="推理强度: low, high 或 max"),
    no_tui: bool = typer.Option(False, "--no-tui", help="不使用 TUI，用极简纯文本对话"),
) -> None:
    """启动树状对话（默认 Textual TUI）。"""
    if effort not in ("low", "high", "max"):
        print(f"无效推理强度: {effort}，可选 low / high / max")
        raise SystemExit(2)

    if no_tui:
        _chat_plain(model, temperature, thinking, effort)
        return

    from mincli.tui.app import ChatApp

    ChatApp(controller=build_controller(model, temperature, thinking, effort)).run()


def _chat_plain(model: str, temperature: float, thinking: bool, effort: str) -> None:
    """极简纯文本对话：input() 逐行输入，无 Rich / prompt_toolkit 依赖。"""
    from mincli.controller import ControllerEvent

    ctrl = build_controller(model, temperature, thinking, effort)

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
                print("命令: /exit 退出 | /clear 清空 | /tree 显示对话树 | /info 节点详情")
                continue
            if low == "/clear":
                ctrl.reset()
                print("已清空当前会话")
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
    print("mincli 配置")
    print(f"  API Key: {'已配置 ✓' if api_key else '未配置 ✗'}")
    print(f"  模型: {MODEL_V4_FLASH} / {MODEL_V4_PRO}")
    print(f"  保存路径: {SAVE_BASE_DIR}")
    print(f"  系统提示词: {SYSTEM_PROMPT_SOURCE or '内置兜底'}（{len(DEFAULT_SYSTEM_PROMPT)} 字符）")
    print("  模式: 树状对话 (Textual TUI)")


def main() -> None:
    if "--mcp-server" in sys.argv:
        from mincli.mcp_server import main as mcp_server_main
        mcp_server_main()
        return
    app()


if __name__ == "__main__":
    main()
