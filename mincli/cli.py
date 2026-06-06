import os

import typer
from openai import OpenAI
from rich.table import Table

from mincli.config import MODEL_V4_FLASH, MODEL_V4_PRO, SAVE_BASE_DIR
from mincli.render import console
from mincli.session import InteractiveSession

app = typer.Typer(help="mincli - 树状对话 AI 助手")


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[red]错误: 未设置 DEEPSEEK_API_KEY[/red]")
        raise typer.Exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


@app.command()
def chat(
    model: str = typer.Option("flash", "--model", "-m", help="模型: flash 或 pro"),
    temperature: float = typer.Option(1.0, "--temp", "-temp-opt", help="温度参数"),
    thinking: bool = typer.Option(False, "--thinking", "-r", help="开启思考模式（默认 high）"),
    effort: str = typer.Option("high", "--effort", help="推理强度: high 或 max"),
) -> None:
    """启动树状对话模式。"""
    selected_model = MODEL_V4_PRO if model.lower() == "pro" else MODEL_V4_FLASH

    if thinking:
        console.print(f"[cyan]🧠 开启思考模式 (effort: {effort})[/cyan]")
    else:
        console.print(f"[dim]🧠 思考模式关闭[/dim]")

    client = get_client()
    session = InteractiveSession(
        client=client,
        default_system="你是一个有用的人工智能助手",
        default_temperature=temperature,
        default_model=selected_model,
        thinking_enabled=thinking,
        reasoning_effort=effort,
    )
    session.run()


@app.command()
def info() -> None:
    """显示当前配置信息。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    table = Table(title="mincli 配置")
    table.add_column("项目", style="cyan")
    table.add_column("状态", style="green")
    table.add_row("API Key", "已配置 ✓" if api_key else "未配置 ✗")
    table.add_row("模型", f"{MODEL_V4_FLASH}\n{MODEL_V4_PRO}")
    table.add_row("保存路径", SAVE_BASE_DIR)
    table.add_row("模式", "树状对话")
    table.add_row("输出方式", "流式实时刷新 + Markdown 渲染")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
