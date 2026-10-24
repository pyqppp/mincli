"""mincli 命令行入口：只保留 ``chat`` 与 ``info``。

所有设置都在 ``~/.mincli/config.toml``，命令行不再接受模型/温度等启动参数。
"""

import typer

from mincli.config import (
    API_KEY_VAR,
    PRICING_PATH,
    SAVE_BASE_DIR,
    api_key,
    config_path,
    ensure_config,
    system_prompt,
)
from mincli.pricing import load_pricing

app = typer.Typer(
    help="mincli - 树状对话 AI 助手",
    no_args_is_help=True,
    add_completion=False,
)


def build_controller(cfg=None):
    """按配置构造 ChatController（惰性导入，保证 info 不需要 OpenAI SDK）。"""
    from openai import OpenAI

    from mincli.config import DEEPSEEK_BASE_URL
    from mincli.controller import ChatController

    cfg = cfg or ensure_config()
    key = api_key()
    if not key:
        print(
            f"错误：未设置 {API_KEY_VAR}，"
            "请在环境变量或 ~/.mincli/.env 中配置"
        )
        raise typer.Exit(1)
    return ChatController(
        client=OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL),
        config=cfg,
        default_system=system_prompt(cfg),
    )


@app.command()
def chat() -> None:
    """启动多对话树对话界面（Textual TUI）。"""
    cfg = ensure_config()
    from mincli.tui.app import ChatApp

    ChatApp(controller=build_controller(cfg)).run()


@app.command()
def info() -> None:
    """显示当前生效的配置（只读；不打印提示词内容）。"""
    cfg = ensure_config()
    print("mincli 配置")
    print(f"  配置文件: {config_path()}")
    print(f"  API Key: {'已配置' if api_key() else '未配置'}（{API_KEY_VAR}）")
    print(f"  模型: {cfg.model}")
    print(f"  温度: {cfg.temperature:.1f}")
    print(f"  思考: {'开' if cfg.thinking else '关'}（强度 {cfg.effort}）")
    print(f"  文件权限: {cfg.files_label}（mode = {cfg.files_mode}）")
    print(f"  联网: {cfg.web_label}")
    print(f"  对话树检索: {'开' if cfg.recall_enabled else '关'}")
    print(f"  附件内联上限: {cfg.inline_max_chars} 字符")
    print(f"  附件目录: {cfg.attachment_dir}")
    print(
        f"  提示词文件: {cfg.prompt_file}"
        f"（{len(system_prompt(cfg))} 字符，客户端不显示内容）"
    )
    print(f"  保存路径: {SAVE_BASE_DIR}")
    pricing = load_pricing()
    print(f"  定价配置: {pricing['path'] or '内置默认'}（可编辑 {PRICING_PATH} 覆盖价格/峰谷）")
    print("  模式: 树状对话 (Textual TUI)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
