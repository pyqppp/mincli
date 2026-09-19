"""mincli CLI 入口（Typer）。

`mincli chat` 启动多对话树对话界面（Textual TUI）。
"""

import os
import sys

import typer

from mincli.config import (
    MODEL_FLASH,
    MODEL_PRO,
    PRICING_PATH,
    SAVE_BASE_DIR,
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SOURCE,
    MODELS_AVAILABLE,
    API_PROVIDERS,
    load_models,
    normalize_model_name,
    register_model,
    get_model_base_url,
    get_model_key_var,
)
from mincli.pricing import image_tokens_per_image, load_pricing

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
    """把简写/旧模型名映射为现役完整模型名。

    flash/f → deepseek-flash；pro/p → deepseek-v4-pro；
    vision 等识图旧名 → deepseek-flash（图片能力已并入 Flash）。
    """
    return normalize_model_name(model)


def build_controller(
    provider: str,
    model: str,
    temperature: float,
    thinking: bool,
    effort: str,
    auto_start_mcp: bool = True,
) -> "ChatController":
    """按 CLI 参数构造 ChatController（支持多 Provider/多模型，惰性导入）。

    auto_start_mcp=False 时不在构造期连接 MCP：TUI 用它把「首屏」和「MCP
    连接」解耦，界面先出来、连接在后台进行（见 ChatApp._start_mcp）。
    """
    from mincli.controller import ChatController
    from openai import OpenAI

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
        auto_start_mcp=auto_start_mcp,
    )


@app.command()
def chat(
    provider: str = typer.Option("deepseek", "-p", "--provider", help="API Provider（deepseek/openai 或已注册的自定义 provider）"),
    model: str = typer.Option("flash", "-m", "--model", help="模型: flash / pro / 完整模型名（如 gpt-4o）"),
    temperature: float = typer.Option(1.0, "--temp", "-temp-opt", help="温度参数"),
    thinking: bool = typer.Option(False, "--thinking", "-r", help="开启思考模式（默认 high）"),
    effort: str = typer.Option("high", "--effort", help="推理强度: low, high 或 max"),
) -> None:
    """启动多对话树对话界面（Textual TUI），支持多 Provider/多模型。"""
    if effort not in ("low", "high", "max"):
        print(f"无效推理强度: {effort}，可选 low / high / max")
        raise SystemExit(2)

    from mincli.tui.app import ChatApp

    # MCP 交给 ChatApp 在首屏之后后台连接，避免主界面等 4 秒左右才出现
    ctrl = build_controller(provider, model, temperature, thinking, effort, auto_start_mcp=False)
    ChatApp(controller=ctrl).run()


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


@app.command()
def info() -> None:
    """显示当前配置信息。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    registered = load_models()
    print("mincli 配置")
    print(f"  API Key: {'已配置 ✓' if api_key else '未配置 ✗'} (DEEPSEEK_API_KEY)")
    print(f"  模型: {MODEL_FLASH} / {MODEL_PRO}（图片理解仅 {MODEL_FLASH} 支持）")
    if registered:
        print(f"  已注册模型: {', '.join(registered.keys())}")
    print(f"  保存路径: {SAVE_BASE_DIR}")
    pricing = load_pricing()
    print(
        f"  定价配置: {pricing['path'] or '内置默认'}"
        f"（可编辑 {PRICING_PATH} 覆盖价格/峰谷/图片 token）"
    )
    print(f"  图片 token 估算: {image_tokens_per_image(pricing)} / 张")
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
