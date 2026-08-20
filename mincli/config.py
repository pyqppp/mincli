import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.expanduser("~/.mincli/.env"))

MODEL_V4_FLASH = "deepseek-v4-flash"
MODEL_V4_PRO = "deepseek-v4-pro"
DEFAULT_MODEL = MODEL_V4_FLASH

# 内置模型映射：model_name -> base_url（OpenAI 兼容 API）
MODELS_AVAILABLE = {
    MODEL_V4_FLASH: "https://api.deepseek.com/v1",
    MODEL_V4_PRO: "https://api.deepseek.com/v1",
}

# API Provider 映射：provider_name -> 环境变量名
# 未注册的自定义 provider 回退到 DEEPSEEK_API_KEY
API_PROVIDERS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# 用户注册的模型配置文件（如：{"custom_openai": {"url": ..., "key_var": ..., "model": ...}}）
MODELS_CONFIG_PATH = os.path.expanduser(
    os.getenv("MINCLI_MODELS_PATH", "~/.mincli/models.json")
)

SAVE_BASE_DIR = os.path.expanduser(
    os.getenv("MINCLI_SAVE_PATH", "~/Documents/mincli_Conversations")
)

TITLE_MAX_TOKENS = 30
TITLE_MAX_LENGTH = 30
PREVIEW_USER_MSG_LEN = 100
PREVIEW_ASSISTANT_MSG_LEN = 200
WEBPAGE_MAX_LENGTH = 5000

# 上下文压缩（/compact）
COMPACT_DEFAULT_KEEP = 5          # 默认保留最近 N 轮原始内容
COMPACT_MAX_TOKENS = 8192         # 压缩摘要的最大输出 token 数（生成失败时回退 4096）
COMPACT_SOURCE_MAX_CHARS = 150_000  # 送入压缩模型的原文上限（超长时截头尾保中间）
COMPACT_REASONING_MAX_CHARS = 800   # 每个节点思考过程计入摘要源的长度上限
COMPACT_TOOL_RESULT_MAX_CHARS = 500  # 每个工具结果计入摘要源的长度上限

# DeepSeek 官方定价（元/百万 tokens，2026-08 峰谷分时版；高峰 = 空闲×2）
# 高峰时段：北京时间 9:00-12:00、14:00-18:00，其余为空闲时段
# 结构: 模型名 -> {"hit": (空闲价, 高峰价), "miss": (空闲价, 高峰价), "output": (空闲价, 高峰价)}
DEEPSEEK_PRICING: dict = {
    MODEL_V4_FLASH: {"hit": (0.05, 0.10), "miss": (1.5, 3.0), "output": (4.5, 9.0)},
    MODEL_V4_PRO: {"hit": (0.15, 0.30), "miss": (4.5, 9.0), "output": (13.5, 27.0)},
}

# 账户余额轮询刷新间隔（秒）
BALANCE_REFRESH_SECONDS = 60
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0

MCP_CONFIG_PATH = os.path.expanduser(
    os.getenv("MINCLI_MCP_CONFIG", "~/.mincli/mcp_servers.json")
)

# 系统提示词独立存放在文件中，每次启动自动导入：
# 1. MINCLI_SYSTEM_PROMPT_PATH 环境变量指定的文件（优先级最高）
# 2. ~/.mincli/system_prompt.md（用户自定义，覆盖默认提示词）
# 3. 包内 system_prompt.md（随项目分发，默认提示词）
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
USER_SYSTEM_PROMPT_PATH = os.path.expanduser("~/.mincli/system_prompt.md")

# 所有提示词文件均不可用时的内置兜底（正常情况下不会用到）
_FALLBACK_SYSTEM_PROMPT = "你是一个有用的人工智能助手。"


def _load_default_system_prompt() -> tuple[str, Optional[str]]:
    """读取系统提示词，返回 (提示词内容, 实际使用的文件路径或 None)。"""
    candidates = []
    env_path = os.getenv("MINCLI_SYSTEM_PROMPT_PATH")
    if env_path:
        candidates.append(os.path.expanduser(env_path))
    candidates.append(USER_SYSTEM_PROMPT_PATH)
    candidates.append(SYSTEM_PROMPT_PATH)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            continue
        if content:
            return content, path
    return _FALLBACK_SYSTEM_PROMPT, None


DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT_SOURCE = _load_default_system_prompt()


def load_mcp_servers() -> dict:
    """加载第三方 MCP server 配置（Claude Desktop 兼容格式）。"""
    if not os.path.exists(MCP_CONFIG_PATH):
        return {}
    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
            return data["mcpServers"]
        return {}
    except Exception:
        return {}


def get_mcp_config_path() -> str:
    return MCP_CONFIG_PATH


def save_mcp_servers(servers: dict) -> str:
    """将第三方 MCP server 配置写回文件，返回文件路径。"""
    os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
    with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"mcpServers": servers}, f, ensure_ascii=False, indent=2)
    return MCP_CONFIG_PATH


# ---------------- 模型注册配置管理 ----------------

def load_models() -> dict:
    """加载用户注册的模型配置：{"模型名": {"url": ..., "key_var": ...}}。"""
    if not os.path.exists(MODELS_CONFIG_PATH):
        return {}
    try:
        with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_models(models: dict) -> bool:
    """保存模型注册配置到 ~/.mincli/models.json，成功返回 True。"""
    try:
        os.makedirs(os.path.dirname(MODELS_CONFIG_PATH), exist_ok=True)
        with open(MODELS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(models, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def register_model(
    provider: str,
    model_name: str,
    base_url: str,
    api_key_var: Optional[str] = None,
) -> bool:
    """注册一个新的模型到配置（provider + 模型名 → URL / API Key 变量）。

    - 内置 provider（deepseek/openai）按模型名注册；
    - 自定义 provider 以 `custom_<provider>` 为键注册（可同时保存模型名）。
    返回是否注册成功。
    """
    models = load_models()
    key_var = api_key_var or API_PROVIDERS.get(provider, "DEEPSEEK_API_KEY")
    models[model_name] = {"url": base_url, "key_var": key_var}
    return save_models(models)


def get_model_base_url(provider: str, model: str) -> str:
    """解析 provider + 模型名 → API base_url。

    优先级：注册配置 > 内置映射 > 默认 DeepSeek。
    """
    registered = load_models()
    if model in registered:
        return registered[model]["url"]
    if model in MODELS_AVAILABLE:
        return MODELS_AVAILABLE[model]
    # 兜底：DeepSeek 默认端点
    return "https://api.deepseek.com/v1"


def get_model_key_var(provider: str, model: str) -> str:
    """解析 provider + 模型名 → API Key 的环境变量名。"""
    registered = load_models()
    if model in registered:
        return registered[model].get("key_var") or "DEEPSEEK_API_KEY"
    return API_PROVIDERS.get(provider, "DEEPSEEK_API_KEY")
