import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.expanduser("~/.mincli/.env"))

MODEL_V4_FLASH = "deepseek-v4-flash"
MODEL_V4_PRO = "deepseek-v4-pro"
DEFAULT_MODEL = MODEL_V4_FLASH

SAVE_BASE_DIR = os.path.expanduser(
    os.getenv("MINCLI_SAVE_PATH", "~/Documents/mincli_Conversations")
)

TITLE_MAX_TOKENS = 30
TITLE_MAX_LENGTH = 30
PREVIEW_USER_MSG_LEN = 100
PREVIEW_ASSISTANT_MSG_LEN = 200
WEBPAGE_MAX_LENGTH = 5000
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
