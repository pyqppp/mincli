import json
import os

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
