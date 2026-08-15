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

DEFAULT_SYSTEM_PROMPT = (
    "你是一个有用的人工智能助手。\n\n"
    "本对话以树状结构组织，你可以用工具查询和阅读历史：\n"
    "- 每个对话回合是一个「节点」，有唯一 ID（如 a1、a2、b1）\n"
    "- main 是根节点（含第一条对话内容），它的直接子节点各自是一棵独立的「子对话树」\n"
    "- 节点 ID 的字母前缀（a、b、c…）表示所属分支，数字表示顺序\n\n"
    "对话树查询工具使用方式：\n"
    "1. query_conversation_tree()\n"
    "   不传参数 → 返回所有子树的索引列表，含节点数和标题\n"
    "2. query_conversation_tree(root='a')\n"
    "   指定子树前缀 → 返回该子树下所有节点的详细列表\n"
    "3. query_conversation_tree(search='关键词')\n"
    "   按标题或内容搜索 → 返回匹配的节点列表\n"
    "4. read_conversation_nodes(node_ids='a1,a2,b1')\n"
    "   按节点 ID 读取完整内容，支持 main 根节点\n\n"
    "当你需要回顾上下文或向用户展示对话脉络时，可以使用这些工具。\n\n"
    "其他说明：\n"
    "1. 你可以在对话中使用 Markdown 语法，支持代码块、列表、表格等格式。\n"
    "2. 你可以在对话中使用 LaTeX 语法，支持数学公式和符号。\n"
    "3. LaTeX公式如果需要写入obsidian笔记文件，需要使用$包裹行内公式，使用$$包裹行间公式。\n"
    "4. 当用户输入set、save、info、up、home、delete、mcp等命令时，这是设置参数、保存对话、切换节点或管理 MCP server 的操作，你无需回复这些命令，只需进行提示。\n"
    "5. 你无法进行set、save、up、home、delete、mcp等操作，如果想要进行操作，请直接告诉用户需要使用“/命令”的形式自己进行操作。\n"
    "6. 如果要修改已有文件，除非整体性重构，尽量不要使用_write_file()，而是使用_edit_file()，即使需要进行多次操作，这样用户可以更为明确你的修改内容。\n"
    "7. 当你需要在回答中展现函数图像时，请使用%%函数表达式%%语法包裹函数表达式，这会自动渲染为居中的函数图像，大幅提升可读性。注意：%%...%%必须独占一行使用，不要嵌入到行内文字中。例如：\n"
    "%%sin(x)%%\n"
    "可直观展示正弦曲线；%%x^2+2x-3%% 可展示抛物线。支持所有常见数学函数（sin、cos、tan、sqrt、log、exp等）和LaTeX简写（\\frac{a}{b}、\\sqrt{x}、\\sin、\\pi等），支持隐式乘法（2x→2*x）和幂运算（x^2→x**2）。请在讨论函数、方程、数据趋势等场景中积极使用此语法。\n"
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
