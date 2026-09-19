import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.expanduser("~/.mincli/.env"))

# ---------------- 模型（2026-09 官方文档） ----------------
# deepseek-flash   = DeepSeek-V4.1-Flash（原生多模态，支持图片理解）
# deepseek-v4-pro  = DeepSeek-V4-Pro-0813（不支持图片）
MODEL_FLASH = "deepseek-flash"
MODEL_PRO = "deepseek-v4-pro"
DEFAULT_MODEL = MODEL_FLASH

# 支持图片理解的模型（仅 Flash；Pro 不支持，收到图片时应提示用户切换）
VISION_MODELS = (MODEL_FLASH,)

# 模型简写 / 已下线旧名 → 现役模型名（旧名自动改写为现役名，见 normalize_model_name）
MODEL_ALIASES = {
    "flash": MODEL_FLASH,
    "f": MODEL_FLASH,
    "v4-flash": MODEL_FLASH,
    "deepseek-v4-flash": MODEL_FLASH,
    "pro": MODEL_PRO,
    "p": MODEL_PRO,
    "v4-pro": MODEL_PRO,
    # 独立识图模型已下线，图片能力并入 Flash
    "vision": MODEL_FLASH,
    "v-flash-vision": MODEL_FLASH,
    "v4-vision": MODEL_FLASH,
    "deepseek-v4-flash-vision-exp": MODEL_FLASH,
    # 更早的旧名（2026-07-24 已停止使用）
    "deepseek-chat": MODEL_FLASH,
    "deepseek-reasoner": MODEL_FLASH,
}


def normalize_model_name(name: str) -> str:
    """把简写/已下线的旧模型名归一为现役模型名；未知名称原样返回。"""
    key = (name or "").strip().lower()
    return MODEL_ALIASES.get(key, name)


# 内置模型映射：model_name -> base_url（OpenAI 兼容 API；官方 base_url 不带 /v1）
MODELS_AVAILABLE = {
    MODEL_FLASH: "https://api.deepseek.com",
    MODEL_PRO: "https://api.deepseek.com",
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

# 网页正文长度上限（fetch_webpage 一次返回给模型的字符数）
# 默认 5000，可用环境变量 MINCLI_WEBPAGE_MAX_LENGTH 调整；为避免超长网页挤爆上下文，
# 无论怎么配置都不会超过硬上限 WEBPAGE_MAX_LENGTH_LIMIT（默认值的 4 倍）。
WEBPAGE_MAX_LENGTH_DEFAULT = 5000
WEBPAGE_MAX_LENGTH_LIMIT = WEBPAGE_MAX_LENGTH_DEFAULT * 4  # 20000


def webpage_max_length() -> int:
    """当前生效的网页正文长度上限。

    读取 MINCLI_WEBPAGE_MAX_LENGTH（.env 同样生效）：空值/非法值静默回退默认值；
    结果夹在 [1, WEBPAGE_MAX_LENGTH_LIMIT] 之间，因此配置再大也突破不了硬上限。
    """
    raw = (os.getenv("MINCLI_WEBPAGE_MAX_LENGTH") or "").strip()
    if not raw:
        return WEBPAGE_MAX_LENGTH_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return WEBPAGE_MAX_LENGTH_DEFAULT
    return max(1, min(value, WEBPAGE_MAX_LENGTH_LIMIT))

# 上下文压缩（/compact）：压缩当前分支全部对话并新建摘要节点
COMPACT_MAX_TOKENS = 8192         # 压缩摘要的最大输出 token 数（生成失败时回退 4096）
COMPACT_SOURCE_MAX_CHARS = 150_000  # 送入压缩模型的原文上限（超长时截头尾保中间）
COMPACT_REASONING_MAX_CHARS = 800   # 每个节点思考过程计入摘要源的长度上限
COMPACT_TOOL_RESULT_MAX_CHARS = 500  # 每个工具结果计入摘要源的长度上限

# DeepSeek 官方定价默认值（元/百万 tokens；2026-09-10 V4.1-Flash 上线后降价）
# 结构: 模型名 -> {"hit": (空闲价, 高峰价), "miss": (...), "output": (...)}
# 可被 ~/.mincli/pricing.json 覆盖（见 mincli/pricing.py）；高峰时段规则同样可配。
DEEPSEEK_PRICING: dict = {
    MODEL_FLASH: {"hit": (0.02, 0.04), "miss": (1.0, 2.0), "output": (4.0, 8.0)},
    MODEL_PRO: {"hit": (0.15, 0.30), "miss": (4.5, 9.0), "output": (13.5, 27.0)},
}

# 高峰时段默认：北京时间（UTC+8）周一至周五 9:00-12:00、14:00-18:00，其余为空闲时段。
# 官方 2026-09 起高峰仅限工作日（周末全天空闲）。
DEFAULT_PEAK_CONFIG: dict = {
    "days": [1, 2, 3, 4, 5],          # ISO 星期：周一=1 … 周日=7
    "ranges": [[9, 12], [14, 18]],    # [起始小时, 结束小时)，可多段
    "timezone_offset_hours": 8,       # 高峰判定所用时区（北京时间为 8）
}

# 图片 token 估算兜底：只用于「尺寸未知」（如外链 URL）的图片，默认取官方单图上限；
# 尺寸已知时按官方预处理公式精确换算（见 tools/images.estimate_image_tokens）
VISION_IMAGE_TOKENS_DEFAULT = 1024

# 定价配置文件（价格、峰谷时段、图片 token 估算兜底值均可在此覆盖）
PRICING_PATH = os.path.expanduser(
    os.getenv("MINCLI_PRICING_PATH", "~/.mincli/pricing.json")
)

# 账户余额轮询刷新间隔（秒）
BALANCE_REFRESH_SECONDS = 60
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0

# 命令执行工具（execute_command）
EXEC_DEFAULT_TIMEOUT = 30          # 未传 timeout 时的默认截止时间（秒）
# timeout 上限（秒）。早期固定 120s，渲染/构建/安装等长任务会被硬截断，模型只能
# 反复「启动休眠」轮询；现在默认放宽到 30 分钟，并可用 MINCLI_EXEC_MAX_TIMEOUT
# 覆盖。真正卡死的命令由用户主动打断终止（见 controller.interrupt）。
EXEC_MAX_TIMEOUT_DEFAULT = 1800
EXEC_DEFAULT_MAX_OUTPUT = 8000     # 输出截断上限（字符），超出时保留首尾并落盘完整输出
EXEC_MAX_OUTPUT = 50_000           # max_output 参数允许的最大值
EXEC_ALLOWED_SHELLS = ("sh", "bash", "zsh")


def exec_max_timeout() -> int:
    """当前生效的 execute_command 超时上限（秒）。

    读取 MINCLI_EXEC_MAX_TIMEOUT（.env 同样生效）：空值/非法值回退默认值；
    结果至少为 1。上限同时决定 MCP 客户端调用超时（见 mcp_client.CALL_TIMEOUT）。
    """
    raw = (os.getenv("MINCLI_EXEC_MAX_TIMEOUT") or "").strip()
    if not raw:
        return EXEC_MAX_TIMEOUT_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return EXEC_MAX_TIMEOUT_DEFAULT
    return max(1, value)


# 模块级常量：多数调用点直接引用（导入时求值一次，启动后不再变化）
EXEC_MAX_TIMEOUT = exec_max_timeout()

# ---------------- 多模态（图片理解，deepseek-flash 原生支持） ----------------
# 官方限制（2026-09）：格式按文件内容识别（不看扩展名/声明 MIME）；图片仅限 user 消息；
# 内联/URL 单图 32MiB、Files API file_id 单图 64MiB；请求体 48MiB；URL ≤8192 字符；
# 单请求 ≤600 图；不含 file_id 的图片总量 ≤64MiB、含 file_id 最高 200MiB；
# 单边 ≤8192px，单请求 ≥15 张时降为 ≤4096px；每图 token 上限 1024。
VISION_SUPPORTED_FORMATS = ("jpeg", "png", "gif", "webp")
VISION_INLINE_IMAGE_MAX_BYTES = 32 * 1024 * 1024   # base64 / 外部 URL 单图上限
VISION_FILE_ID_IMAGE_MAX_BYTES = 64 * 1024 * 1024  # Files API file_id 单图上限
VISION_REQUEST_MAX_BYTES = 48 * 1024 * 1024        # 请求体上限（内联 base64 回退路径预检）
VISION_REQUEST_IMAGES_MAX_COUNT = 600              # 单请求图片数上限
VISION_REQUEST_INLINE_TOTAL_MAX_BYTES = 64 * 1024 * 1024  # 不含 file_id 的图片总量上限
VISION_REQUEST_TOTAL_MAX_BYTES = 200 * 1024 * 1024        # 含 file_id 的图片总量上限
VISION_MAX_SIDE = 8192             # 单图单边最大像素
VISION_MAX_SIDE_MANY = 4096        # 单请求 ≥ VISION_MANY_IMAGES_THRESHOLD 张时的单边上限
VISION_MANY_IMAGES_THRESHOLD = 15
VISION_URL_MAX_CHARS = 8192
VISION_DEFAULT_DETAIL = "auto"   # low(512²缩放,省token) / high=original / auto≈original
VISION_LOW_SCALE_SIDE = 512        # detail=low 时先把图片缩到单边不超过该值

# 图片 token 换算（官方实现参数，取自 deepseek-ai/DeepSeek-V4.1-Flash 的
# vision_config，与 vLLM 参考实现 vllm/models/deepseek_v41/common/mm_preprocess.py
# 的 num_image_tokens 一致）：
#   best_w = ceil(w / patch) * patch（宽高各自对齐到 patch 的整数倍）
#   n_h = ceil(best_h / patch / downsample)，n_w 同理
#   单图 token = n_h * (n_w + 1) + 2，上限 VISION_MAX_IMAGE_TOKENS
# 小于 min_pixels 的图片先按长宽比放大到 min_pixels（约 544×544），因此小图不会
# 低于约 184 token；上限 1024 对应约 1300×1300 的缩放结果。
VISION_PATCH_SIZE = 14
VISION_DOWNSAMPLE_RATIO = 3
VISION_MAX_IMAGE_TOKENS = 1024
VISION_MIN_PIXELS = 295936         # 544 * 544，小图放大阈值

# detail=low 时改用 base64 内联（file 块不支持 detail，只有内联/外链才真正生效）。
# 内联走 base64（体积约 4/3），这里按原始字节给一个请求级预算；取 24MiB 是为了给
# 「48MiB 请求体」留出文本历史的余量（24MiB → base64 约 32MiB）。
VISION_LOW_INLINE_BUDGET_BYTES = 24 * 1024 * 1024

# Files API 存储配额与 /files 列表分页（官方文档 2026-09）
FILES_MAX_COUNT = 10_000                   # 单用户最大存储文件数
FILES_MAX_BYTES = 25 * 1024 * 1024 * 1024  # 单用户最大存储空间 25 GiB
FILES_LIST_PAGE = 20                       # /files list 默认展示条数
FILES_LIST_PAGE_MAX = 1000                 # 官方单页上限

MCP_CONFIG_PATH = os.path.expanduser(
    os.getenv("MINCLI_MCP_CONFIG", "~/.mincli/mcp_servers.json")
)

# 工作流持久化文件（/wf 系列命令；与会话分开长期保存）
WORKFLOWS_PATH = os.path.expanduser(
    os.getenv("MINCLI_WORKFLOWS_PATH", "~/.mincli/workflows.json")
)

# ---------------- 工作流（/wf）提炼 ----------------
WF_EXTRACT_MAX_TOKENS = 4000      # 提炼输出上限（失败回退 2000）
WF_SOURCE_MAX_CHARS = 60_000      # 送入提炼模型的原文上限（超长截头尾保中间）
WF_REASONING_MAX_CHARS = 400      # 每节点思考计入提炼源的长度上限
WF_TOOL_ARGS_MAX_CHARS = 800      # 每个工具调用参数计入提炼源的长度上限
WF_TOOL_RESULT_MAX_CHARS = 400    # 每个工具结果计入提炼源的长度上限
WF_ANSWER_MAX_CHARS = 1200        # 每节点最终回答计入提炼源的长度上限

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
