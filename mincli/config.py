"""mincli 配置：单一配置文件 ``~/.mincli/config.toml``。

设计原则：
- 所有可调项都在一个 TOML 文件里，界面开关变化立即写回（注释按模板重新生成）。
- 非法值静默回退默认值，不阻塞启动。
- 系统提示词单独放在 ``~/.mincli/prompt.md``，客户端不展示、不可编辑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 兜底
    import tomli as tomllib  # type: ignore[no-redef]

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

# DeepSeek 官方 OpenAI 兼容端点（不含 /v1，由 SDK 拼接）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

API_KEY_VAR = "DEEPSEEK_API_KEY"


def normalize_model_name(name: str) -> str:
    """把简写/已下线的旧模型名归一为现役模型名；未知名称原样返回。"""
    key = (name or "").strip().lower()
    return MODEL_ALIASES.get(key, name)


def api_key() -> str:
    """当前生效的 DeepSeek API Key（未配置时为空字符串）。"""
    return (os.getenv(API_KEY_VAR) or "").strip()


# ---------------- 能力开关 ----------------
# 文件权限四态，按限制从松到紧：direct > confirm > read > off
FILE_MODES = ("off", "read", "confirm", "direct")
FILE_MODE_LABELS = {
    "off": "关闭",
    "read": "只读",
    "confirm": "写需确认",
    "direct": "写直通",
}
DEFAULT_FILE_MODE = "off"

EFFORTS = ("low", "high", "max")
DEFAULT_EFFORT = "high"

# 附件内联进消息的字符上限：超过则截断，并把完整正文留给 read_attachment
INLINE_MAX_CHARS_DEFAULT = 3000
INLINE_MAX_CHARS_MIN = 200
INLINE_MAX_CHARS_MAX = 200_000

TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0


def file_mode_rank(mode: str) -> int:
    """文件模式的可比较等级：off=0 / read=1 / confirm=2 / direct=3。"""
    return FILE_MODES.index(mode) if mode in FILE_MODES else 0


def file_mode_label(mode: str) -> str:
    return FILE_MODE_LABELS.get(mode, FILE_MODE_LABELS[DEFAULT_FILE_MODE])


# ---------------- 路径 ----------------
MINCLI_DIR = os.path.expanduser("~/.mincli")


def config_path() -> str:
    """配置文件路径；可用 MINCLI_CONFIG_PATH 覆盖（测试用）。"""
    return os.path.expanduser(
        os.getenv("MINCLI_CONFIG_PATH") or os.path.join(MINCLI_DIR, "config.toml")
    )


def default_prompt_path() -> str:
    """系统提示词文件路径；可用 MINCLI_PROMPT_PATH 覆盖（调试/测试用）。"""
    return os.path.expanduser(
        os.getenv("MINCLI_PROMPT_PATH") or os.path.join(MINCLI_DIR, "prompt.md")
    )


def default_attachment_dir() -> str:
    """附件正文落盘目录；可用 MINCLI_ATTACHMENTS_PATH 覆盖。"""
    return os.path.expanduser(
        os.getenv("MINCLI_ATTACHMENTS_PATH") or os.path.join(MINCLI_DIR, "attachments")
    )


SAVE_BASE_DIR = os.path.expanduser(
    os.getenv("MINCLI_SAVE_PATH", "~/Documents/mincli_Conversations")
)

TITLE_MAX_TOKENS = 30
TITLE_MAX_LENGTH = 30
PREVIEW_USER_MSG_LEN = 100
PREVIEW_ASSISTANT_MSG_LEN = 200

# 上下文压缩（/compact）：压缩当前分支全部对话并新建摘要节点
COMPACT_MAX_TOKENS = 8192           # 压缩摘要的最大输出 token 数（生成失败时回退 4096）
COMPACT_SOURCE_MAX_CHARS = 150_000  # 送入压缩模型的原文上限（超长时截头尾保中间）
COMPACT_REASONING_MAX_CHARS = 800   # 每个节点思考过程计入摘要源的长度上限
COMPACT_TOOL_RESULT_MAX_CHARS = 500  # 每个工具结果计入摘要源的长度上限

# ---------------- 系统提示词 ----------------

DEFAULT_PROMPT = """\
你是 mincli 的 AI 助手，运行在用户的终端里，通过对话帮用户阅读资料、写作与查证。

## 工作方式
- 语言跟随用户；代码、命令、路径、标识符保持原样，不要翻译。
- 用 Markdown 组织回答；数学公式用 LaTeX（行内 $...$，行间 $$...$$）。
- 若用户要把内容写入 Obsidian 笔记，行内公式用 $...$，行间公式用 $$...$$。
- 能用工具确认的事先去确认，不要凭猜测描述文件内容。
- 改文件前先读原文件；小改动用 edit_file，整体重写才用 write_file；改完说明改了什么。
- 工具失败或结果为空时如实说明，不要编造成功或结果。
- 删除、覆盖这类不可逆操作，先说清影响再执行。

## 资料与附件
- 用户消息里的 [PDF ...]、[DOCX ...]、[IMAGE ...]、[WEB ...] 是附件引用。
- 短附件正文直接附在消息末尾的 --- 之后；长附件只附前一段并标注「已截断」。
- 标注已截断时，用 read_attachment 按行继续读取（可多次读同一附件）。
- 图片以多模态块单独发送，正文里只有引用块。
- 没有 read_file 这类工具时，说明当前文件权限是关闭的，请用户在底部状态栏打开。

## 能力
- 可用工具以本轮实际提供的为准；缺少完成当前任务的工具时，直接告诉用户当前能力状态，不要假装已执行。
- 历史对话以树状保存，用户可以在任意节点继续提问；只有提供了树查询工具时你才看得到历史节点。

## 客户端
- 以 / 开头的输入由客户端处理，你不会收到；用户想用时直接给出正确的 /命令。
"""


def system_prompt(cfg: "Config" | None = None) -> str:
    """读取生效的系统提示词：提示词文件为空/缺失时回退内置默认。"""
    path = (cfg.prompt_file if cfg else default_prompt_path())
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return DEFAULT_PROMPT.strip()
    return content or DEFAULT_PROMPT.strip()


# ---------------- 配置对象 ----------------

_COMMENT_COL = 36


def _entry(text: str, comment: str) -> str:
    """一行「键 = 值」加对齐的注释；值过长时只补一个空格。"""
    return (text.ljust(_COMMENT_COL) if len(text) < _COMMENT_COL else text + " ") + (
        f"# {comment}"
    )


@dataclass
class Config:
    """一份生效配置。字段名与 TOML 结构一一对应。"""

    model: str = DEFAULT_MODEL
    temperature: float = 1.0
    thinking: bool = False
    effort: str = DEFAULT_EFFORT
    files_mode: str = DEFAULT_FILE_MODE
    inline_max_chars: int = INLINE_MAX_CHARS_DEFAULT
    attachment_dir: str = ""
    web_enabled: bool = True
    recall_enabled: bool = False
    prompt_file: str = ""

    def __post_init__(self) -> None:
        if not self.attachment_dir:
            self.attachment_dir = default_attachment_dir()
        if not self.prompt_file:
            self.prompt_file = default_prompt_path()

    # ---- 派生属性 ----
    @property
    def files_read_enabled(self) -> bool:
        return file_mode_rank(self.files_mode) >= 1

    @property
    def files_write_enabled(self) -> bool:
        return file_mode_rank(self.files_mode) >= 2

    @property
    def files_write_needs_confirm(self) -> bool:
        return self.files_mode == "confirm"

    @property
    def files_label(self) -> str:
        return file_mode_label(self.files_mode)

    @property
    def web_label(self) -> str:
        return "开" if self.web_enabled else "关"

    def to_toml(self, path: Optional[str] = None) -> str:
        lines = [
            "# mincli 配置文件",
            f"# 位置：{path or config_path()}",
            "# 修改后在下次启动生效；界面里的开关变化会立即写回本文件（注释按模板重新生成）。",
            "",
            _entry(
                f'model = "{self.model}"',
                "deepseek-flash（支持图片）| deepseek-v4-pro",
            ),
            _entry(f"temperature = {self.temperature:.1f}", "0.0 - 2.0"),
            _entry(
                f"thinking = {'true' if self.thinking else 'false'}",
                "是否默认开启思考",
            ),
            _entry(f'effort = "{self.effort}"', "low | high | max"),
            "",
            "[files]",
            _entry(
                f'mode = "{self.files_mode}"',
                "off 关闭 | read 只读 | confirm 写需确认 | direct 写直通",
            ),
            _entry(
                f"inline_max_chars = {self.inline_max_chars}",
                "附件内联进消息的最大字符数，超出则截断并留 read_attachment",
            ),
            _entry(
                f'attachment_dir = "{self.attachment_dir}"',
                "附件正文落盘目录",
            ),
            "",
            "[web]",
            _entry(
                f"enabled = {'true' if self.web_enabled else 'false'}",
                "联网搜索与网页抓取",
            ),
            "",
            "[recall]",
            _entry(
                f"enabled = {'true' if self.recall_enabled else 'false'}",
                "对话树检索工具（也可在界面用 /recall on|off 切换）",
            ),
            "",
            "[prompt]",
            _entry(
                f'file = "{self.prompt_file}"',
                "系统提示词文件；客户端不显示、不可编辑",
            ),
            "",
        ]
        return "\n".join(lines)

    def with_files_mode(self, mode: str) -> "Config":
        from dataclasses import replace

        return replace(self, files_mode=mode if mode in FILE_MODES else self.files_mode)

    def with_web(self, enabled: bool) -> "Config":
        from dataclasses import replace

        return replace(self, web_enabled=bool(enabled))

    def with_recall(self, enabled: bool) -> "Config":
        from dataclasses import replace

        return replace(self, recall_enabled=bool(enabled))


# ---------------- 读取与校验 ----------------


def _as_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(lo, min(hi, number))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return default


def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _as_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def config_from_dict(data: dict, path: Optional[str] = None) -> Config:
    """把解析后的 TOML 字典转成 Config，非法值静默回退默认。"""
    files = data.get("files") if isinstance(data.get("files"), dict) else {}
    web = data.get("web") if isinstance(data.get("web"), dict) else {}
    recall = data.get("recall") if isinstance(data.get("recall"), dict) else {}
    prompt = data.get("prompt") if isinstance(data.get("prompt"), dict) else {}

    model = normalize_model_name(_as_str(data.get("model"), DEFAULT_MODEL))
    if model not in (MODEL_FLASH, MODEL_PRO):
        model = DEFAULT_MODEL

    effort = _as_str(data.get("effort"), DEFAULT_EFFORT).lower()
    if effort not in EFFORTS:
        effort = DEFAULT_EFFORT

    files_mode = _as_str(files.get("mode"), DEFAULT_FILE_MODE).lower()
    if files_mode not in FILE_MODES:
        files_mode = DEFAULT_FILE_MODE

    return Config(
        model=model,
        temperature=_as_float(
            data.get("temperature"), 1.0, TEMPERATURE_MIN, TEMPERATURE_MAX
        ),
        thinking=_as_bool(data.get("thinking"), False),
        effort=effort,
        files_mode=files_mode,
        inline_max_chars=_as_int(
            files.get("inline_max_chars"),
            INLINE_MAX_CHARS_DEFAULT,
            INLINE_MAX_CHARS_MIN,
            INLINE_MAX_CHARS_MAX,
        ),
        attachment_dir=os.path.expanduser(
            _as_str(files.get("attachment_dir"), default_attachment_dir())
        ),
        web_enabled=_as_bool(web.get("enabled"), True),
        recall_enabled=_as_bool(recall.get("enabled"), False),
        prompt_file=os.path.expanduser(
            _as_str(prompt.get("file"), default_prompt_path())
        ),
    )


def load_config(path: Optional[str] = None) -> Config:
    """读取配置；文件不存在或损坏时返回默认配置（不写盘）。"""
    target = os.path.expanduser(path) if path else config_path()
    try:
        with open(target, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return Config()
    if not isinstance(data, dict):
        return Config()
    return config_from_dict(data, target)


def save_config(cfg: Config, path: Optional[str] = None) -> str:
    """把配置按模板写回（值从当前状态填入，注释保留）。"""
    target = os.path.expanduser(path) if path else config_path()
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(cfg.to_toml(target))
    return target


def ensure_prompt_file(cfg: Config) -> str:
    """提示词文件不存在或为空时写入默认提示词，返回路径。"""
    target = os.path.expanduser(cfg.prompt_file)
    try:
        with open(target, "r", encoding="utf-8") as f:
            if f.read().strip():
                return target
    except OSError:
        pass
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(DEFAULT_PROMPT)
    return target


def ensure_config(path: Optional[str] = None) -> Config:
    """首次启动生成配置与提示词文件，返回生效配置。"""
    target = os.path.expanduser(path) if path else config_path()
    if not os.path.exists(target):
        save_config(Config(), target)
    cfg = load_config(target)
    ensure_prompt_file(cfg)
    return cfg


# ---------------- 其他运行期常量 ----------------

# 网页正文长度上限（web_fetch 一次返回给模型的字符数）
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
