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
DISPLAY_BODY_PADDING = 8
DISPLAY_BODY_MIN = 30
PREVIEW_USER_MSG_LEN = 100
PREVIEW_ASSISTANT_MSG_LEN = 200
WEBPAGE_MAX_LENGTH = 5000
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0

BOCHA_API_BASE = "https://api.bocha.cn/v1/web-search"
