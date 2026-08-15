"""mincli 的 Textual TUI 包。

导入本包（或任何 mincli.tui 子模块）之前，请确保没有任何 `textual`
模块被导入，因为本模块会在导入时配置键盘协议相关的环境变量。
"""

from __future__ import annotations

import os

# iTerm2 与 kitty 键盘协议（CSI u）冲突：
# - 开启 kitty 协议后，iTerm2 绕过自身的 IME 组合流程，中文输入法无法工作；
# - 同时 iTerm2 把 Caps Lock 上报为裸大写字母（表现为敲出 "A"），
#   应用层无法与真实按键区分。
# 因此默认禁用 kitty 协议，让终端自行处理 IME 与锁定键。
# 用 setdefault：用户若显式设置 TEXTUAL_DISABLE_KITTY_KEY，则以用户为准。
os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")


def __getattr__(name: str):
    # 惰性导出 ChatApp：避免 `python -m mincli.tui.app` 时
    # runpy 先导入本包、再导入 .app 导致的 “found in sys.modules” 警告。
    if name == "ChatApp":
        from .app import ChatApp

        return ChatApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ChatApp"]
