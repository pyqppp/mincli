"""对话树的 8 套配色主题。

一棵树 = 一个颜色，颜色由编号固定推导（1..8 循环，见 `trees.color_of`）。
8 套主题共用同一套近黑灰背景与面板色，只轮换主色、强调色、边框、Markdown
标题、滚动条与选区——切树时整机换色但底色不变，不会因为明暗跳变而刺眼。

树 1 就是 mincli 原有的青色主题（数值与历史版本完全一致，观感不变）。
"""

from typing import List

from textual.theme import Theme

# 每套颜色的取值：primary 主色（边框/按钮/选中）、accent 强调（聚焦边框/标题）、
# secondary 辅助、scrim_* 为滚动条相关的暗色调（保持同一亮度层级）。
TREE_PALETTE: List[dict] = [
    {
        "key": "cyan",
        "label": "青",
        "primary": "#00d0dc",
        "accent": "#00e5ff",
        "secondary": "#008394",
        "border": "#00dce6",
        "scrollbar": "#1e6272",
        "scrollbar_hover": "#2b8494",
        "scrim": "#07141a",
        "scrim_hover": "#0a1c24",
        "link_hover": "#7ff3ff",
    },
    {
        "key": "purple",
        "label": "紫",
        "primary": "#b58cff",
        "accent": "#cfaaff",
        "secondary": "#6a4a9e",
        "border": "#bb93ff",
        "scrollbar": "#4a3670",
        "scrollbar_hover": "#63498f",
        "scrim": "#0d0a16",
        "scrim_hover": "#150f20",
        "link_hover": "#e3d2ff",
    },
    {
        "key": "green",
        "label": "绿",
        "primary": "#4fd07a",
        "accent": "#6ee79a",
        "secondary": "#2f8a4f",
        "border": "#55d883",
        "scrollbar": "#27603a",
        "scrollbar_hover": "#33804d",
        "scrim": "#081410",
        "scrim_hover": "#0b1e17",
        "link_hover": "#a8f5c4",
    },
    {
        "key": "orange",
        "label": "橙",
        "primary": "#ffa94d",
        "accent": "#ffc078",
        "secondary": "#b06a26",
        "border": "#ffb05c",
        "scrollbar": "#6b4520",
        "scrollbar_hover": "#8c5a2a",
        "scrim": "#160f08",
        "scrim_hover": "#20160b",
        "link_hover": "#ffd9a8",
    },
    {
        "key": "pink",
        "label": "粉",
        "primary": "#ff7fb0",
        "accent": "#ff9ec7",
        "secondary": "#b04070",
        "border": "#ff8cb8",
        "scrollbar": "#6b2f47",
        "scrollbar_hover": "#8c3d5c",
        "scrim": "#160a10",
        "scrim_hover": "#200f17",
        "link_hover": "#ffc7dd",
    },
    {
        "key": "blue",
        "label": "蓝",
        "primary": "#5aa9ff",
        "accent": "#85c2ff",
        "secondary": "#3a7ab5",
        "border": "#69b1ff",
        "scrollbar": "#274a70",
        "scrollbar_hover": "#33608f",
        "scrim": "#080f16",
        "scrim_hover": "#0b1620",
        "link_hover": "#b8dbff",
    },
    {
        "key": "yellow",
        "label": "黄",
        "primary": "#e8c547",
        "accent": "#f5da6b",
        "secondary": "#a8892f",
        "border": "#eed05a",
        "scrollbar": "#615322",
        "scrollbar_hover": "#7d6b2c",
        "scrim": "#141206",
        "scrim_hover": "#1e1a09",
        "link_hover": "#fbe9a8",
    },
    {
        "key": "red",
        "label": "红",
        "primary": "#ff6b6b",
        "accent": "#ff8f8f",
        "secondary": "#b04040",
        "border": "#ff7a7a",
        "scrollbar": "#6b2b2b",
        "scrollbar_hover": "#8c3a3a",
        "scrim": "#160a0a",
        "scrim_hover": "#200f0f",
        "link_hover": "#ffbcbc",
    },
]

TREE_COLOR_COUNT = len(TREE_PALETTE)


def clamp_color(color: int) -> int:
    """把颜色序号夹到 1..8（非法值退回 1）。"""
    try:
        n = int(color)
    except (TypeError, ValueError):
        return 1
    if n < 1 or n > TREE_COLOR_COUNT:
        return ((n - 1) % TREE_COLOR_COUNT) + 1 if n > 0 else 1
    return n


def theme_name(color: int) -> str:
    """颜色序号 → Textual 主题名。"""
    return f"mincli-tree-{clamp_color(color)}"


def color_label(color: int) -> str:
    """颜色序号 → 中文色名（/tree 列表用）。"""
    return TREE_PALETTE[clamp_color(color) - 1]["label"]


def _build_theme(color: int) -> Theme:
    p = TREE_PALETTE[clamp_color(color) - 1]
    return Theme(
        name=theme_name(color),
        primary=p["primary"],
        secondary=p["secondary"],
        accent=p["accent"],
        # 语义色（警告/错误/成功）8 套主题保持一致：它们的含义不该随树改变
        warning="#ffb454",
        error="#ff6b81",
        success="#3dd68c",
        foreground="#ffffff",
        background="#0a0d10",
        surface="#101a20",
        panel="#15222b",
        variables={
            "text": "#ffffff",
            "scrollbar": p["scrollbar"],
            "scrollbar-hover": p["scrollbar_hover"],
            "scrollbar-active": p["accent"],
            "scrollbar-background": p["scrim"],
            "scrollbar-background-hover": p["scrim_hover"],
            "scrollbar-background-active": p["scrim_hover"],
            "scrollbar-corner-color": p["scrim"],
            "border": p["border"],
            "border-blurred": "#14323a" if color == 1 else p["secondary"],
            "block-cursor-background": p["accent"],
            "block-cursor-foreground": "#041114",
            "block-cursor-text-style": "none",
            "input-selection-background": f"{p['accent']} 35%",
            "screen-selection-background": f"{p['accent']} 35%",
            "markdown-h1-color": p["accent"],
            "markdown-h1-text-style": "bold",
            "markdown-h2-color": p["primary"],
            "markdown-h2-text-style": "underline",
            "markdown-h3-color": p["primary"],
            "markdown-h4-color": p["primary"],
            "markdown-h5-color": p["primary"],
            "markdown-h6-color": p["primary"],
            "link-color": p["accent"],
            "link-color-hover": p["link_hover"],
            "footer-key-foreground": p["accent"],
        },
    )


# 8 套树主题（下标 0..7 对应颜色序号 1..8）
TREE_THEMES: List[Theme] = [_build_theme(i) for i in range(1, TREE_COLOR_COUNT + 1)]
