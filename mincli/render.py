from rich.console import Console
from rich.theme import Theme

MD_THEME = Theme({
    "markdown.h1": "cyan",
    "markdown.h2": "cyan",
    "markdown.h3": "cyan",
    "markdown.h4": "cyan",
    "markdown.h5": "cyan",
    "markdown.h6": "cyan",
    "markdown.block_quote": "bright_black",
})
console = Console(stderr=True, highlight=False, theme=MD_THEME)
