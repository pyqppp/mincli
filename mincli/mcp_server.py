import asyncio
from typing import Dict, Optional

from mcp.server import MCPServer

from mincli.config import EXEC_DEFAULT_MAX_OUTPUT, EXEC_DEFAULT_TIMEOUT
from mincli.tools.file_ops import (
    parse_file, list_directory as _list_directory,
    write_file_content, edit_file_content,
)
from mincli.tools.web_fetch import fetch_webpage as _fetch_webpage
from mincli.tools.execute import execute_command as _execute_command

mcp = MCPServer("mincli")


@mcp.tool()
async def read_file(filepath: str) -> str:
    """读取本地文件的内容。支持 txt、md、py、csv、pdf、docx 等常见格式，也支持其他纯文本文件。二进制文件不可读取

    Args:
        filepath: 文件路径，支持绝对路径和 ~ 开头的路径
    """
    return await asyncio.to_thread(parse_file, filepath)


@mcp.tool()
async def fetch_webpage(url: str) -> str:
    """抓取指定 URL 的网页内容并提取正文，返回网页标题和文本内容

    Args:
        url: 网页 URL，如 https://example.com
    """
    return await asyncio.to_thread(_fetch_webpage, url)


@mcp.tool()
async def list_directory(directory: str, show_hidden: bool = False) -> str:
    """列出指定目录的内容，可选择是否包含隐藏文件（以 . 开头的文件），默认不包含隐藏文件

    Args:
        directory: 目录路径，支持绝对路径和 ~ 开头的路径
        show_hidden: 是否包含隐藏文件，默认 false
    """
    return await asyncio.to_thread(_list_directory, directory, show_hidden)


@mcp.tool()
async def write_file(filepath: str, content: str) -> str:
    """将内容写入文件。如果文件不存在则创建新文件，存在则覆盖原内容。写入前会请求用户确认

    Args:
        filepath: 文件路径，支持绝对路径和 ~ 开头的路径
        content: 要写入的文件内容
    """
    return await asyncio.to_thread(write_file_content, filepath, content)


@mcp.tool()
async def edit_file(filepath: str, old_string: str, new_string: str) -> str:
    """在文件中搜索 old_string 并替换为 new_string（仅替换第一个匹配项）。old_string 必须与文件内容精确匹配（包括空格和换行）。操作前会请求用户确认

    Args:
        filepath: 文件路径，支持绝对路径和 ~ 开头的路径
        old_string: 要被替换的精确原文（区分大小写、包含空格和换行）
        new_string: 替换后的新内容
    """
    return await asyncio.to_thread(edit_file_content, filepath, old_string, new_string)


@mcp.tool()
async def execute_command(
    command: str,
    timeout: int = EXEC_DEFAULT_TIMEOUT,
    cwd: str = "",
    env: Optional[Dict[str, str]] = None,
    shell: str = "sh",
    max_output: int = EXEC_DEFAULT_MAX_OUTPUT,
) -> str:
    """在用户电脑上执行 shell 命令。每个命令在执行前会经过 AI 安全审核和用户确认；命中高危模式的命令（如 rm -rf /、dd 写磁盘、curl|bash 等）无论审核结果如何都会强制要求用户确认。命令以非交互方式运行（stdin 已关闭），交互式命令（vim、ssh、python REPL 等）会因无输入而立即结束。默认工作目录为 mincli 启动目录（可用 /set workspace 修改，也可用 cwd 参数临时指定）。若预计输出很长，请在命令中限制输出（如追加 | head、>/dev/null）以节省 token

    Args:
        command: 要执行的 shell 命令（可用 && 串联多条）
        timeout: 执行截止时间（秒），默认 30，上限 120。超时后整个进程组会被强制终止，已产生的部分输出仍会返回
        cwd: 可选。工作目录（绝对路径或 ~ 开头）；留空时使用 /set workspace 设置的目录，未设置则用 mincli 启动目录
        env: 可选。额外环境变量字典，如 {"PATH": "/usr/local/bin:..."}，叠加到当前环境
        shell: 可选。使用的 shell：sh、bash 或 zsh，默认 sh
        max_output: 可选。输出截断上限（字符），默认 8000；超出时保留首尾并写入 /tmp 临时文件，路径随结果返回
    """
    return await asyncio.to_thread(
        _execute_command, command, timeout,
        cwd=cwd, env=env or None, shell=shell, max_output=max_output,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
