import sys

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件的内容，支持 txt、md、py、bat、sh、csv、pdf、docx 格式，返回文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "文件路径，支持绝对路径和 ~ 开头的路径",
                    }
                },
                "required": ["filepath"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "抓取指定 URL 的网页内容并提取正文，返回网页标题和文本内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "网页 URL，如 https://example.com",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录的内容，可选择是否包含隐藏文件（以 . 开头的文件），默认不包含隐藏文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "目录路径，支持绝对路径和 ~ 开头的路径",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "是否包含隐藏文件，默认 false",
                    },
                },
                "required": ["directory"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件。如果文件不存在则创建新文件，存在则覆盖原内容。写入前会请求用户确认",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "文件路径，支持绝对路径和 ~ 开头的路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容",
                    },
                },
                "required": ["filepath", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "在文件中搜索 old_string 并替换为 new_string（仅替换第一个匹配项）。old_string 必须与文件内容精确匹配（包括空格和换行）。操作前会请求用户确认",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "文件路径，支持绝对路径和 ~ 开头的路径",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要被替换的精确原文（区分大小写、包含空格和换行）",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的新内容",
                    },
                },
                "required": ["filepath", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网信息。注意：该工具调用需要用户事先通过 /search 命令授权",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "freshness": {
                        "type": "string",
                        "description": "时间范围: noLimit(不限)/oneDay(一天内)/oneWeek(一周内)/oneMonth(一月内)/oneYear(一年内)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "返回结果条数(1-50), 默认10",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": f"在用户电脑上执行 shell 命令。当前操作系统: {sys.platform}（{'Windows' if sys.platform == 'win32' else 'macOS/Linux'}）。每个命令在执行前会经过 AI 安全审核和用户确认。默认工作目录为用户家目录。注意：若预计输出很长，请在命令中关闭输出（如 {'追加 >nul 2>&1' if sys.platform == 'win32' else '追加 >/dev/null 2>&1'}）以节省 token。必须设置 deadline（timeout 参数），超时后命令将被强制终止，但会返回已产生的部分输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "执行截止时间（秒）。必须设置，超时后命令会被强制终止，已产生的部分输出仍会返回",
                    },
                },
                "required": ["command", "timeout"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_conversation_tree",
            "description": "查询对话节点树的结构，返回文本。不传参数时返回所有子树索引；传root只查某子树（如root='a'）；传search按标题或内容关键词搜索节点",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "可选。子对话树的字母前缀，如 'a'、'b'。只返回该子树下所有节点的详细列表。不传则返回全局索引",
                    },
                    "search": {
                        "type": "string",
                        "description": "可选。关键词，搜索节点标题和用户消息中包含该词的节点",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_conversation_nodes",
            "description": "读取指定对话节点的完整内容（用户输入+思考过程+AI回答）。可一次读多个节点，用逗号分隔节点ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "string",
                        "description": "节点ID，多个用逗号分隔，如 'a1,a2,b1'",
                    }
                },
                "required": ["node_ids"],
                "additionalProperties": False,
            },
        },
    },
]
