> [🇬🇧 English](readme.md)

# mincli

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 DeepSeek V4 模型、构建在 **Textual TUI** 之上的树状对话 AI 助手。  
流式 Markdown 输出、树状对话分支、完整推理链显示。  
支持动态切换模型/提示词/温度/思考模式，AI 可自主调用文件读写、网页抓取、目录浏览、Shell 执行等工具。

---

## 特性

- 🖥️ **Textual TUI** — 左侧会话树 + 流式 Markdown 消息区 + 多行输入框
- 🚀 **流式输出** — 实时 Markdown 渲染，表格按终端宽度自动换行
- 🌲 **树状对话** — 主线＋分支节点，全局唯一 ID；点击节点切换，`/<id>` 直接跳转
- 🧠 **思考模式** — 支持 DeepSeek V4 推理链，可随时开关
- 🔧 **工具调用** — AI 自主调用工具：读/写/编辑文件、抓网页、列目录、执行命令（写/执行需用户确认）
- ⌨️ **命令补全** — 输入 `/` 弹出命令列表；字母过滤候选；`Tab` 循环/补全；命令补全后自动显示用法帮助
- 🛡️ **确认弹窗** — 破坏性操作（`/delete`、`/mcp remove`）需确认；`←`/`→` 切换按钮，默认选中"取消"
- 💾 **会话自动保存** — 退出自动保存，下次启动恢复
- 📄 **导出 Markdown** — `/save` 将节点对话导出为 `.md` 文件
- ⚙️ **动态配置** — `/set` 命令随时修改系统提示词、温度、模型、思考开关、推理强度
- 🧩 **双模型** — `deepseek-v4-flash`（轻量快速）和 `deepseek-v4-pro`（旗舰性能）

---

## 推荐终端

**macOS** 建议使用 [iTerm2](https://iterm2.com/)，键盘协议处理可靠（中文输入法、锁定键均正常）。  
其他终端（Windows Terminal、Linux 终端）也可用，输入法表现取决于终端。

---

## 安装

### 前置要求
- Python 3.10+

### 1. 克隆
```bash
git clone <仓库地址>
cd mincli
```

### 2. 安装（推荐虚拟环境）
```bash
python3 -m venv venv
source venv/bin/activate

# 开发模式安装（修改源码即时生效）
pip install -e .
```

如需加速依赖下载（国内）：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

### 3. 配置 API 密钥
```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

配置加载优先级（高 → 低）：
| 优先级 | 来源 |
|--------|------|
| 1 | Shell 环境变量 |
| 2 | `~/.mincli/.env` |
| 3 | 当前目录 `.env` |

### 4. 启动
```bash
# 推荐：Textual TUI
mincli chat

# 纯文本回退（无 TUI、无额外依赖）
mincli chat --no-tui

# Python 模块方式
python -m mincli chat

# 传统方式（兼容）
python main.py chat
```

---

## 快速开始

```bash
# 基本对话（TUI）
mincli chat

# 开启思考模式
mincli chat --thinking

# 指定模型 + 推理强度
mincli chat --model pro --thinking --effort max

# 查看所有选项
mincli chat --help
```

### TUI 快捷键

| 按键 | 作用 |
|------|------|
| `Enter` | 发送消息 |
| `Ctrl+J` / `Alt+Enter` | 换行 |
| `Tab` | 命令补全 / 循环切换补全候选 |
| `↑` / `↓` | 滚动回答区（输入框为空时）；双击按住为 2 倍速 |
| `Ctrl+C` | 退出（选中文字时优先复制） |

### 对话中常用操作
```
# 导入文件或网页后提问
/import ~/document.pdf
这个文档讲了什么？

# AI 自主工具调用（直接提问即可）
帮我看看 config.json 的内容
查一下 https://example.com
当前目录下有哪些文件？
```

---

## 配置

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | 无 | DeepSeek API 密钥 |
| `MINCLI_SAVE_PATH` | 否 | `~/Documents/mincli_Conversations` | 对话导出目录 |

命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m` / `--model` | `flash` | 模型：`flash` \| `pro` |
| `--thinking` | 关 | 开启思考模式 |
| `--effort` | `high` | 推理强度：`low` \| `high` \| `max` |
| `--temp` | `1.0` | 温度参数 |
| `--no-tui` | 关 | 使用极简纯文本对话（不使用 Textual TUI） |

---

## 交互命令

| 命令 | 说明 |
|------|------|
| `/exit`, `/quit` | 退出（自动保存会话） |
| `/clear`, `/c` | 清空会话 |
| `/set system <内容>` | 修改系统提示词 |
| `/set temp <数值>` | 修改温度 |
| `/set model <flash\|pro>` | 切换模型 |
| `/set thinking <on\|off>` | 开关思考模式 |
| `/set effort <low\|high\|max>` | 推理强度 |
| `/set show` | 显示当前配置 |
| `/mcp list` | 显示 MCP server 配置与连接状态 |
| `/mcp add <名称> <命令> [参数...]` | 添加第三方 MCP server（本地命令）；第二参数为 `http(s)://` 地址时按远程 server 添加 |
| `/mcp remove <名称>` | 移除第三方 MCP server（需确认） |
| `/mcp reload` | 重新加载 MCP server 配置 |
| `/import <路径或URL>` | 导入文件（txt/md/py/csv/pdf/docx）或抓取网页 |
| `/<节点ID>`（如 `/a3`） | 直接跳转到指定节点 |
| `/tree` | 列出所有节点 |
| `/info [节点ID]` | 查看节点详情 |
| `/up` | 回到父节点 |
| `/home` | 跳回根节点 |
| `/full` | 全览模式：隐藏回答区，节点树全宽（输入框保留；再按一次或切换节点退出） |
| `/reasoning` | 展开/折叠当前消息的思考过程（正文开始后自动折叠，也可点击折叠块展开） |
| `/save [节点ID]` | 导出节点为 Markdown |
| `/delete <节点ID>` | 删除节点及其子节点（需确认） |
| `/view` | 用编辑器打开当前回答 |

在输入框输入 `/` 即可看到命令列表：继续输入字母过滤候选，`Tab` 补全；命令完整输入后输入框上方会自动显示用法帮助。

---

## AI 工具参考

AI 在对话中视需要自主调用以下工具：

| 工具 | 功能 | 参数 |
|------|------|------|
| `read_file` | 读取文件（txt/md/py/csv/pdf/docx） | `filepath` |
| `fetch_webpage` | 抓取网页并提取正文 | `url` |
| `list_directory` | 列出目录内容 | `directory`; `show_hidden`（可选） |
| `write_file` | 写入/覆盖文件（需用户确认） | `filepath`; `content` |
| `edit_file` | 搜索替换文件内容（需用户确认） | `filepath`; `old_string`; `new_string` |
| `execute_command` | 执行 Shell 命令（AI 审核 + 用户确认） | `command`; `timeout` |
| `query_conversation_tree` | 查询对话树结构（内存内，不走 MCP） | `root`; `search`（可选） |
| `read_conversation_nodes` | 读取对话节点内容（内存内，不走 MCP） | `node_ids` |

---

## MCP 接入

mincli 的工具执行基于标准 [MCP 协议](https://modelcontextprotocol.io/)（Model Context Protocol）：

- **自建 MCP server**：6 个外部工具（文件读写、网页抓取、命令执行）由 mincli 启动的子进程 server 提供，client 通过 stdio 调用。安全/交互策略（用户确认、AI 审核）仍留在客户端，行为与之前一致。
- **对话树工具**（`query_conversation_tree` / `read_conversation_nodes`）依赖内存中的对话状态，保留在进程内直接分发。

### 接入第三方 MCP server

在 `~/.mincli/mcp_servers.json` 配置（Claude Desktop 兼容格式，可用 `MINCLI_MCP_CONFIG` 改路径），或在对话中用 `/mcp add` 交互式添加、`/mcp list` 查看状态、`/mcp reload` 生效：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    }
  }
}
```

启动后第三方 server 的工具会自动合并进 AI 工具列表；与已有工具重名时以 mincli 自带的为准。

**支持两种类型的第三方 server：**
- **本地命令（stdio）**：`command` + `args` + 可选 `env`，如上面的 filesystem 示例
- **远程 HTTP（streamable-http）**：只写 `url` 即可，如：

```json
{
  "mcpServers": {
    "remote-tools": { "url": "https://example.com/mcp" }
  }
}
```

对话中也可直接 `/mcp add <名称> <URL>` 添加远程 server。

---

## 项目结构

```
.
├── main.py                  # 入口（兼容 python main.py）
├── pyproject.toml           # 项目元数据 + 依赖声明
├── .env.example             # 配置模板
├── readme.md                # 英文文档
├── readme.zh.md             # 中文文档
│
├── mincli/                  # 核心包
│   ├── __init__.py          # 版本号
│   ├── __main__.py          # python -m mincli 入口
│   ├── cli.py               # Typer CLI：chat（TUI / --no-tui）、info
│   ├── config.py            # 常量 + 配置加载
│   ├── controller.py        # ChatController（核心逻辑 + 事件流）
│   ├── models.py            # ConversationNode/Tree
│   ├── helpers.py           # 工具函数（token/标题/公式转换）
│   ├── streaming.py         # 流式 API 交互
│   ├── mcp_client.py        # MCP 客户端（异步桥 + 自建/第三方 server）
│   ├── mcp_server.py        # 自建 MCP server
│   ├── tui/                 # Textual TUI
│   │   ├── app.py           # ChatApp（布局、命令、事件处理）
│   │   ├── chat.tcss        # TUI 样式
│   │   ├── confirm.py       # 确认弹窗（←/→ 切换，默认取消）
│   │   └── widgets.py       # ChatInput（多行输入 + 命令补全）
│   └── tools/
│       ├── registry.py      # 本地工具定义列表（对话树工具）
│       ├── execute.py       # 命令执行 + AI 安全审计
│       ├── file_ops.py      # 文件读写/解析
│       ├── web_fetch.py     # 网页抓取 + 搜索
│       └── thinking.py      # 审计系统提示词
│
└── tests/                   # Headless 测试（test_controller / test_tui）
```

---

## 常见问题

**Q：启动后提示"会话文件损坏"？**  
A：删除 `~/.mincli_session.json` 后重启。

**Q：思考模式开启但看不到推理过程？**  
A：请确认使用 `flash` 或 `pro` 模型并已开启 `--thinking`。

**Q：`/import` 导入 PDF/DOCX 报错？**  
A：确保依赖已安装：`pip install pdfminer.six python-docx`。

**Q：TUI 无法启动（如输出被重定向、终端不支持）？**  
A：使用 `mincli chat --no-tui` 走纯文本回退模式。

---

## 许可

MIT License
