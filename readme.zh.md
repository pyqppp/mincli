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
- 🧩 **三模型** — `deepseek-v4-flash`（轻量快速）、`deepseek-v4-pro`（旗舰性能）、`deepseek-v4-flash-vision-exp`（多模态图像理解）
- 🖼️ **多模态（图片理解）** — `/img` 或 `/import 图片` 附带图片提问（本地文件自动上传 Files API，跨轮复用 file_id；上传失败自动回退 base64 内联；图片消息自动切换视觉模型）
- 📊 **实时用量状态条** — 输入栏下方两分栏实时显示：缓存命中率、账户余额、下一次输入 token 估算与预计价格（按 DeepSeek 峰谷分时定价折算）

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
| 3 | 项目目录 `.env` |

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
| `MINCLI_SYSTEM_PROMPT_PATH` | 否 | 包内 `mincli/system_prompt.md` | 自定义系统提示词文件路径 |

### 系统提示词

系统提示词独立存放在文件中，每次启动自动导入。加载优先级（高 → 低）：

| 优先级 | 来源 |
|--------|------|
| 1 | `MINCLI_SYSTEM_PROMPT_PATH` 环境变量指定的文件 |
| 2 | `~/.mincli/system_prompt.md` |
| 3 | 包内 `mincli/system_prompt.md`（默认，随项目分发） |

直接编辑对应文件即可自定义默认提示词，重启后生效；`mincli info` 可查看当前实际使用的提示词文件。若文件均不可用，则回退到内置兜底提示词。

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
| `/compact [N]` | 压缩上下文：把当前分支早期对话压成**详细**摘要，保留最近 N 轮原文（默认 5，0=全部压缩；再次执行会按最新对话重新压缩） |
| `/compact off` | 清除压缩摘要，恢复发送完整原始消息 |
| `/set system <内容>` | 修改系统提示词 |
| `/set temp <数值>` | 修改温度 |
| `/set model <flash\|pro\|vision\|模型名>` | 切换模型（vision = `deepseek-v4-flash-vision-exp` 图像理解模型） |
| `/set thinking <on\|off>` | 开关思考模式 |
| `/set effort <low\|high\|max>` | 推理强度 |
| `/set audit <1-4>` | 命令审核层级（1=AI审核+确认 / 2=低风险自动 / 3=文本匹配 / 4=无审核） |
| `/set workspace <路径>` | 命令执行默认工作目录（默认 mincli 启动目录） |
| `/set detail <low\|auto\|high\|original>` | 图片清晰度（low 缩放 512² 更省 token；auto≈original 保留原图最清晰） |
| `/set show` | 显示当前配置 |
| `/mcp list` | 显示 MCP server 配置与连接状态 |
| `/mcp add <名称> <命令> [参数...] [--header 'K: V']` | 添加第三方 MCP server（本地命令）；第二参数为 `http(s)://` 地址时按远程 server 添加，`--header` 用于远程 server 的鉴权请求头 |
| `/mcp remove <名称>` | 移除第三方 MCP server（需确认） |
| `/mcp reload` | 重新加载 MCP server 配置 |
| `/import <路径或URL>` | 导入文件（txt/md/py/csv/pdf/docx）或抓取网页；**图片文件（jpg/png/gif/webp）自动转为待发送图片** |
| `/img <路径或URL> [...]` | 添加待发送图片（多图可一次添加；`/img clear` 清除；发送时自动附带并切换视觉模型） |
| `/files list` | 列出 Files API 已上传的图片文件（ID/文件名/大小/过期） |
| `/files delete <ID>` | 删除一个已上传的图片文件 |
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
| `execute_command` | 执行 Shell 命令（AI 审核 + 用户确认） | `command`; `timeout`; `cwd`; `env`; `shell`; `max_output`（除 `command` 外均可选） |
| `query_conversation_tree` | 查询对话树结构（内存内，不走 MCP） | `root`; `search`（可选） |
| `read_conversation_nodes` | 读取对话节点内容（内存内，不走 MCP） | `node_ids` |

### 命令执行工具（execute_command）

`execute_command` 让 AI 在你的电脑上执行 shell 命令，带多层安全与可用性设计：

- **审核分级**（`/set audit <1-4>`）：默认 1 = AI 审核 + 用户确认；2 = AI 审核、低风险自动执行；3 = 仅文本匹配；4 = 直接执行。
- **高危硬门**：命中高危模式（`rm -rf /`、`dd` 写磁盘、`curl|bash`、`git push --force`、`pkill` 等）时，除「无审核」外一律强制用户确认，不受 AI 评级影响。
- **只读快速通道**：无 shell 元字符的纯只读命令（`ls`、`cat`、`pwd` 等）跳过 AI 审核（省时省 token）；level-1 仍会确认，level-2 自动执行。
- **审核缓存**：同一命令在一次会话内只审核一次，重复执行直接复用结果。
- **工作目录**：默认是 mincli 启动目录；可用 `/set workspace <路径>` 持久化修改，AI 也可用 `cwd` 参数临时指定。
- **可调参数**：`timeout`（默认 30s、上限 120s，超时终止整个进程组并返回部分输出）、`shell`（sh/bash/zsh）、`env`（额外环境变量）、`max_output`（输出截断上限，默认 8000 字符，超限保留首尾并把完整输出写入 `/tmp/mincli_exec_*.txt` 供 `read_file` 读取）。
- **非交互执行**：命令 stdin 已关闭（防止 vim/ssh 等交互命令破坏 TUI 或挂死）。

---

## 多模态（图片理解）

基于官方视觉模型 `deepseek-v4-flash-vision-exp`（实验性，与 flash 同价），支持在提问时附带图片：描述图片、识别截图文字、分析图表等。

### 用法

```text
/img ~/截图.png "https://example.com/chart.jpg"     # 添加待发送图片（可多个）
/img clear                                            # 清除待发送图片
/import 图片.png                                      # 图片文件自动转为待发送图片
发送消息                                              # 图片自动附带（自动切换视觉模型）
```

- 待发送图片会显示在输入框下方（`📷 待发送图片 N 张 …`），发送后绑定到该对话节点。
- 图片消息会自动把模型切换到 `deepseek-v4-flash-vision-exp`（提示后生效；`/set model flash` 可切回）。
- 聊天区以 `[图片: 文件名 (宽x高)]` 文本占位展示（不依赖终端图像协议，可选中/拷贝）。

### 传图机制（Files API 优先）

1. **上传一次，跨轮复用**：本地图片首次发送时上传到 DeepSeek Files API，得到 `file_id`（`file-api-...`），节点持久化该 ID；后续所有请求（含历史重放、分支追问）都通过 `file` 内容块引用，请求体极小、序列化稳定，**不破坏前缀缓存**。
2. **自动回退**：上传失败时自动改用 base64 内联发送（单图 ≤32MiB、请求体 ≤48MiB 预检，超限报错提示）。
3. **文件管理**：`/files list` 查看已上传文件，`/files delete <ID>` 删除；删除对话节点时会尽力清理其关联文件。
4. **外部 URL**：直接传 `http(s)` 链接（≤8192 字符，API 下载）。

### 图片 token 与成本（实测校准）

- 图片按尺寸换算 token 与文本一并计费：**每图固定开销 ≈117 token**，尺寸附加额按面积增长、**封顶 ≈240**（单图合计最高 ~357 token）；`detail=low` 缩放 512²，可省约 60%。
- **实测：图片部分不参与前缀缓存**（无论 base64 还是 file_id 均按未命中价计费）；文本前缀可正常缓存（同一前缀二次请求命中率 ~99%）。
- 因此成本估算宜按未命中价（1.5 元/百万 token）计；具体以接口返回 `usage` 为准（状态条/对话结束显示均取真实 usage）。

### 限制（官方）

| 项 | 值 |
|---|---|
| 格式 | JPEG / PNG / GIF / WebP（**按文件内容识别**，不看扩展名） |
| 内联单图 | ≤32 MiB（Files API 引用最大 64 MiB） |
| 请求体 | ≤48 MiB |
| 图片位置 | 仅 user 消息（system/assistant 带图会 400） |
| 单请求图片数 | ≤600；单边 ≤8192px（≥15 张时 4096px） |
| Files API | 单文件 ≤64 MiB、默认永久有效、存储 25 GiB / 1 万个文件 |

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
- **远程 HTTP（streamable-http）**：只写 `url` 即可；需要鉴权（如 Bearer Token）时用 `headers` 附加请求头，如：

```json
{
  "mcpServers": {
    "obsidian": {
      "url": "http://localhost:3001/mcp",
      "headers": { "Authorization": "Bearer <你的 Token>" }
    }
  }
}
```

对话中也可直接添加，例如：

```
/mcp add obsidian http://localhost:3001/mcp --header "Authorization: Bearer <你的 Token>"
```

`--header 'K: V'`（或 `-H`）可重复使用，仅对远程 server 生效；添加后运行 `/mcp reload` 生效。

---

## 上下文压缩（/compact）

对话变长后上下文占用大量 token，可用 `/compact` 把**当前分支**的早期对话压缩成一份**详尽**的摘要（由模型生成，按主题组织、完整保留目标/约束/决定/命令/路径/数据/待办等关键信息，宁可长、不要短），之后发送消息时自动用摘要替代被压缩部分的原始消息，仅保留最近 N 轮原文：

```
/compact         # 压缩早期对话，保留最近 5 轮原文
/compact 0       # 全部压缩，不保留原文
/compact 3       # 保留最近 3 轮原文
/compact off     # 清除压缩摘要，恢复发送完整原始消息
```

说明：

- 压缩**不删除任何节点**：对话树、历史、导出均不受影响，随时可 `/compact off` 恢复原文；再次执行 `/compact` 会基于最新对话重新生成摘要。
- 摘要与当前节点一起保存到会话文件，重启后仍生效。
- 压缩走当前模型（`/set model` 指定的模型），摘要质量随模型而定。

---

## 实时用量状态条

输入栏下方有两分栏状态条（仅适配 DeepSeek API），所有数据均来自 API 返回，实时更新：

**左栏：缓存命中率 + 账户余额**
- 缓存命中率 = `usage.prompt_cache_hit_tokens ÷ (prompt_cache_hit_tokens + prompt_cache_miss_tokens)`，取当前节点最近一次请求的累计值（DeepSeek 上下文缓存自动生效，命中部分按缓存命中价计费）
- 账户余额来自 `GET /user/balance` 的 `total_balance`（优先 CNY），每 60 秒自动刷新

**右栏：下次输入估算**
- 下次输入 token：**未压缩**时 = 上次完整输入 + 本节点输出（DeepSeek API 真实 `usage` 口径，与对话结束显示的输入/输出直接对应）；**压缩后** = `/compact` 报告的 after_tokens（压缩时写入），与压缩报告数字严格一致，直观体现节省；用户新输入内容量小，忽略不计
- 预计价格 = token 量 × 折算单价 ÷ 100 万，按当前时段（北京时间高峰 9-12、14-18 点）与缓存命中率折算（命中部分按缓存命中价、其余按未命中价），定价见 `config.DEEPSEEK_PRICING`

> 口径说明：对话结束显示的「输入/输出 tokens」来自 DeepSeek API 真实 `usage`；压缩报告的 before/after 用 tiktoken 本地估算同一份消息（压缩前后无可用的新 API 请求），两者因 tokenizer 差异数字不等，但压缩后状态条与压缩报告严格一致、压缩前后状态条与 API 显示直接对应。

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
│   ├── system_prompt.md     # 系统提示词（每次启动自动导入）
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
│       ├── images.py        # 图片附件：格式嗅探/尺寸解析/base64/token 估算
│       ├── files.py         # Files API 客户端（上传/列表/删除）
│       └── thinking.py      # 审计系统提示词
│
└── tests/                   # Headless 测试（test_controller / test_tui / test_images）
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

**Q：图片消息报错 "This model does not support image"？**  
A：图片必须使用视觉模型。mincli 在带图发送时会自动切换到 `deepseek-v4-flash-vision-exp`；若手动 `/set model` 切换回了非视觉模型，发送图片时会提示切换。

**Q：上传图片失败？**  
A：会自动回退为 base64 内联发送（仅影响请求体大小）；可通过 `/files list` 查看已上传文件。单文件超过 32 MiB 或格式不支持（仅 JPEG/PNG/GIF/WebP，按内容识别）会报错提示。

**Q：图片 token 消耗大吗？**  
A：实测每图约 117~357 token（尺寸相关，`/set detail low` 可省约 60%）；图片部分不参与前缀缓存（按未命中价计费），文本前缀可正常缓存。具体以接口返回 `usage` 为准。

---

## 许可

MIT License
