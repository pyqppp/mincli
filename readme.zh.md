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
- 🔁 **工作流（/wf）** — 把某次/一连串已完成操作保存为可复用工作流并长期保存；`/wf use` 挂载到下次输入、`/wf run` 立即执行，重复任务无需再描述
- ⌨️ **命令补全** — 输入 `/` 弹出命令列表；字母过滤候选；`Tab` 循环/补全；命令补全后自动显示用法帮助
- 🛡️ **确认弹窗** — 破坏性操作（`/delete`、`/mcp remove`）需确认；`←`/`→` 切换按钮，默认选中"取消"
- 💾 **会话自动保存** — 退出自动保存，下次启动恢复
- ♻️ **中断可继续** — API 报错/中断也会保留节点（含已生成的部分回答、思考与工具结果），树上标 `⚠ 中断`，输入「继续」接着生成，不再整轮丢失
- 📄 **导出 Markdown** — `/save` 将节点对话导出为 `.md` 文件
- ⚙️ **动态配置** — `/set` 命令随时修改系统提示词、温度、模型、思考开关、推理强度
- 🧩 **两模型** — `deepseek-flash`（DeepSeek-V4.1-Flash，轻量快速 + 原生多模态图像理解）、`deepseek-v4-pro`（旗舰性能，不支持图片）
- 🖼️ **多模态（图片理解）** — `/import 图片` 附带图片提问（本地文件自动上传 Files API，跨轮复用 file_id；上传失败自动回退 base64 内联；图片理解仅 `deepseek-flash` 支持，其他模型会提示切换；支持一次导入多个文件/图片）
- 📊 **实时用量状态条** — 输入栏下方两分栏实时显示：缓存命中率、账户余额、下一次输入 token 估算与预计价格（按 DeepSeek 峰谷分时定价折算；价格/峰谷时段/图片 token 均可在 `~/.mincli/pricing.json` 覆盖）

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
| `/compact` | 压缩上下文：把当前分支全部对话压成**详细**摘要，并**新建摘要节点**（在摘要节点输入基于摘要继续对话；其他节点仍使用完整历史） |
| `/wf list` | 列出所有已保存的工作流 |
| `/wf show <名>` | 查看工作流完整规范（目标/变量/步骤与命令细节） |
| `/wf save <名> [起点节点ID]` | 把**当前节点**（或「起点节点 → 当前节点」的一连串操作）提炼为可复用工作流并长期保存；同名需确认覆盖 |
| `/wf use <名>` | 挂载工作流到**下一次输入**：发送下一条消息即按工作流执行（一次性；`/wf stop` 取消） |
| `/wf run <名> [键=值...]` | **立即**按工作流执行，无需再输入（未提供值的变量由模型结合当前情况推断） |
| `/wf edit <名> [修改要求]` | 带要求=让模型按你的要求修订规范；macOS 下不带要求=用系统编辑器打开修改（保存后自动回写） |
| `/wf rename <旧名> <新名>` / `/wf delete <名>` | 重命名 / 删除工作流（删除需确认） |
| `/set system <内容>` | 修改系统提示词 |
| `/set temp <数值>` | 修改温度 |
| `/set model <flash\|pro\|模型名>` | 切换模型（`flash` = `deepseek-flash`，支持图片理解；`vision` 为 flash 的旧别名） |
| `/set thinking <on\|off>` | 开关思考模式 |
| `/set effort <low\|high\|max>` | 推理强度 |
| `/set audit <1-4>` | 命令审核层级（1=AI审核+确认 / 2=低风险自动 / 3=文本匹配 / 4=无审核） |
| `/set file_confirm <on\|off>` | 写文件/编辑文件时是否弹窗确认（默认 on；off 时 AI 可直接写入/修改文件） |
| `/set workspace <路径>` | 命令执行默认工作目录（默认 mincli 启动目录） |
| `/set detail <low\|auto\|high\|original>` | 图片清晰度（low 缩放 512² 更省 token；auto≈original 保留原图最清晰） |
| `/set show` | 显示当前配置 |
| `/mcp list` | 显示 MCP server 配置与连接状态 |
| `/mcp add <名称> <命令> [参数...] [--header 'K: V']` | 添加第三方 MCP server（本地命令）；第二参数为 `http(s)://` 地址时按远程 server 添加，`--header` 用于远程 server 的鉴权请求头 |
| `/mcp remove <名称>` | 移除第三方 MCP server（需确认） |
| `/mcp reload` | 重新加载 MCP server 配置 |
| `/import <路径或URL> [...]` | 导入文件（txt/md/py/csv/pdf/docx）、抓取网页或添加图片，可一次导入多个；**图片文件（jpg/png/gif/webp）自动转为待发送图片**；`/import clear` 清除待导入内容。路径解析跨平台：Windows 反斜杠路径（`C:\Users\me\a.txt`）与带引号/空格路径均可 |
| `/files list` | 列出 Files API 已上传的图片文件（ID/文件名/大小/过期） |
| `/files delete <ID>` | 删除一个已上传的图片文件 |
| `/<节点ID>`（如 `/a3`） | 直接跳转到指定节点 |
| `/tree` | 列出所有节点 |
| `/info [节点ID]` | 查看节点详情 |
| `/up` | 回到父节点 |
| `/home` | 跳回根节点 |
| `/full` | 全览模式：隐藏回答区，节点树全宽（输入框保留；再按一次或切换节点退出） |
| `/save [节点ID]` | 导出节点为 Markdown |
| `/delete <节点ID> [...]` | 删除一个或多个节点及其子节点（需确认；子节点随父节点级联删除，不单独报错） |
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

`deepseek-flash`（DeepSeek-V4.1-Flash）原生支持多模态：在提问时附带图片即可描述图片、识别截图文字、分析图表等。

> 注意：`deepseek-v4-pro` **不支持图片**；带图发送时若当前模型是 Pro，会提示 `/set model flash` 而不会自动切换。

### 用法

```text
/import ~/截图.png "https://example.com/chart.jpg"     # 导入图片（可多个，自动转为待发送图片）
/import ~/report.pdf ~/data.csv                        # 导入文本/文档（发送时自动附带）
/import ~/截图.png ~/说明.txt                          # 图片与文本可混合一次导入
/import clear                                          # 清除全部待导入内容
发送消息                                              # 图片自动附带（需 flash 模型；Pro 会提示切换）
```

- 导入的文件会显示在输入框下方状态栏（`📎 已导入 N 个文件：前2个文件名…`）；鼠标悬停该状态栏会弹出完整文件名列表，移开鼠标自动消失。
- **支持把文件直接拖进终端窗口**：终端会把路径粘贴进输入框（多文件为多个带引号路径），mincli 自动识别并直接导入，无需手动敲 `/import`；输入框无残留路径文本，普通文本粘贴不受影响。若焦点不在输入框（如在对话树上），拖入路径同样会兜底导入。路径解析跨平台：macOS/Linux 的引号与转义空格、Windows 的反斜杠路径（如 `C:\Users\me\a.txt`）与带引号/空格路径均可正确识别。
- 待导入图片发送后绑定到该对话节点；文本/网页内容随下一次发送自动附带（一次生效）。
- 图片消息要求 `deepseek-flash`；当前模型不支持图片（如 `deepseek-v4-pro`）时会提示 `/set model flash`，不会自动切换。
- 聊天区以 `[图片: 文件名 (宽x高)]` 文本占位展示（不依赖终端图像协议，可选中/拷贝）。

### 传图机制（Files API 优先）

1. **上传一次，跨轮复用**：本地图片首次发送时上传到 DeepSeek Files API，得到 `file_id`（`file-api-...`），节点持久化该 ID；后续所有请求（含历史重放、分支追问）都通过 `file` 内容块引用，请求体极小、序列化稳定，**不破坏前缀缓存**。
2. **自动回退**：上传失败时自动改用 base64 内联发送（内联单图 ≤32MiB、请求体 ≤48MiB 预检）；本地图片最大可到 64MiB（Files API 单图上限），超过 32MiB 且上传失败时会明确报错。
3. **文件管理**：`/files list` 查看已上传文件，`/files delete <ID>` 删除；删除对话节点时会尽力清理其关联文件。
4. **外部 URL**：直接传 `http(s)` 链接（≤8192 字符，API 需在 60 秒内下载完成）。

### 图片 token 与成本（固定值估算）

- 官方说明：每张图片进入模型前会自动缩放（总像素 <约 544×544 会放大，更大的会缩小到约 1300×1300 总像素），**单图 token 数存在上限 1024**；官方未公开精确换算公式。
- mincli 因此采用**固定值估算**：默认 `1024 token/图`，可在 `~/.mincli/pricing.json` 的 `image_tokens` 字段随时调整（例如改为 512 更贴近小图）。
- **实测（旧视觉模型）：图片部分不参与前缀缓存**（base64 或 file_id 均按未命中价计费）；文本前缀可正常缓存。新模型如有变化以接口 `usage` 为准。
- 成本估算按未命中价折算（`deepseek-flash` 空闲 1 元 / 高峰 2 元每百万 token）；状态条与对话结束显示均以真实 `usage` 为准。

### 限制（官方）

| 项 | 值 |
|---|---|
| 格式 | JPEG / PNG / GIF / WebP（**按文件内容识别**，不看扩展名） |
| 内联/URL 单图 | ≤32 MiB |
| Files API 单图 | ≤64 MiB（上传文件同样 ≤64 MiB） |
| 请求体 | ≤48 MiB |
| 单请求图片总量 | 不含 `file_id` ≤64 MiB；含 `file_id` ≤200 MiB |
| 图片位置 | 仅 user 消息（system/assistant 带图会 400） |
| 单请求图片数 | ≤600 |
| 图片最大尺寸 | 单边 ≤8192px；单请求 ≥15 张时降为 ≤4096px |
| 外部 URL | ≤8192 字符，需 60 秒内下载完成 |
| Files API | 单文件 ≤64 MiB、默认永久有效（可选 1 小时–30 天）、存储 25 GiB / 1 万个文件 |

---

## MCP 接入

mincli 的工具执行基于标准 [MCP 协议](https://modelcontextprotocol.io/)（Model Context Protocol）：

- **自建 MCP server**：6 个外部工具（文件读写、网页抓取、命令执行）由 mincli 启动的子进程 server 提供，client 通过 stdio 调用。安全/交互策略（用户确认、AI 审核）仍留在客户端，行为与之前一致。**该内置 server 仅应由 mincli 主进程启动使用，请勿将其暴露给其他 MCP 客户端（如 Claude Desktop、Cursor 等）直接连接**——它不独立实现审核/确认/高危命令防护，安全策略完全依赖 mincli 主进程。
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

## 生成中断与「继续」

API 报错（限流、网络中断、`Content Exists Risk` 等内容风控、超时）不会再把这一轮丢掉：

- **节点照常保存**：已流式生成的部分正文与思考过程、已跑完的工具调用与结果、已产生的 token 用量都会写进当前节点，树上该节点标 `⚠ 中断`。
- **正文里给出提示**：中断处显示失败原因，并提示「本轮已保存到当前节点，可直接输入「继续」接着生成」。
- **直接继续**：在该节点下输入「继续」（或任何补充要求）即可接着生成；发给模型的历史里带上已保存的半截回答与工具结果，模型能接着写而不是从头重来。
- **空回答不发回模型**：中断时一个字都没生成的话，该节点不会向模型发送空的 assistant 消息（避免 API 报错），但你的提问仍留在上下文里。
- 切回中断节点时会看到失败原因；确实不需要时可用 `/delete <节点ID>` 删除。

> 若失败原因是内容风控，同一段上下文再次发送仍可能被拒；此时换个说法或另起新节点更有效。

---

## 上下文压缩（/compact）

对话变长后上下文占用大量 token，可用 `/compact` 把**当前分支的全部对话**压缩成一份**详尽**的摘要（由模型生成，按主题组织、完整保留目标/约束/决定/命令/路径/数据/待办等关键信息，宁可长、不要短），并**新建一个摘要节点**：

```
/compact    # 压缩当前分支全部对话，新建摘要节点（不保留任何原文轮次）
```

压缩后的行为：

- **摘要节点**（标题为「上下文压缩摘要」）会直接显示压缩后的信息，并设为当前节点；在摘要节点（或其子节点）上继续输入时，发送给模型的消息 = 摘要 + 新输入，上下文大幅缩短。
- **其他节点不受影响**：切换到摘要节点之前的任意节点，仍发送**完整原始消息**（需要完整历史时直接切换过去即可）。
- 压缩**不删除任何节点**：对话树、历史、导出均不受影响。
- 当前节点已是摘要节点时，再次 `/compact` 会被拦截（需先切换到其他节点，或在摘要节点上继续对话若干轮后再压缩）。
- 摘要与当前节点一起保存到会话文件，重启后仍生效。
- 压缩走当前模型（`/set model` 指定的模型），摘要质量随模型而定。

---

## 工作流（/wf）

做重复性任务时（例如每周生成 changelog、定期巡检仓库、按固定流程发布），不必每次重新向 AI 描述一遍：先把**某一次完整操作**保存成工作流，之后只需一条命令就能按同样流程执行。

**保存（从操作记录提炼）**

```
/wf save changelog            # 提炼当前节点这一次任务（含其间 AI 的工具/命令步骤）
/wf save release a1           # 提炼「a1 → 当前节点」这一连串操作
```

保存时调用当前模型把对话记录提炼成规范文档：**工作内容（目标）、执行步骤、每一步调用命令/工具的细节**；凡是每次执行会变化的实例数据（版本号、日期、路径、本次正文等）一律**不写入**，改写为 `{变量}` 并注明含义（如 `git log {旧版本}..HEAD`）。提炼失败会回退为保存原始记录，可用 `/wf edit` 修正。

**执行**

```
/wf use changelog             # 挂载到下一次输入（状态条显示提示）
（下一条输入：把这次改动的 commit 也写进去）  → 发送即按工作流执行，用后自动解除

/wf run release 旧版本=v1.0 新版本=v2.0   # 立即执行，无需再输入
/wf run release v1.0 v2.0                 # 位置参数按变量顺序填充
/wf stop                      # 取消已挂载的工作流
```

工作流执行与普通发送完全一致（流式输出、工具审核与用户确认依旧生效）；规范中的 `{变量}` 已提供值的按值替换，未提供的由模型结合本次输入与当前环境推断。

**管理（长期保存，重启后仍在）**

```
/wf list                      # 列出工作流（目标/步骤数/变量/运行次数）
/wf show changelog            # 查看完整规范
/wf edit changelog 增加一条校验步骤     # 让模型按你的要求修订
/wf rename changelog cl       # 重命名
/wf delete changelog          # 删除（需确认）
```

- 工作流与对话会话分开，保存于 `~/.mincli/workflows.json`（可用环境变量 `MINCLI_WORKFLOWS_PATH` 改路径）。
- macOS 下 `/wf edit 名`（不带修改要求）会用系统编辑器打开规范文档，保存后自动回写；其他平台请用带“修改要求”的模型修订方式。
- 同名再次 `/wf save` 会覆盖旧版本（需确认）；也可用“先做一遍新操作再 save”来更新工作流。
- `--no-tui` 纯文本模式支持 `/wf list/show/save/use/run/delete/rename/edit`（编辑仅支持模型修订方式）。

---

## 实时用量状态条

输入栏下方有两分栏状态条（仅适配 DeepSeek API），所有数据均来自 API 返回，实时更新：

**左栏：缓存命中率 + 账户余额**
- 缓存命中率 = `usage.prompt_cache_hit_tokens ÷ (prompt_cache_hit_tokens + prompt_cache_miss_tokens)`，取当前节点最近一次请求的累计值（DeepSeek 上下文缓存自动生效，命中部分按缓存命中价计费）
- 账户余额来自 `GET /user/balance` 的 `total_balance`（优先 CNY），每 60 秒自动刷新

**右栏：下次输入估算**
- 下次输入 token：普通节点 = 上次完整输入 + 本节点输出（DeepSeek API 真实 `usage` 口径，与对话结束显示的输入/输出直接对应，随对话推进持续更新）；**摘要节点**（无 API 用量）按实际发送的摘要上下文实时估算，与 `/compact` 报告的 after_tokens 一致；用户新输入内容量小，忽略不计
- 预计价格 = token 量 × 折算单价 ÷ 100 万，按当前时段（北京时间**周一至周五** 9-12、14-18 点为高峰，其余为空闲）与缓存命中率折算（命中部分按缓存命中价、其余按未命中价）

> 口径说明：对话结束显示的「输入/输出 tokens」来自 DeepSeek API 真实 `usage`；摘要节点的估算用 tiktoken 本地计算同一份消息（摘要节点本身没有可用的新 API 请求），两者因 tokenizer 差异数字不等，但摘要节点状态条与压缩报告严格一致。

### 定价配置（`~/.mincli/pricing.json`）

DeepSeek 价格会频繁调整，且 2026-09-10 起 Flash 已降价、并改为**仅工作日高峰**。为避免每次改价都要改代码，价格、峰谷时段与图片 token 估算都可通过配置文件覆盖（路径可用环境变量 `MINCLI_PRICING_PATH` 修改；文件不存在或字段非法时静默回退内置默认）：

```json
{
  "peak": {
    "days": [1, 2, 3, 4, 5],
    "ranges": [[9, 12], [14, 18]],
    "timezone_offset_hours": 8
  },
  "models": {
    "deepseek-flash": { "hit": [0.02, 0.04], "miss": [1.0, 2.0], "output": [4.0, 8.0] },
    "deepseek-v4-pro": { "miss": 4.5, "output": 13.5 }
  },
  "image_tokens": 1024
}
```

- 单位为「元 / 百万 tokens」；每个价格字段支持 `[空闲价, 高峰价]` 或单个数字（表示不分峰谷）。
- `models` 按模型、按字段合并：只写要改的模型/字段，其余沿用内置默认（内置默认即 2026-09-10 官方价）。
- `peak.days` 用 ISO 星期（周一=1 … 周日=7）；`ranges` 为 `[起始小时, 结束小时)`，可多段；`timezone_offset_hours` 默认 8（北京时间）。
- `image_tokens` 为每张图片的固定估算 token（官方单图上限 1024）。
- 可用 `mincli info` 查看当前生效的定价配置文件与图片 token 值。

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
│   ├── workflows.py         # 工作流（/wf）数据模型与持久化存储
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
A：图片理解仅 `deepseek-flash` 支持（`deepseek-v4-pro` 不支持）。若当前模型不是 flash，发送图片时会提示 `/set model flash`，不会自动切换模型。

**Q：上传图片失败？**  
A：会自动回退为 base64 内联发送（仅影响请求体大小）；可通过 `/files list` 查看已上传文件。单文件超过 64 MiB、内联超过 32 MiB 且上传失败、或格式不支持（仅 JPEG/PNG/GIF/WebP，按内容识别）会报错提示。

**Q：图片 token 消耗大吗？**  
A：实测每图约 117~357 token（尺寸相关，`/set detail low` 可省约 60%）；图片部分不参与前缀缓存（按未命中价计费），文本前缀可正常缓存。具体以接口返回 `usage` 为准。

---

## 许可

MIT License
