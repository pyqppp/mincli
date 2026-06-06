> [🇬🇧 English](readme.md)

# mincli

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个功能丰富的树状对话命令行 AI 助手，基于 DeepSeek V4 模型。  
支持流式实时输出、Markdown 渲染、树状对话分支，以及完整的推理过程（思考链）显示。  
你可以随时动态切换模型、系统提示词、温度和思考模式。  
AI 可自主调用工具：读取文件（txt/md/csv/pdf/docx）、抓取网页、列出目录、搜索互联网。  
也支持手动命令导入文件（`/imp`）和抓取网页（`/fetch`）。

---

## 特性

- 🚀 **流式输出**：实时刷新 Markdown 渲染，回答逐字呈现。
- 🌲 **树状对话**：主线＋分支节点，全局唯一 ID，可在任意节点间跳转。
- 🧠 **思考模式**：支持 V4 模型的推理链（reasoning），灰色显示思考过程，可开关。
- 🔧 **工具调用**：AI 可自主调用工具——读取文件、抓取网页、列出目录、搜索互联网，无需手动操作。
- 💾 **会话自动保存**：退出时自动保存，下次启动恢复。
- 📄 **保存为 Markdown**：`/save` 将节点对话保存为 `.md` 文件。
- 📁 **文件导入**：`/imp` 命令或 AI 自主读取，支持 txt/md/csv/pdf/docx 格式。
- 🌐 **网页抓取**：`/fetch` 命令或 AI 自主抓取，自动提取正文并识别编码。
- ⚙️ **动态调整**：`/set` 命令可在对话中随时修改系统提示词、温度、模型、思考开关和推理强度。
- 🧩 **模型选择**：支持 `deepseek-v4-flash`（轻量快速）和 `deepseek-v4-pro`（旗舰性能）。

---

## 推荐终端

为了获得最佳体验，**强烈建议使用 [iTerm2](https://iterm2.com/) (macOS)**。  
本工具在 iTerm2 下拥有独家优化：清屏时会同时重置滚动缓冲区，让界面始终整洁。  
其他终端（包括 Windows Terminal、Linux 终端）也能正常使用，仅清屏方式不同（无滚动缓冲区重置）。

---

## 安装

### 前置要求
- Python 3.8 或更高版本

### 1. 克隆仓库
```bash
git clone <你的仓库地址>
cd mincli
```

### 2. 安装

推荐创建虚拟环境后安装（可选，但建议）：
```bash
python3 -m venv venv
source venv/bin/activate
```

安装 mincli 包（开发模式，修改源码即时生效）：
```bash
pip install -e .
```

也可直接安装（不编辑源码时）：
```bash
pip install .
```

### 3. 配置 API 密钥

mincli 从多个位置加载配置（优先级从高到低）：

| 优先级 | 位置 | 说明 |
|--------|------|------|
| 1（最高） | 环境变量 | 已在 shell 中设置 |
| 2 | `~/.mincli/.env` | 用户级配置文件 |
| 3（最低） | `.env`（当前目录） | 项目级配置文件 |

复制模板文件并填入你的 API Key：
```bash
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
```

也可以自定义会话保存路径（可选）：
```
MINCLI_SAVE_PATH=~/Documents/MyChats
```

### 4. 启动
安装后可直接运行：
```bash
mincli chat    # 启动树状对话
```
或使用 Python 模块方式：
```bash
python -m mincli chat    # 同上
python main.py chat      # 兼容方式
```

---

## 从源码构建（macOS）

你可以将 mincli 打包为独立可执行文件并制作 `.dmg` 安装包。

### 前置要求
- Python 3.8 或更高版本
- [Homebrew](https://brew.sh)（用于安装 `create-dmg`）

### 1. 安装构建依赖
```bash
pip install pyinstaller
```

### 2. 构建可执行文件
```bash
pyinstaller mincli.spec
```

生成的可执行文件位于 `dist/mincli`。

### 3. 制作 DMG 安装包
```bash
brew install create-dmg
create-dmg \
  --volname "mincli" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "mincli" 175 200 \
  --hide-extension "mincli" \
  --app-drop-link 425 200 \
  "mincli.dmg" \
  "dist/"
```

或使用 macOS 自带的 `hdiutil`：
```bash
mkdir dmg && cp dist/mincli dmg/ && ln -s /usr/local/bin dmg/
hdiutil create -volname mincli -srcfolder dmg -ov -format UDZO mincli.dmg
rm -rf dmg
```

用户挂载 `.dmg` 后将 `mincli` 拖入 `/usr/local/bin` 即可全局使用。

---

## 快速开始

### 启动对话
```bash
python main.py chat
```

### 启用思考模式
```bash
python main.py chat --thinking
```

### 选择模型并指定推理强度
```bash
python main.py chat --model pro --thinking --effort max
```

### 导入文件并提问
```bash
# 运行后，在交互提示下输入：
# /imp ~/document.pdf
# 然后直接提问即可自动附加上 PDF 内容
```

### AI 自主调用工具
对话中正常提问即可，AI 会自动判断是否需要调用工具：
- "帮我看看 config.json 的内容"
- "查一下 https://example.com 这个网页"
- "当前目录下有哪些文件？"
- "帮我搜索一下最近的 AI 新闻"（需先通过 `/search` 授权）

### 查看所有选项
```bash
python main.py chat --help
```

---

## 配置说明

mincli 从多个位置加载 `.env` 文件（优先级：环境变量 > `~/.mincli/.env` > 本地 `.env`）。  
支持以下变量：

| 变量名 | 必需 | 默认值 | 描述 |
|--------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | 无 | DeepSeek API 密钥 |
| `BOCHA_API_KEY` | 否 | 无 | 博查搜索 API 密钥（启用 web_search 工具需要） |
| `MINCLI_SAVE_PATH` | 否 | `~/Documents/mincli_Conversations` | 对话 Markdown 文件保存目录 |

启动参数可在命令行直接指定，例如：
- `-m` / `--model`：模型选择（`flash` 或 `pro`，默认 `flash`）
- `--thinking`：开启思考模式（默认关闭）
- `--effort`：推理强度（`high` 或 `max`，默认 `high`）
- `--temp`：温度参数（默认 1.0）

---

## 交互命令

你可以在对话中直接输入以下命令（以 `/` 开头）：

| 命令 | 说明 |
|------|------|
| `/exit`, `/quit` | 退出程序（自动保存会话） |
| `/clear`, `/c` | 清空当前会话历史 |
| `/set system <内容>` | 修改系统提示词 |
| `/set temp <数值>` | 修改温度（0.0~2.0） |
| `/set model <flash\|pro>` | 切换模型 |
| `/set thinking <on\|off>` | 开启/关闭思考模式 |
| `/set effort <high\|max>` | 设置推理强度（开启思考后生效） |
| `/set show` | 显示当前配置 |
| `/search <次数>` | **授权 AI 搜索互联网**，为 AI 分配 N 次 web_search 调用配额（需配置 BOCHA_API_KEY） |
| `/imp <文件路径>` | **导入文件内容**（支持 .txt/.md/.py/.bat/.sh/.csv/.pdf/.docx），下次提问自动以"文件名：内容"格式附加 |
| `/fetch <URL>` | **抓取网页内容**，自动提取正文，下次提问自动附加 |
| `/cd <节点ID>` | 跳转到指定节点（如 `a1`, `b1`） |
| `/list` | 列出所有节点 |
| `/info [节点ID]` | 查看节点详细信息 |
| `/back` | 返回父节点 |
| `/root` | 跳到根节点（`main`） |
| `/save [节点ID]` | 保存当前或指定节点为 Markdown |
| `/rm <节点ID>` | 删除节点及其所有子节点（根节点不可删除） |

---

## AI 工具参考

AI 在对话中可根据需要自主调用以下工具：

| 工具名 | 对应方法 | 功能 | 参数 |
|--------|---------|------|------|
| `read_file` | `InteractiveSession._parse_file()` | 读取本地文件内容 | `filepath`（必填）：文件路径 |
| `fetch_webpage` | `InteractiveSession._fetch_webpage()` | 抓取网页并提取正文 | `url`（必填）：网页 URL |
| `list_directory` | `InteractiveSession._list_directory()` | 列出目录内容 | `directory`（必填）：目录路径；`show_hidden`（可选）：是否包含隐藏文件 |
| `write_file` | `InteractiveSession._write_file()` | 创建新文件或覆盖写入（需用户确认） | `filepath`（必填）：文件路径；`content`（必填）：写入内容 |
| `edit_file` | `InteractiveSession._edit_file()` | 搜索替换文件内容（需用户确认） | `filepath`（必填）；`old_string`（必填）：被替换的原文；`new_string`（必填）：替换后的新内容 |
| `web_search` | `InteractiveSession._web_search()` | 搜索互联网信息（需用户通过 `/search` 授权） | `query`（必填）：搜索词；`freshness`（可选）：时间范围；`count`（可选）：返回条数 |
| `execute_command` | `InteractiveSession._execute_command()` | 执行 shell 命令（每次执行前会经 AI 安全审核和用户确认；必须设置超时；超时后返回已产生的部分输出） | `command`（必填）：要执行的命令；`timeout`（必填）：超时秒数 |

---

## 项目结构

```
.
├── main.py                 # 主程序
├── setup.sh                # macOS/Linux 安装脚本
├── setup.bat               # Windows 安装脚本
├── setup.zh.sh             # macOS/Linux 安装脚本（中文）
├── setup.zh.bat            # Windows 安装脚本（中文）
├── .env                    # API 密钥等配置（需自行创建）
├── .gitignore
├── requirements.txt        # pip 依赖清单
├── mincli.spec             # PyInstaller 构建配置
├── readme.md               # English documentation (default)
├── readme.zh.md            # 中文文档
└── README.md
```

---

## 常见问题

**Q：为什么启动后提示“⚠️ 会话文件损坏”或无法加载上次会话？**  
A：会话文件可能因版本更新不兼容。可以手动删除 `~/.mincli_session.json` 然后重新启动。

**Q：思考模式开启了但看不到思考过程？**  
A：只有 DeepSeek V4 模型在开启思考后才会输出 `reasoning_content`，请确保使用 `flash` 或 `pro` 模型，并且 `--thinking` 已启用。

**Q：生成标题失败，显示“对话_XXXXXXXX”？**  
A：通常是标题生成时模型默认开启了思考导致输出异常，现在已在生成标题时显式关闭思考，更新到最新代码即可。

**Q：使用 `/imp` 导入 PDF 或 DOCX 时报错“需安装 xxx”？**  
A：请确保已通过 `pip install pdfminer.six python-docx` 安装相应依赖，或重新运行 `setup.sh`/`setup.bat`。

**Q：在 Windows 下清屏不完美？**  
A：本工具在 iTerm2 (macOS) 下拥有最佳清屏体验（含滚动缓冲区重置），Windows 下使用 `cls` 清屏，不影响使用。

---

## 许可

MIT License
