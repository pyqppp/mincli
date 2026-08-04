> [🇬🇧 English](readme.md)

# mincli

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 DeepSeek V4 模型的树状对话 CLI AI 助手。  
流式输出、Markdown 渲染、树状对话分支、完整推理链显示。  
支持动态切换模型/提示词/温度/思考模式，AI 可自主调用文件读写、网页抓取、目录浏览、互联网搜索、Shell 执行等工具。

---

## 特性

- 🚀 **流式输出** — 实时 Markdown 渲染，回答逐字呈现
- 🌲 **树状对话** — 主线＋分支节点，全局唯一 ID，任意节点间自由跳转
- 🧠 **思考模式** — 支持 DeepSeek V4 推理链，可随时开关
- 🔧 **工具调用** — AI 自主调用 7 种工具：读文件、写文件、编辑文件、抓网页、列目录、搜网络、执行命令
- 💾 **会话自动保存** — 退出自动保存，下次启动恢复
- 📄 **导出 Markdown** — `/save` 将节点对话导出为 `.md` 文件
- ⚙️ **动态配置** — `/set` 命令随时修改系统提示词、温度、模型、思考开关、推理强度
- 🧩 **双模型** — `deepseek-v4-flash`（轻量快速）和 `deepseek-v4-pro`（旗舰性能）

---

## 推荐终端

**macOS** 建议使用 [iTerm2](https://iterm2.com/)，清屏时自动重置滚动缓冲区。  
其他终端（Windows Terminal、Linux 终端）功能完全正常，仅清屏时无滚动缓冲区重置。

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
# 方式一：直接运行（推荐）
mincli chat

# 方式二：Python 模块
python -m mincli chat

# 方式三：传统方式（兼容）
python main.py chat
```

---

## 快速开始

```bash
# 基本对话
mincli chat

# 开启思考模式
mincli chat --thinking

# 指定模型 + 推理强度
mincli chat --model pro --thinking --effort max

# 查看所有选项
mincli chat --help
```

### 对话中常用操作
```
# 导入文件或网页后提问
/import ~/document.pdf
这个文档讲了什么？

# AI 自主工具调用（直接提问即可）
帮我看看 config.json 的内容
查一下 https://example.com
当前目录下有哪些文件？
帮我搜索最近的 AI 新闻
```

---

## 配置

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | 无 | DeepSeek API 密钥 |
| `BOCHA_API_KEY` | 否 | 无 | 博查搜索密钥（启用 web_search） |
| `MINCLI_SAVE_PATH` | 否 | `~/Documents/mincli_Conversations` | 对话导出目录 |

命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m` / `--model` | `flash` | 模型：`flash` \| `pro` |
| `--thinking` | 关 | 开启思考模式 |
| `--effort` | `high` | 推理强度：`high` \| `max` |
| `--temp` | `1.0` | 温度参数 |

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
| `/set effort <high\|max>` | 推理强度 |
| `/set show` | 显示当前配置 |
| `/search <次数>` | 授权 AI 搜索网页（需 BOCHA_API_KEY） |
| `/import <路径或URL>` | 导入文件（txt/md/py/csv/pdf/docx）或抓取网页 |
| `/<节点ID>`（如 `/a3`） | 直接跳转到指定节点 |
| `/tree` | 列出所有节点 |
| `/info [节点ID]` | 查看节点详情 |
| `/up` | 回到父节点 |
| `/home` | 跳回根节点 |
| `/save [节点ID]` | 导出节点为 Markdown |
| `/delete <节点ID>` | 删除节点及其子节点 |
| `/view` | 用编辑器打开当前回答 |

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
| `web_search` | 联网搜索（需 `/search` 授权） | `query`; `freshness`（可选）; `count`（可选） |
| `execute_command` | 执行 Shell 命令（AI 审核 + 用户确认） | `command`; `timeout` |

---

## 项目结构

```
.
├── main.py                  # 入口（兼容 python main.py）
├── pyproject.toml           # 项目元数据 + 依赖声明
├── mincli.spec              # PyInstaller 构建配置
├── .env.example             # 配置模板
├── readme.md                # 英文文档
├── readme.zh.md             # 中文文档
│
├── mincli/                  # 核心包（14 个模块）
│   ├── __init__.py          # 版本号
│   ├── __main__.py          # python -m mincli 入口
│   ├── cli.py               # Typer CLI 命令
│   ├── config.py            # 常量 + 配置加载
│   ├── models.py            # ConversationNode/Tree、StreamResult
│   ├── helpers.py           # 工具函数（余额/标题/token估算）
│   ├── render.py            # Rich 主题 + console
│   ├── streaming.py         # 流式 API 交互 + 实时渲染
│   ├── session.py           # InteractiveSession 主循环
│   └── tools/
│       ├── registry.py      # 工具定义列表
│       ├── execute.py       # 命令执行 + AI 安全审计
│       ├── file_ops.py      # 文件读写/解析
│       ├── web_fetch.py     # 网页抓取 + 搜索
│       └── thinking.py      # 审计系统提示词
│
└── venv/                    # 虚拟环境（未追踪）
```

---

## 从源码构建（macOS）

构建独立可执行文件并打包 `.dmg`：

```bash
# 1. 构建可执行文件
pip install pyinstaller
pyinstaller mincli.spec
# dist/mincli

# 2. 制作 DMG（需 Homebrew）
brew install create-dmg
create-dmg \
  --volname "mincli" \
  --window-pos 200 120 --window-size 600 400 \
  --icon-size 100 --icon "mincli" 175 200 \
  --hide-extension "mincli" --app-drop-link 425 200 \
  "mincli.dmg" "dist/"
```

用户挂载 `.dmg` 后将 `mincli` 拖入 `/usr/local/bin` 即可全局使用。

---

## 常见问题

**Q：启动后提示"会话文件损坏"？**  
A：删除 `~/.mincli_session.json` 后重启。

**Q：思考模式开启但看不到推理过程？**  
A：请确认使用 `flash` 或 `pro` 模型并已开启 `--thinking`。

**Q：`/import` 导入 PDF/DOCX 报错？**  
A：确保依赖已安装：`pip install pdfminer.six python-docx`。

---

## 许可

MIT License
