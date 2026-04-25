# DeepSeek CLI

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个功能丰富的交互式命令行 AI 助手，基于 DeepSeek V4 模型。  
支持流式实时输出、Markdown 渲染、历史回溯、树状对话分支，以及完整的推理过程（思考链）显示。  
你可以随时动态切换模型、系统提示词、温度和思考模式。

---

## 特性

- 🚀 **流式输出**：实时刷新 Markdown 渲染，回答逐字呈现。
- 🌲 **树状对话**：主线＋分支节点，全局唯一 ID，可在任意节点间跳转。
- 🧠 **思考模式**：支持 V4 模型的推理链（reasoning），灰色显示思考过程，可开关。
- 📜 **历史浏览**：线性模式下按 ↑/↓ 回溯历史对话，ESC 回到最新。
- 💾 **会话自动保存**：退出时自动保存，下次启动恢复（支持跨模式恢复）。
- 📄 **保存为 Markdown**：`/save` 或 `/save_node` 将单条对话保存为 `.md` 文件。
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
cd deepseek-cli
```

### 2. 安装依赖

#### macOS / Linux
```bash
chmod +x setup.sh
./setup.sh
```

#### Windows
双击运行 `setup.bat` 或在命令行中执行：
```cmd
setup.bat
```

该脚本会自动创建虚拟环境并安装所有依赖。  
如果无法连接外网，可手动执行：
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install tiktoken typer python-dotenv openai rich prompt-toolkit
```

### 3. 配置 API 密钥
在项目根目录创建 `.env` 文件，写入你的 DeepSeek API Key：
```
DEEPSEEK_API_KEY=your_api_key_here
```
也可以自定义会话保存路径（可选）：
```
DEEPSEEK_SAVE_PATH=~/Documents/MyChats
```

### 4. 启动
激活虚拟环境后运行：
```bash
python main.py chat -i    # 普通交互模式
python main.py chat -t    # 树状对话模式
```

---

## 快速开始

### 普通交互模式（线性）
```bash
python main.py chat -i
```

### 树状对话模式
```bash
python main.py chat -t
```

### 启用思考模式
```bash
python main.py chat -i --thinking
```

### 选择模型并指定推理强度
```bash
python main.py chat -t --model pro --thinking --effort max
```

### 查看所有选项
```bash
python main.py chat --help
```

---

## 配置说明

主配置文件为 `.env`（在项目根目录），支持以下变量：

| 变量名 | 必需 | 默认值 | 描述 |
|--------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | 无 | DeepSeek API 密钥 |
| `DEEPSEEK_SAVE_PATH` | 否 | `~/Documents/DeepSeek_Conversations` | 对话 Markdown 文件保存目录 |

启动参数可在命令行直接指定，例如：
- `-i` / `--interactive`：线性交互模式
- `-t` / `--tree`：树状对话模式
- `-m` / `--model`：模型选择（`flash` 或 `pro`，默认 `flash`）
- `--thinking`：开启思考模式（默认关闭）
- `--effort`：推理强度（`high` 或 `max`，默认 `high`）
- `--temp`：温度参数（默认 1.0）

---

## 交互命令

无论是在线性还是树状模式下，你都可以直接在对话中输入以下命令（以 `/` 开头）：

### 通用命令
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
| `/save <序号>` | 保存线性对话中的第 N 条对话为 Markdown |
| `/tree` | 从线性模式切换到树状模式（转换历史） |

### 树状模式专属命令
| 命令 | 说明 |
|------|------|
| `/cd <节点ID>` | 跳转到指定节点（如 `a1`, `b1`） |
| `/list` | 列出所有节点 |
| `/info [节点ID]` | 查看节点详细信息 |
| `/back` | 返回父节点 |
| `/root` | 跳到根节点（`main`） |
| `/save_node [节点ID]` | 保存当前或指定节点为 Markdown |

**线性模式额外功能**：在输入框为空时按 **↑** / **↓** 可以浏览历史对话，**ESC** 回到最新。

---

## 项目结构

```
.
├── main.py                 # 主程序
├── setup.sh                # macOS/Linux 安装脚本
├── setup.bat               # Windows 安装脚本
├── .env                    # API 密钥等配置（需自行创建）
└── README.md
```

---

## 常见问题

**Q：为什么启动后提示“⚠️ 会话文件损坏”或无法加载上次会话？**  
A：会话文件可能因版本更新不兼容。可以手动删除 `~/.deepseek_cli_linear_session.json` 或 `~/.deepseek_cli_tree_session.json` 然后重新启动。

**Q：思考模式开启了但看不到思考过程？**  
A：只有 DeepSeek V4 模型在开启思考后才会输出 `reasoning_content`，请确保使用 `flash` 或 `pro` 模型，并且 `--thinking` 已启用。

**Q：生成标题失败，显示“对话_XXXXXXXX”？**  
A：通常是标题生成时模型默认开启了思考导致输出异常，现在已在生成标题时显式关闭思考，更新到最新代码即可。

**Q：在 Windows 下清屏不完美？**  
A：本工具在 iTerm2 (macOS) 下拥有最佳清屏体验（含滚动缓冲区重置），Windows 下使用 `cls` 清屏，不影响使用。

---

## 许可

MIT License
```