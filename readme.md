> [🇨🇳 中文](readme.zh.md)

# mincli

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A feature-rich tree-structured CLI AI assistant powered by DeepSeek V4 models.  
Features streaming real-time output, Markdown rendering, branching conversations, and full reasoning chain display.  
You can dynamically switch models, system prompts, temperature, and thinking mode on the fly.  
AI can autonomously invoke tools: read files (txt/md/csv/pdf/docx), fetch web pages, list directories, and search the web.  
Manual commands (`/imp`, `/fetch`) are also available for importing content.

---

## Features

- 🚀 **Streaming Output** — Real-time Markdown rendering, word-by-word response display.
- 🌲 **Tree Conversations** — Main line + branch nodes with globally unique IDs; jump between any nodes freely.
- 🧠 **Thinking Mode** — Supports V4 model reasoning chains (reasoning_content), displayed in dim text, toggleable.
- 🔧 **Tool Calling** — AI autonomously invokes tools to read files, fetch web pages, list directories, and search the web.
- 💾 **Auto-Save Session** — Session is saved on exit and restored on next launch.
- 📄 **Save as Markdown** — `/save` exports a node's conversation as a `.md` file.
- 📁 **File Import** — Use `/imp` or let AI read files; supports txt/md/csv/pdf/docx.
- 🌐 **Web Fetching** — Use `/fetch` or let AI fetch; auto-extracts body text with encoding detection.
- ⚙️ **Dynamic Configuration** — `/set` command to adjust system prompt, temperature, model, thinking toggle, and reasoning effort mid-conversation.
- 🧩 **Model Selection** — Supports `deepseek-v4-flash` (lightweight, fast) and `deepseek-v4-pro` (flagship performance).

---

## Recommended Terminal

For the best experience, **we strongly recommend [iTerm2](https://iterm2.com/) on macOS**.  
mincli has an exclusive optimization for iTerm2: clearing the screen also resets the scrollback buffer, keeping the interface clean.  
Other terminals (Windows Terminal, Linux terminals) work fine as well — only the clear-screen behavior differs (no scrollback reset).

---

## Installation

### Prerequisites
- Python 3.8 or later

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd mincli
```

### 2. Install

Create a virtual environment (recommended but optional):
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the mincli package (editable mode for development):
```bash
pip install -e .
```

Or install normally:
```bash
pip install .
```

### 3. Configure API Keys

mincli loads configuration from three locations (priority: high → low):

| Priority | Location | Description |
|----------|----------|-------------|
| 1 (highest) | Environment variables | Already set in your shell |
| 2 | `~/.mincli/.env` | User-level config file |
| 3 (lowest) | `.env` (current directory) | Project-level config file |

Copy the template and fill in your API key:
```bash
cp .env.example .env
# Edit .env with DEEPSEEK_API_KEY
```

Optionally set a custom save path and Bocha Search API key (required for the `web_search` tool):
```
MINCLI_SAVE_PATH=~/Documents/MyChats
BOCHA_API_KEY=your_bocha_api_key_here
```

### 4. Launch

After installation, run directly:
```bash
mincli chat              # Start tree conversation mode
```
Or use Python module:
```bash
python -m mincli chat    # Same as above
python main.py chat      # Legacy compatibility
```

---

## Building from Source (macOS)

You can build a standalone executable and package it as a `.dmg` for distribution.

### Prerequisites
- Python 3.8 or later
- [Homebrew](https://brew.sh) (for `create-dmg`)

### 1. Install Build Dependencies
```bash
pip install pyinstaller
```

### 2. Build the Executable
```bash
pyinstaller mincli.spec
```

The executable will be at `dist/mincli`.

### 3. Create DMG
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

Or using the built-in `hdiutil`:
```bash
mkdir dmg && cp dist/mincli dmg/ && ln -s /usr/local/bin dmg/
hdiutil create -volname mincli -srcfolder dmg -ov -format UDZO mincli.dmg
rm -rf dmg
```

The `.dmg` file is now ready for distribution — users mount it and drag `mincli` into `/usr/local/bin`.

---

## Quick Start

### Start a Conversation
```bash
python main.py chat
```

### Enable Thinking Mode
```bash
python main.py chat --thinking
```

### Select Model and Set Reasoning Effort
```bash
python main.py chat --model pro --thinking --effort max
```

### Import a File and Ask
```bash
# After starting, type at the prompt:
# /imp ~/document.pdf
# Then ask your question — the file content is automatically attached
```

### AI Autonomous Tool Invocation
Ask naturally, and AI will decide whether to use tools:
- "Show me the contents of config.json"
- "Fetch https://example.com for me"
- "What files are in the current directory?"
- "Search for recent AI news" (requires `/search` authorization first)

### View All Options
```bash
python main.py chat --help
```

---

## Configuration

mincli loads `.env` files from multiple locations (priority: environment variables > `~/.mincli/.env` > local `.env`).  
Supported variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | None | DeepSeek API key |
| `BOCHA_API_KEY` | No | None | Bocha Search API key (enables the web_search tool) |
| `MINCLI_SAVE_PATH` | No | `~/Documents/mincli_Conversations` | Directory for saving conversation Markdown files |

Startup arguments can be passed on the command line:
- `-m` / `--model`: Model selection (`flash` or `pro`, default `flash`)
- `--thinking`: Enable thinking mode (default off)
- `--effort`: Reasoning effort (`high` or `max`, default `high`)
- `--temp`: Temperature (default 1.0)

---

## Interactive Commands

Enter these commands (starting with `/`) during conversation:

| Command | Description |
|---------|-------------|
| `/exit`, `/quit` | Exit the program (session auto-saved) |
| `/clear`, `/c` | Clear current session history |
| `/set system <text>` | Change system prompt |
| `/set temp <value>` | Change temperature (0.0–2.0) |
| `/set model <flash\|pro>` | Switch model |
| `/set thinking <on\|off>` | Toggle thinking mode |
| `/set effort <high\|max>` | Set reasoning effort (takes effect when thinking is on) |
| `/set show` | Display current configuration |
| `/search <count>` | **Authorize AI to search the web** — grants N web_search quota (requires BOCHA_API_KEY) |
| `/imp <path>` | **Import file content** (txt/md/py/bat/sh/csv/pdf/docx), auto-attached on next question |
| `/fetch <URL>` | **Fetch web page content**, auto-extract body, auto-attached on next question |
| `/cd <node-id>` | Jump to a specific node (e.g., `a1`, `b1`) |
| `/list` | List all nodes |
| `/info [node-id]` | Show node details |
| `/back` | Go back to parent node |
| `/root` | Jump to root node (`main`) |
| `/save [node-id]` | Save current or specified node as Markdown |
| `/rm <node-id>` | Delete a node and all its children (root cannot be deleted) |

---

## AI Tool Reference

AI can autonomously invoke the following tools during conversation:

| Tool | Method | Description | Parameters |
|------|--------|-------------|------------|
| `read_file` | `InteractiveSession._parse_file()` | Read local file content | `filepath` (required) |
| `fetch_webpage` | `InteractiveSession._fetch_webpage()` | Fetch and extract web page text | `url` (required) |
| `list_directory` | `InteractiveSession._list_directory()` | List directory contents | `directory` (required); `show_hidden` (optional) |
| `write_file` | `InteractiveSession._write_file()` | Create or overwrite a file (user confirmation required) | `filepath` (required); `content` (required) |
| `edit_file` | `InteractiveSession._edit_file()` | Search and replace in a file (user confirmation required) | `filepath` (required); `old_string` (required); `new_string` (required) |
| `web_search` | `InteractiveSession._web_search()` | Search the web (requires `/search` authorization) | `query` (required); `freshness` (optional); `count` (optional) |
| `execute_command` | `InteractiveSession._execute_command()` | Execute a shell command (each command is AI-audited and requires user confirmation; timeout is mandatory; partial output returned on timeout) | `command` (required); `timeout` (required, seconds) |

---

## Project Structure

```
.
├── main.py                 # Main program
├── setup.sh                # macOS/Linux setup script
├── setup.bat               # Windows setup script
├── setup.zh.sh             # macOS/Linux setup script (Chinese)
├── setup.zh.bat            # Windows setup script (Chinese)
├── .env                    # API keys & config (create your own)
├── .gitignore
├── requirements.txt        # pip dependency list
├── mincli.spec             # PyInstaller build configuration
├── readme.md               # English documentation (default)
├── readme.zh.md            # Chinese documentation
└── README.md
```

---

## FAQ

**Q: Why does it show "⚠️会话文件损坏" or fail to load the previous session?**  
A: The session file may be incompatible after a version update. Delete `~/.mincli_session.json` and restart.

**Q: Thinking mode is on but I don't see the reasoning process?**  
A: Only DeepSeek V4 models output `reasoning_content` when thinking is enabled. Make sure you're using `flash` or `pro` with `--thinking` enabled.

**Q: Title generation fails and shows "对话_XXXXXXXX"?**  
A: This used to happen when the title generation model had thinking enabled by default. The latest code explicitly disables thinking for title generation — update to the latest version.

**Q: `/imp` fails to import PDF or DOCX with "need to install xxx"?**  
A: Make sure `pdfminer.six` and `python-docx` are installed, or re-run `setup.sh`/`setup.bat`.

**Q: Screen clearing is imperfect on Windows?**  
A: mincli has optimal clear-screen behavior (with scrollback reset) on iTerm2 (macOS). Windows uses `cls` for clearing, which works fine but doesn't reset scrollback.

---

## License

MIT License
