> [🇨🇳 中文](readme.zh.md)

# mincli

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tree-structured CLI AI assistant powered by DeepSeek V4 models.  
Streaming output, Markdown rendering, branching conversations, full reasoning chain display.  
Dynamically switch models, system prompts, temperature, and thinking mode.  
AI can autonomously invoke tools: read/write/edit files, fetch web pages, list directories, search the web, and execute shell commands.

---

## Features

- 🚀 **Streaming Output** — Real-time Markdown rendering, word-by-word display
- 🌲 **Tree Conversations** — Main line + branch nodes with globally unique IDs; jump freely between nodes
- 🧠 **Thinking Mode** — Full V4 reasoning chain display, toggleable on the fly
- 🔧 **Tool Calling** — AI autonomously invokes 7 tools: read/write/edit files, fetch web pages, list directories, search the web, execute commands
- 💾 **Auto-Save Session** — Saved on exit, restored on next launch
- 📄 **Export as Markdown** — `/save` exports any node as `.md`
- ⚙️ **Dynamic Config** — `/set` changes system prompt, temperature, model, thinking, reasoning effort mid-conversation
- 🧩 **Dual Model** — `deepseek-v4-flash` (fast) and `deepseek-v4-pro` (flagship)

---

## Recommended Terminal

**macOS**: [iTerm2](https://iterm2.com/) recommended — clear screen also resets scrollback buffer.  
Other terminals (Windows Terminal, Linux) work fully, just without scrollback reset on clear.

---

## Installation

### Prerequisites
- Python 3.10+

### 1. Clone
```bash
git clone <repo-url>
cd mincli
```

### 2. Install (venv recommended)
```bash
python3 -m venv venv
source venv/bin/activate

# Editable mode (code changes take effect immediately)
pip install -e .
```

Use a mirror to speed up dependency downloads (China):
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env, set DEEPSEEK_API_KEY
```

Config load order (high → low):

| Priority | Source |
|----------|--------|
| 1 | Shell environment variables |
| 2 | `~/.mincli/.env` |
| 3 | Local `.env` |

### 4. Launch
```bash
# Recommended
mincli chat

# Python module
python -m mincli chat

# Legacy compat
python main.py chat
```

---

## Quick Start

```bash
# Basic conversation
mincli chat

# Enable thinking mode
mincli chat --thinking

# Select model + reasoning effort
mincli chat --model pro --thinking --effort max

# View all options
mincli chat --help
```

### In-conversation examples
```
/imp ~/document.pdf
What does this document say?

# Or just ask — AI calls tools autonomously
Show me config.json
Fetch https://example.com
What files are here?
Search for recent AI news
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | — | DeepSeek API key |
| `BOCHA_API_KEY` | No | — | Bocha Search key (enables web_search) |
| `MINCLI_SAVE_PATH` | No | `~/Documents/mincli_Conversations` | Export directory |

CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--model` | `flash` | Model: `flash` \| `pro` |
| `--thinking` | off | Enable thinking mode |
| `--effort` | `high` | Reasoning effort: `high` \| `max` |
| `--temp` | `1.0` | Temperature |

---

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/exit`, `/quit` | Exit (session auto-saved) |
| `/clear`, `/c` | Clear session |
| `/set system <text>` | Change system prompt |
| `/set temp <value>` | Change temperature |
| `/set model <flash\|pro>` | Switch model |
| `/set thinking <on\|off>` | Toggle thinking |
| `/set effort <high\|max>` | Set reasoning effort |
| `/set show` | Show current config |
| `/search <N>` | Authorize N web searches (needs BOCHA_API_KEY) |
| `/imp <path>` | Import file (txt/md/py/csv/pdf/docx) |
| `/fetch <URL>` | Fetch web page |
| `/cd <node-id>` | Jump to node |
| `/list` | List all nodes |
| `/info [node-id]` | Show node details |
| `/back` | Go to parent node |
| `/root` | Jump to root |
| `/save [node-id]` | Export node as Markdown |
| `/rm <node-id>` | Delete node and children |
| `/show` | Open reply in editor |

---

## AI Tool Reference

AI autonomously invokes these tools as needed:

| Tool | Function | Parameters |
|------|----------|------------|
| `read_file` | Read file (txt/md/py/csv/pdf/docx) | `filepath` |
| `fetch_webpage` | Fetch and extract web page | `url` |
| `list_directory` | List directory contents | `directory`; `show_hidden` (opt) |
| `write_file` | Write/overwrite file (user confirms) | `filepath`; `content` |
| `edit_file` | Search & replace in file (user confirms) | `filepath`; `old_string`; `new_string` |
| `web_search` | Web search (needs `/search` auth) | `query`; `freshness` (opt); `count` (opt) |
| `execute_command` | Execute shell command (AI-audited + user confirms) | `command`; `timeout` |

---

## Project Structure

```
.
├── main.py                  # Entry point (python main.py compat)
├── pyproject.toml           # Package metadata + dependencies
├── mincli.spec              # PyInstaller build config
├── .env.example             # Config template
├── readme.md                # English docs
├── readme.zh.md             # Chinese docs
│
├── mincli/                  # Core package (14 modules)
│   ├── __init__.py          # Version
│   ├── __main__.py          # python -m mincli entry
│   ├── cli.py               # Typer CLI commands
│   ├── config.py            # Constants + config loading
│   ├── models.py            # ConversationNode/Tree, StreamResult
│   ├── helpers.py           # Utilities (balance, tokens, title gen)
│   ├── render.py            # Rich theme + console
│   ├── streaming.py         # Streaming API + live rendering
│   ├── session.py           # InteractiveSession main loop
│   └── tools/
│       ├── registry.py      # Tool definitions
│       ├── execute.py       # Command execution + AI audit
│       ├── file_ops.py      # File read/parse operations
│       ├── web_fetch.py     # Web scraping + search
│       └── thinking.py      # Audit system prompt
│
└── venv/                    # Virtual env (untracked)
```

---

## Building from Source (macOS)

Build a standalone executable + `.dmg`:

```bash
# 1. Build executable
pip install pyinstaller
pyinstaller mincli.spec
# dist/mincli

# 2. Create DMG (needs Homebrew)
brew install create-dmg
create-dmg \
  --volname "mincli" \
  --window-pos 200 120 --window-size 600 400 \
  --icon-size 100 --icon "mincli" 175 200 \
  --hide-extension "mincli" --app-drop-link 425 200 \
  "mincli.dmg" "dist/"
```

Users mount the `.dmg` and drag `mincli` into `/usr/local/bin`.

---

## FAQ

**Q: "Session file corrupted" on startup?**  
A: Delete `~/.mincli_session.json` and restart.

**Q: Thinking mode on but no reasoning shown?**  
A: Make sure using `flash`/`pro` model with `--thinking` enabled.

**Q: `/imp` fails to import PDF/DOCX?**  
A: Install deps: `pip install pdfminer.six python-docx`.

---

## License

MIT License
