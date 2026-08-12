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
- 🔧 **Tool Calling** — AI autonomously invokes 6 tools: read/write/edit files, fetch web pages, list directories, execute commands
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
/import ~/document.pdf
What does this document say?

# Or just ask — AI calls tools autonomously
Show me config.json
Fetch https://example.com
What files are here?
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | — | DeepSeek API key |
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
| `/mcp list` | Show MCP server config & connection status |
| `/mcp add <name> <command> [args...]` | Add a third-party MCP server (local command); a `http(s)://` second arg adds it as a remote server |
| `/mcp remove <name>` | Remove a third-party MCP server |
| `/mcp reload` | Reload MCP server config |
| `/import <path-or-URL>` | Import file (txt/md/py/csv/pdf/docx) or fetch web page |
| `/<node-id>` (e.g. `/a3`) | Jump to node directly |
| `/tree` | List all nodes |
| `/info [node-id]` | Show node details |
| `/up` | Go to parent node |
| `/home` | Jump to root |
| `/save [node-id]` | Export node as Markdown |
| `/delete <node-id>` | Delete node and children |
| `/view` | Open reply in editor |

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
| `execute_command` | Execute shell command (AI-audited + user confirms) | `command`; `timeout` |
| `query_conversation_tree` | Query conversation tree (in-memory, no MCP) | `root`; `search` (opt) |
| `read_conversation_nodes` | Read conversation nodes (in-memory, no MCP) | `node_ids` |

---

## MCP Integration

mincli's tool execution is built on the standard [MCP protocol](https://modelcontextprotocol.io/):

- **Bundled MCP server**: the 6 external tools (file ops, web fetch, command execution) are provided by a subprocess server that mincli launches and talks to over stdio. Safety/interaction policies (user confirmation, AI audit) stay client-side, so behavior is unchanged.
- **Conversation tree tools** (`query_conversation_tree` / `read_conversation_nodes`) depend on in-memory session state and stay in-process.

### Add third-party MCP servers

Configure `~/.mincli/mcp_servers.json` (Claude Desktop-compatible; override path with `MINCLI_MCP_CONFIG`), or use `/mcp add` interactively in chat, `/mcp list` to check status, `/mcp reload` to apply changes:

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

Tools from third-party servers are merged into the AI tool list on startup; on name collision, mincli's own tools win.

**Two kinds of third-party servers are supported:**
- **Local command (stdio)**: `command` + `args` + optional `env`, like the filesystem example above
- **Remote HTTP (streamable-http)**: just set `url`, e.g.:

```json
{
  "mcpServers": {
    "remote-tools": { "url": "https://example.com/mcp" }
  }
}
```

In chat you can also add a remote server directly with `/mcp add <name> <URL>`.

### Packaging note

When packaged with PyInstaller, mincli re-execs itself (`mincli --mcp-server`) to run the server — no separate Python environment needed.

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
├── mincli/                  # Core package (16 modules)
│   ├── __init__.py          # Version
│   ├── __main__.py          # python -m mincli entry
│   ├── cli.py               # Typer CLI commands
│   ├── config.py            # Constants + config loading
│   ├── models.py            # ConversationNode/Tree, StreamResult
│   ├── helpers.py           # Utilities (balance, tokens, title gen)
│   ├── render.py            # Rich theme + console
│   ├── streaming.py         # Streaming API + live rendering
│   ├── session.py           # InteractiveSession main loop
│   ├── mcp_client.py        # MCP client (async bridge + bundled/third-party)
│   ├── mcp_server.py        # Bundled MCP server (7 external tools)
│   └── tools/
│       ├── registry.py      # Local tool defs (conversation tree tools)
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

**Q: `/import` fails to import PDF/DOCX?**  
A: Install deps: `pip install pdfminer.six python-docx`.

---

## License

MIT License
