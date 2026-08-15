> [🇨🇳 中文](readme.zh.md)

# mincli

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tree-structured chat AI assistant powered by DeepSeek V4 models, built on a **Textual TUI**.  
Streaming Markdown output, branching conversations, full reasoning chain display.  
Switch models / system prompts / temperature / thinking mode on the fly; the AI can autonomously invoke tools — file read/write/edit, web fetching, directory listing, and shell execution.

---

## Features

- 🖥️ **Textual TUI** — sidebar conversation tree + streaming Markdown chat log + multi-line input box
- 🚀 **Streaming Output** — real-time Markdown rendering; tables wrap to terminal width
- 🌲 **Tree Conversations** — main line + branch nodes with globally unique IDs; click nodes to switch, jump with `/<id>`
- 🧠 **Thinking Mode** — full V4 reasoning chain display, toggleable on the fly
- 🔧 **Tool Calling** — AI autonomously invokes tools: read/write/edit files, fetch web pages, list directories, execute commands (user-confirmed)
- ⌨️ **Command Completion** — type `/` to list commands; letters filter candidates; `Tab` cycles / completes; a completed command shows its usage help
- 🛡️ **Confirm Dialogs** — destructive actions (`/delete`, `/mcp remove`) ask for confirmation; `←`/`→` switches buttons and the default is *Cancel*
- 💾 **Auto-Save Session** — saved on exit, restored on next launch
- 📄 **Export as Markdown** — `/save` exports any node as `.md`
- ⚙️ **Dynamic Config** — `/set` changes system prompt, temperature, model, thinking mode, reasoning effort mid-conversation
- 🧩 **Dual Model** — `deepseek-v4-flash` (fast) and `deepseek-v4-pro` (flagship)

---

## Recommended Terminal

**macOS**: [iTerm2](https://iterm2.com/) recommended — reliable keyboard protocol handling (Chinese IME and lock keys work correctly).  
Other terminals (Windows Terminal, Linux) work too; IME behavior depends on the terminal.

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
# Recommended: Textual TUI
mincli chat

# Plain text fallback (no TUI, no extra dependencies)
mincli chat --no-tui

# Python module
python -m mincli chat

# Legacy compat
python main.py chat
```

---

## Quick Start

```bash
# Basic conversation (TUI)
mincli chat

# Enable thinking mode
mincli chat --thinking

# Select model + reasoning effort
mincli chat --model pro --thinking --effort max

# View all options
mincli chat --help
```

### TUI keyboard shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+J` / `Alt+Enter` | Newline |
| `Tab` | Complete / cycle command completion candidates |
| `↑` / `↓` | Scroll the answer area (when the input is empty); double-press and hold for 2× speed |
| `Ctrl+C` | Quit (copy wins when text is selected) |

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
| `--effort` | `high` | Reasoning effort: `low` \| `high` \| `max` |
| `--temp` | `1.0` | Temperature |
| `--no-tui` | off | Plain-text chat loop (no Textual TUI) |

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
| `/set effort <low\|high\|max>` | Set reasoning effort |
| `/set show` | Show current config |
| `/mcp list` | Show MCP server config & connection status |
| `/mcp add <name> <command> [args...]` | Add a third-party MCP server (local command); a `http(s)://` second arg adds it as a remote server |
| `/mcp remove <name>` | Remove a third-party MCP server (confirmed) |
| `/mcp reload` | Reload MCP server config |
| `/import <path-or-URL>` | Import file (txt/md/py/csv/pdf/docx) or fetch web page |
| `/<node-id>` (e.g. `/a3`) | Jump to node directly |
| `/tree` | List all nodes |
| `/info [node-id]` | Show node details |
| `/up` | Go to parent node |
| `/home` | Jump to root |
| `/full` | Full-view mode: hide the answer area, tree takes full width (input stays; toggle again or switch a node to exit) |
| `/reasoning` | Expand/collapse the current message's reasoning (auto-collapsed once the answer starts streaming; click the folded block too) |
| `/save [node-id]` | Export node as Markdown |
| `/delete <node-id>` | Delete node and children (confirmed) |
| `/view` | Open reply in editor |

Type `/` in the input box to see the command list; keep typing to filter, `Tab` to complete, and a fully-typed command shows its usage help above the input.

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

---

## Project Structure

```
.
├── main.py                  # Entry point (python main.py compat)
├── pyproject.toml           # Package metadata + dependencies
├── .env.example             # Config template
├── readme.md                # English docs
├── readme.zh.md             # Chinese docs
│
├── mincli/                  # Core package
│   ├── __init__.py          # Version
│   ├── __main__.py          # python -m mincli entry
│   ├── cli.py               # Typer CLI: chat (TUI / --no-tui), info
│   ├── config.py            # Constants + config loading
│   ├── controller.py        # ChatController (logic + event stream)
│   ├── models.py            # ConversationNode/Tree
│   ├── helpers.py           # Utilities (tokens, title gen, formulas)
│   ├── streaming.py         # Streaming API interaction
│   ├── mcp_client.py        # MCP client (async bridge + bundled/third-party)
│   ├── mcp_server.py        # Bundled MCP server
│   ├── tui/                 # Textual TUI
│   │   ├── app.py           # ChatApp (layout, commands, events)
│   │   ├── chat.tcss        # TUI styles
│   │   ├── confirm.py       # Confirm dialog (←/→ switch, default cancel)
│   │   └── widgets.py       # ChatInput (multi-line + completion)
│   └── tools/
│       ├── registry.py      # Local tool defs (conversation tree tools)
│       ├── execute.py       # Command execution + AI audit
│       ├── file_ops.py      # File read/parse operations
│       ├── web_fetch.py     # Web scraping + search
│       └── thinking.py      # Audit system prompt
│
└── tests/                   # Headless tests (test_controller / test_tui)
```

---

## FAQ

**Q: "Session file corrupted" on startup?**  
A: Delete `~/.mincli_session.json` and restart.

**Q: Thinking mode on but no reasoning shown?**  
A: Make sure using `flash`/`pro` model with `--thinking` enabled.

**Q: `/import` fails to import PDF/DOCX?**  
A: Install deps: `pip install pdfminer.six python-docx`.

**Q: TUI won't start (e.g. output is piped or terminal isn't supported)?**  
A: Run `mincli chat --no-tui` for the plain-text fallback.

---

## License

MIT License
