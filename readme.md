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
- 🔁 **Workflows (`/wf`)** — save one finished task (or a whole run of turns) as a reusable workflow that lives on disk; `/wf use` attaches it to your next message, `/wf run` executes it right away — no need to re-describe repetitive work
- ⌨️ **Command Completion** — type `/` to list commands; letters filter candidates; `Tab` cycles / completes; a completed command shows its usage help
- 🛡️ **Confirm Dialogs** — destructive actions (`/delete`, `/mcp remove`) ask for confirmation; `←`/`→` switches buttons and the default is *Cancel*
- 💾 **Auto-Save Session** — saved on exit, restored on next launch
- ♻️ **Resumable Interruptions** — press `Esc` (or `Ctrl+C` while busy) to stop generation or kill a running command at any time; a failed/interrupted turn keeps its node (partial answer, reasoning and finished tool results) marked `⚠ interrupted`; type `继续` to carry on instead of losing the turn
- 📄 **Export as Markdown** — `/save` exports any node as `.md`
- 📊 **Real-time Usage Bar** — cache hit rate, balance, next-input tokens and estimated price under the input box (derived from real API `usage`, so it matches the input/output shown at the end of a turn; pricing/peak hours/image tokens are configurable via `~/.mincli/pricing.json`)
- ⚙️ **Dynamic Config** — `/set` changes system prompt, temperature, model, thinking mode, reasoning effort mid-conversation
- 🧩 **Two Models** — `deepseek-flash` (DeepSeek-V4.1-Flash: fast, native multimodal image understanding) and `deepseek-v4-pro` (flagship, text only)

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
| 3 | Project Directory `.env` |

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
| `Esc` / `Ctrl+C` (while busy) | Interrupt the current generation or running command (the partial turn is kept); `Ctrl+C` when idle quits, and a second `Ctrl+C` forces quit if a turn is stuck |
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

> **Tip:** you can also **drag files straight into the terminal window** — most terminals paste the quoted path(s) into the input, and mincli auto-detects them and imports directly (no `/import` needed; multi-file drags work, and plain-text pastes are unaffected).

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | — | DeepSeek API key |
| `MINCLI_SAVE_PATH` | No | `~/Documents/mincli_Conversations` | Export directory |
| `MINCLI_SYSTEM_PROMPT_PATH` | No | Package `mincli/system_prompt.md` | Path to a custom system prompt file |
| `MINCLI_PRICING_PATH` | No | `~/.mincli/pricing.json` | Pricing / peak-hour / image-token overrides |
| `MINCLI_EXEC_MAX_TIMEOUT` | No | `1800` | Upper bound (seconds) for `execute_command`'s `timeout`; raises the old 120s hard cap so long rendering/build tasks aren't cut off |
| `MINCLI_WEBPAGE_MAX_LENGTH` | No | `5000` | Max characters `fetch_webpage` returns (longer pages are truncated with a notice). Hard cap `20000`; out-of-range or invalid values fall back to the default |

### System prompt

The system prompt lives in its own file and is auto-loaded on every startup. Resolution order (highest first):

| Priority | Source |
|----------|--------|
| 1 | File pointed to by `MINCLI_SYSTEM_PROMPT_PATH` |
| 2 | `~/.mincli/system_prompt.md` |
| 3 | Package `mincli/system_prompt.md` (default, ships with the project) |

Edit the matching file to customize the default prompt — it takes effect on the next launch. `mincli info` shows which prompt file is actually in use. If none of the files are available, a minimal built-in fallback prompt is used.

### Real-time usage bar

The two-column bar under the input box (DeepSeek API only) updates live:

**Left: cache hit rate + balance**
- Cache hit rate = `usage.prompt_cache_hit_tokens ÷ (prompt_cache_hit_tokens + prompt_cache_miss_tokens)` for the current node (DeepSeek context caching is automatic; hits are billed at the cache-hit price)
- Balance comes from `GET /user/balance` (`total_balance`, CNY preferred), refreshed every 60 s

**Right: next input + estimated price**
- Next input tokens: for a normal node = **last request's real** `usage.prompt_tokens` + that round's `usage.completion_tokens` (tool definitions are already inside `prompt_tokens`; the answer and its reasoning are sent back as history on the next request and billed the same way), so it lines up with the input/output shown at the end of a turn (measured error ≤5 tokens). Multi-round tool turns use the **last** round, not the turn total. Nodes without usage (`/compact` summary nodes, old sessions, turns that never finished) fall back to a local estimate — tiktoken over messages plus the tool-definition overhead — which over-counts Chinese (1.6–1.9× the real value), so treat it as order-of-magnitude only
- Estimated price = tokens × unit price ÷ 1,000,000, using the current peak/off-peak window (Beijing time, **weekdays** 9-12 and 14-18 are peak) weighted by the cache hit rate (hits at the cache-hit price, the rest at the miss price)

> Note: the input/output tokens shown at the end of a turn are DeepSeek's real `usage`, **summed over every API call** of that turn (what the turn cost); the bar's "next input" only counts the last call (the next request's context).
> The local estimator (tiktoken `cl100k_base`) is not DeepSeek's tokenizer: plain Chinese is over-counted ~1.6×, LaTeX-heavy math ~1.9×, English ~1.0×. The **tool definitions** sent on every request (2 built-in + all MCP server tools; 21 tools ≈ 6.8k tokens) are part of the API's `prompt_tokens` and are added to the fallback estimate.
> `/compact` reports before/after as a tiktoken estimate of the messages alone (no tool definitions), so those numbers differ from the bar by design.

### Pricing (`~/.mincli/pricing.json`)

DeepSeek prices change often (Flash was cut on 2026-09-10, and peak hours are now **weekdays only**), so prices, peak-hour rules and the image-token estimate are configurable instead of hard-coded. If the file is missing or invalid, the built-in defaults (current official prices) are used:

```json
{
  "peak": { "days": [1, 2, 3, 4, 5], "ranges": [[9, 12], [14, 18]], "timezone_offset_hours": 8 },
  "models": {
    "deepseek-flash": { "hit": [0.02, 0.04], "miss": [1.0, 2.0], "output": [4.0, 8.0] },
    "deepseek-v4-pro": { "miss": 4.5, "output": 13.5 }
  },
  "image_tokens": 1024
}
```

- Prices are CNY per million tokens; each field accepts `[off-peak, peak]` or a single number (same price all day).
- `models` is merged per model and per field, so you can override just what changed.
- `peak.days` uses ISO weekdays (Mon=1 … Sun=7); `image_tokens` is the fixed per-image estimate (official cap is 1024).
- Run `mincli info` to see the effective pricing file and image-token value.

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
| `/wf list` | List saved workflows |
| `/wf show <name>` | Show a workflow's full spec (goal / variables / steps & command details) |
| `/wf save <name> [start-node-id]` | Distill the **current node** (or the run from `start-node` to the current node) into a reusable workflow; overwriting an existing name asks for confirmation |
| `/wf use <name>` | Attach the workflow to your **next message** — the next non-command send runs it (one-shot; `/wf stop` cancels) |
| `/wf run <name> [key=value...]` | Execute a workflow **immediately** without typing (unset `{variables}` are inferred by the model from context) |
| `/wf edit <name> [change request]` | With a request: model revises the spec. Without (macOS): open the spec in a system editor and auto-import on save |
| `/wf rename <old> <new>` / `/wf delete <name>` | Rename / delete a workflow (delete is confirmed) |
| `/set system <text>` | Change system prompt |
| `/set temp <value>` | Change temperature |
| `/set model <flash\|pro>` | Switch model (`flash` supports image input; `vision` is a legacy alias for flash) |
| `/set thinking <on\|off>` | Toggle thinking |
| `/set effort <low\|high\|max>` | Set reasoning effort |
| `/set file_confirm <on\|off>` | Confirm before writing/editing files (default on; off lets AI write/modify files directly) |
| `/set show` | Show current config |
| `/mcp list` | Show MCP server config & connection status |
| `/mcp add <name> <command> [args...]` | Add a third-party MCP server (local command); a `http(s)://` second arg adds it as a remote server |
| `/mcp remove <name>` | Remove a third-party MCP server (confirmed) |
| `/mcp reload` | Reload MCP server config |
| `/import <path-or-URL> [...]` | Import file (txt/md/py/csv/pdf/docx), fetch web page, or add images — multiple targets at once; image files become pending images; `/import clear` clears pending imports. Path parsing is cross-platform: Windows backslash paths (`C:\Users\me\a.txt`) and quoted/paths-with-spaces both work |
| `/<node-id>` (e.g. `/a3`) | Jump to node directly |
| `/tree` | List all nodes |
| `/info [node-id]` | Show node details |
| `/up` | Go to parent node |
| `/home` | Jump to root |
| `/full` | Full-view mode: hide the answer area, tree takes full width (input stays; toggle again or switch a node to exit) |
| `/save [node-id]` | Export node as Markdown |
| `/delete <node-id> [...]` | Delete one or more nodes and their children (confirmed; children of a deleted parent are removed together, no not-found error) |
| `/view` | Open reply in editor |

Type `/` in the input box to see the command list; keep typing to filter, `Tab` to complete, and a fully-typed command shows its usage help above the input.

---

## AI Tool Reference

AI autonomously invokes these tools as needed:

| Tool | Function | Parameters |
|------|----------|------------|
| `read_file` | Read file (txt/md/py/csv/pdf/docx) | `filepath` |
| `fetch_webpage` | Fetch and extract web page (truncated at `MINCLI_WEBPAGE_MAX_LENGTH`; failures report the HTTP status) | `url` |
| `list_directory` | List directory contents | `directory`; `show_hidden` (opt) |
| `write_file` | Write/overwrite file (user confirms) | `filepath`; `content` |
| `edit_file` | Search & replace in file (user confirms) | `filepath`; `old_string`; `new_string` |
| `execute_command` | Execute shell command (AI-audited + user confirms) | `command`; `timeout` |
| `query_conversation_tree` | Query conversation tree (in-memory, no MCP) | `root`; `search` (opt) |
| `read_conversation_nodes` | Read conversation nodes (in-memory, no MCP) | `node_ids` |

---

## MCP Integration

mincli's tool execution is built on the standard [MCP protocol](https://modelcontextprotocol.io/):

- **Bundled MCP server**: the 6 external tools (file ops, web fetch, command execution) are provided by a subprocess server that mincli launches and talks to over stdio. Safety/interaction policies (user confirmation, AI audit) stay client-side, so behavior is unchanged. **This bundled server is meant to be launched by mincli only — do not expose it to other MCP clients (e.g. Claude Desktop, Cursor)**: it does not implement its own audit/confirmation/high-risk command protection and relies entirely on mincli's client-side policies.
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

## Workflows (`/wf`)

For repetitive jobs (weekly changelog, periodic repo inspection, a fixed release flow…), save **one finished run** as a workflow instead of describing it from scratch every time:

```
/wf save changelog            # distill the current node (incl. the AI's tool/command steps)
/wf save release a1           # distill the whole run from node a1 → current node
```

Saving invokes the current model to turn the transcript into a spec doc: **goal, steps, and the exact commands/tools used**, with anything that varies per run (versions, dates, paths, per-run content) rewritten as `{variables}` with explanations (e.g. `git log {old}..HEAD`). If distillation fails, the raw transcript is saved instead and you can fix it with `/wf edit`.

```
/wf use changelog             # attach to the next input (hint shown in the status bar)
# next message: run it, then the workflow is auto-detached

/wf run release old=v1.0 new=v2.0   # execute immediately, no typing needed
/wf run release v1.0 v2.0           # positional values fill variables in order
/wf stop                            # cancel an attached workflow
```

Workflow runs go through the exact same pipeline as a normal send (streaming, tool audit and user confirmation still apply). Provided values substitute the placeholders; unset ones are inferred by the model from the message and current context.

```
/wf list                    # list workflows (goal / steps / vars / run count)
/wf show changelog          # print the full spec
/wf edit changelog add a verification step   # model revises the spec
/wf rename changelog cl     # rename
/wf delete changelog        # delete (confirmed)
```

- Workflows are stored apart from the session in `~/.mincli/workflows.json` (override with `MINCLI_WORKFLOWS_PATH`).
- On macOS, `/wf edit <name>` without a request opens the spec in a system editor and auto-imports it on save; other platforms use the model-revision form.
- Re-running `/wf save <name>` overwrites (with confirmation) — redo the task first, then re-save to update a workflow.
- `--no-tui` text mode supports `/wf list/show/save/use/run/delete/rename/edit` (edit = model revision only).

---

## Interrupted Generations & "Continue"

An API failure (rate limit, dropped connection, `Content Exists Risk` moderation, timeout) no longer throws the turn away — and neither does a manual interrupt:

- **Manual interrupt** — press `Esc` (or `Ctrl+C` while a turn is running) to stop the current generation or kill a running command. Streaming stops at the next chunk and the running command's process group is terminated, so a stuck rendering/build job doesn't have to wait for its timeout.
- **The node is saved anyway** — whatever was already streamed (partial answer and reasoning), the tool calls/results that already finished, and the tokens spent are all written to the current node, which is marked `⚠ interrupted` in the tree.
- **The chat log says so** — on an API failure the reason is shown, followed by a note that the turn is saved and that typing `继续` (continue) resumes it.
- **Just continue** — send `继续` (or any follow-up) under that node; the history sent to the model contains the partial answer and tool results, so it picks up where it stopped instead of starting over.
- **Empty answers are never sent back** — if nothing was generated, no empty assistant message is sent to the model (the API would reject it); your question still stays in context.
- Reopening an interrupted node shows whatever was saved before the interruption; delete it with `/delete <node-id>` when you no longer need it.

> If the failure was content moderation, resending the same context may be rejected again — rephrasing or starting a fresh node works better.

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
│   ├── system_prompt.md     # Default system prompt (auto-loaded on startup)
│   ├── controller.py        # ChatController (logic + event stream)
│   ├── models.py            # ConversationNode/Tree
│   ├── workflows.py         # Workflow (/wf) data model + persistent store
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
└── tests/                   # Headless tests (test_controller / test_images / test_tui / test_web_fetch)
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
