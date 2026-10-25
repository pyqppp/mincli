> [中文](readme.zh.md)

# mincli

A tree-structured chat assistant built on **Textual TUI** and DeepSeek V4, focused on reading material and writing.
Streaming Markdown, reasoning display, branching conversations and multiple trees. It can read files (including images), edit files, search the web and fetch pages — but it never runs shell commands or connects to external services.

---

## Features

- **Tree-structured chat** — main line plus branches; click a node to switch, or jump with an id such as `/a1`
- **Multiple trees** — each tree has a number and a fixed color (the theme follows the active tree); `Ctrl+1..8` switches trees. All trees are identical, and a new tree starts empty
- **Streaming output** — live Markdown rendering; interrupted answers can be continued
- **Reasoning** — DeepSeek reasoning chain shown in its own block, toggleable
- **Attachments** — drop a file or paste a path to create a chip such as `[DOCX report.docx]`; short content is inlined, long content is truncated and can be paged with `read_attachment`
- **Three built-in tool groups** — files (read/write/edit/list), web (official Search / Fetch API), conversation-tree recall (off by default)
- **Capability switches** — four file modes, web access and thinking, reachable from the status bar, keyboard shortcuts, or commands; changes are written back to the config immediately
- **Context compaction** — `/compact` turns the current branch into a detailed summary node
- **Image understanding** — native multimodal support on `deepseek-flash`; local images are uploaded to the Files API, managed with `/files`
- **Usage and balance** — the status line shows cache hit rate, balance, next input tokens and estimated cost

---

## Install

```bash
pip install -e .
```

Python 3.10+ required (3.11+ uses built-in `tomllib`; 3.10 installs `tomli` automatically).

Set the API key (either way):

```bash
export DEEPSEEK_API_KEY=sk-...
# or put it in ~/.mincli/.env
echo 'DEEPSEEK_API_KEY=sk-...' > ~/.mincli/.env
```

## Run

```bash
mincli chat     # start the TUI
mincli info     # show the effective config (read-only)
```

The first run creates `~/.mincli/config.toml` and `~/.mincli/prompt.md`.

---

## Configuration

Everything lives in `~/.mincli/config.toml`; the CLI accepts no startup flags. Edits take effect on the next start; switches changed inside the UI are written back immediately (comments are regenerated from the template).

```toml
model = "deepseek-flash"              # deepseek-flash (vision) | deepseek-v4-pro
temperature = 1.0                     # 0.0 - 2.0
thinking = false                      # enable reasoning by default
effort = "high"                       # low | high | max

[files]
mode = "off"                          # off | read | confirm | direct
inline_max_chars = 3000               # max characters inlined into a message
attachment_dir = "~/.mincli/attachments"

[web]
enabled = true                        # web search and page fetch

[recall]
enabled = false                       # conversation-tree tools (also /recall on|off)

[prompt]
file = "~/.mincli/prompt.md"          # system prompt; never shown or edited in the client
```

- Invalid values silently fall back to defaults and never block startup
- The system prompt is this single file; edit it with any editor, it is not visible inside the client
- `MINCLI_CONFIG_PATH` overrides the config path (debugging)

---

## Capability switches

Three entry points change the same state, written back to `config.toml` immediately:

| Entry | File permission | Web |
| --- | --- | --- |
| Status bar | click "文件 ..." to cycle four modes | click "联网 开/关" |
| Shortcut | `Ctrl+R` | `Ctrl+W` |
| Command | `/file off\|read\|confirm\|direct` | `/web on\|off` |

File modes:

| Mode | Meaning | Tools exposed |
| --- | --- | --- |
| `off` | default | `read_attachment` only (when there are attachments) |
| `read` | reading allowed | adds `read_file`, `list_directory` |
| `confirm` | confirm every write | adds `write_file`, `edit_file` |
| `direct` | write without asking | same, no confirmation |

Web and file permissions are independent. Conversation-tree recall is controlled separately by `/recall on|off` (off by default, not shown in the status bar).

Thinking has the same three entry points: the status bar item, `Ctrl+T`, and `/thinking on|off`. Changes apply to the next request and are written back to `thinking` in `config.toml`; reasoning effort still lives in the config file.

---

## Attachments

- Drop a file or paste a path; the input box creates a chip: `[IMAGE x.jpg]`, `[DOCX x.docx]`, `[PDF x.pdf]`, `[WEB example.com/title]`
- Chips mix with text, are deleted as one unit with Backspace/Delete, and the arrow keys skip over them
- On send, bodies are appended after `---` at the end of the message:
  - shorter than `inline_max_chars`: full text
  - longer: truncated on a line boundary with a hint to continue via `read_attachment`
  - images never go into the text; they are sent as multimodal blocks
- The same file/URL is inlined once per turn, no matter how many chips reference it
- Attachment bodies are stored under `attachment_dir`, so past sessions can still page them; unreferenced bodies are collected at startup

---

## Tools

| Tool | Purpose | Available when |
| --- | --- | --- |
| `read_attachment` | page through attachment text | any attachment exists |
| `read_file` | read a local file by lines (text, MD, CSV, PDF, DOCX) | file mode >= read |
| `list_directory` | list a directory (non-recursive) | file mode >= read |
| `write_file` | write/overwrite a file | file mode >= confirm |
| `edit_file` | replace one unique snippet | file mode >= confirm |
| `web_search` | search the web (title/link/summary) | web enabled |
| `web_fetch` | fetch a page as Markdown | web enabled |
| `query_conversation_tree` | inspect the conversation tree | `/recall on` |
| `read_conversation_nodes` | read past nodes | `/recall on` |

Web access uses the official DeepSeek endpoints (`POST /v1/tools/search`, `POST /v1/tools/fetch`) and is billed per successful request.

---

## Commands

| Command | Description |
| --- | --- |
| `/help` | show help |
| `/clear` | clear the current tree |
| `/compact` | compact the current branch into a summary node |
| `/exit` | quit (saves automatically) |
| `/tree`, `/tree new`, `/tree delete <n>` | list, create, delete trees |
| `/up`, `/home` | go to parent / root node |
| `/delete <node-id>` | delete a node and its subtree (asks first) |
| `/recall on\|off` | toggle conversation-tree tools |
| `/file <off\|read\|confirm\|direct>` | set file permission |
| `/files [list\|info\|delete\|clean]` | manage Files API images |
| `/save [node-id]` | export a node as Markdown |
| `/web <on\|off>` | toggle web access |
| `/model [flash\|pro]` | show or switch model |
| `/thinking [on\|off]` | show or toggle thinking |
| `/a1` style input | jump to a node |

Type `/` for completions: `Tab` cycles, `Shift+Tab` reverses, `Enter` accepts.

---

## Key bindings

| Key | Action |
| --- | --- |
| `Enter` | send |
| `Shift+Enter` | newline |
| `Esc` | interrupt generation |
| `Ctrl+C` | quit |
| `Ctrl+N` | new tree |
| `Ctrl+R` | cycle file mode |
| `Ctrl+W` | toggle web |
| `Ctrl+T` | toggle thinking |
| `Ctrl+1..8` | switch to tree N |
| `Delete` | delete the selected sidebar node (asks first) |
| `Ctrl+Backspace` / `Alt+Backspace` | delete a word in the input |

---

## Data files

| Path | Contents |
| --- | --- |
| `~/.mincli/config.toml` | all settings |
| `~/.mincli/prompt.md` | system prompt |
| `~/.mincli/trees/` | one JSON per tree plus an index |
| `~/.mincli/attachments/` | attachment text copies |
| `~/.mincli/.env` | API key |
| `~/.mincli/pricing.json` | optional price/peak-hour overrides |
| `~/Documents/mincli_Conversations/` | Markdown exports from `/save` |

---

## Differences from the previous version

This refactor narrows the scope: from "a coding agent with pluggable capabilities and shell access" to "a workbench for conversation, reading and writing".

- Removed shell execution, all MCP, all workflows, model registration and multi-provider support
- Removed in-client prompt viewing/editing and per-tree capability mounting
- Removed `/import`; dropped files or pasted paths now create chips
- Old files `~/.mincli/workflows.json`, `mcp_servers.json`, `models.json` stay on disk but are no longer read

## Limitations

- Only the two DeepSeek models; `deepseek-v4-pro` does not accept images
- `.doc` must be converted to `.docx` first
- Search and Fetch are billed per successful request
