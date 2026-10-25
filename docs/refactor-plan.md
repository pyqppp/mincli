# mincli 理念重构：实施计划与进度

> 这份文档是重构的唯一事实来源。对话被压缩后，先读本文件，再按「第 7 节 进度」继续执行；不要凭记忆补细节。

最后更新：2026-10-24
分支：`refactor`
最近提交：见 `git log --oneline -5`（每个 Phase 一个提交）
当前状态：Phase 0-6 的实现已完成，等待统一验收

---

## 1. 目标与边界

一句话：mincli 从「可挂载能力、能改代码、能跑命令的编码代理」，收敛为「只做对话与资料阅读/写作的工作台」。

保留：
- 多对话树与节点分支（产品辨识度）
- 图片理解（DeepSeek Flash）
- 流式输出、思考过程、上下文压缩、打断生成
- 文件读取（含图片）、写入与修改、联网搜索与网页抓取

移除：
- shell 执行（`execute_command`）及其审核链、工作区、审核层级
- 全部 MCP（内置 server、第三方 server、客户端、配置、命令、向导外置区）
- 全部工作流（`/wf`、`workflows.py`、`WF_*` 常量）
- 多 provider 与模型注册（只兼容 DeepSeek）
- 客户端内展示/编辑提示词
- 每棵树的能力挂载设置（所有树一样）

---

## 2. 已确认的全部决策

| # | 项 | 决定 |
| --- | --- | --- |
| 1 | 兼容性 | 只兼容 DeepSeek；删除 `register` / `models` 与 `models.json`；`/model` 只在 flash / pro 间切换 |
| 2 | 技能 | 两组固定内置：文件（读含图片、写、改）、联网（搜索 + 抓取，走官方 Search/Fetch API）；另有第三组「树查询」，只由命令开启 |
| 3 | 文件权限 | 四态：关闭 / 只读 / 写需确认 / 写直通，默认关闭 |
| 4 | 联网 | 单开关，默认开 |
| 5 | 树查询 | 保留 `query_conversation_tree`、`read_conversation_nodes`；默认关；不占 footer、不占快捷键；命令 `/recall on|off` |
| 6 | 权限入口 | footer 状态栏（可点击）+ `Ctrl+R` / `Ctrl+W` + `/file` `/web` 命令 |
| 7 | 对话树 | 保留多树与分支；新建即空树；欢迎页按钮 + 侧栏新建 + `Ctrl+N` |
| 8 | 提示词 | 单份 `~/.mincli/prompt.md`；`config.toml` 的 `[prompt] file` 指向它；客户端不显示、不可改 |
| 9 | 配置 | 统一 `~/.mincli/config.toml` |
| 10 | 附件形态 | 输入框内 `[TYPE 名称]` 引用块，原子删除；内容追加在消息末尾 `---` 之后；同一轮同一附件只附一次 |
| 11 | 截断 | 默认 3000 字符，可配；按行边界截；报告总行数与已显示行数 |
| 12 | 分段读取 | 按行，默认 200 行；同一文件允许多次读取 |
| 13 | 行号 | 内联段不带行号；工具输出带行号 |
| 14 | 引用块标签 | 图片 `[IMAGE x.jpg]`；文档按扩展名大写 `[DOCX x.docx]` `[PDF x.pdf]` `[MD x.md]`；网页 `[WEB example.com/标题]`，无标题退化为 `[WEB example.com]`；文件名完整显示不省略 |
| 15 | 聊天区 | 引用块用行内代码样式显示；消息下方一行附件按钮，点击弹只读全文 |
| 16 | 命令面 | 15 条：`/help` `/clear` `/exit` `/tree` `/model` `/thinking` `/compact` `/save` `/up` `/home` `/files` `/file` `/web` `/delete` `/recall`；另有 `/a1` 这类节点跳转语法 |
| 17 | 删节点 | 侧栏选中 + `Delete` 键，同时保留 `/delete` 命令 |
| 18 | 目录列举 | `list_directory` 保留，只读及以上可用 |
| 19 | CLI | 只留 `mincli chat` 与 `mincli info`；启动参数收敛到配置文件（不加 `-m/--temp` 等） |
| 20 | 导入入口 | 取消 `/import`；拖入文件或粘贴路径即生成引用块 |
| 21 | 旧数据 | 旧 `workflows.json`、`mcp_servers.json`、`models.json` 保留文件但不读取、不删除 |
| 22 | 树编号 | 取最小可用正整数：删掉 1 号后新建就是 1 号；有 1、3 号时新建是 2 号 |
| 23 | 通知 | 成功类操作一律静默（开关切换、新建/删除树、删除节点、清空、保存、模型切换、复制、附件插入、取消删除/清理）；只保留警告、错误，以及压缩过程与结果 |
| 24 | 思考开关 | 三路入口：状态栏「思考 开/关」、`Ctrl+T`、`/thinking on|off`；改动立即生效并写回配置 |

### 已执行的数据清理（2026-10-24）

- 删除 `~/.mincli/trees/`（1 棵树 412 KB，含草稿与索引）
- 删除 `~/.mincli/mcp_servers.json`
- 删除远端 Files API 全部 42 个文件（21.6 MB），复查剩余 0
- 保留：`~/Documents/mincli_Conversations/` 的 8 个导出 Markdown（用户未勾选删除）、两个 `.env`（API Key）、`~/.mincli/.DS_Store`

---

## 3. 配置设计

### 3.1 `~/.mincli/config.toml`

```toml
model = "deepseek-flash"        # deepseek-flash | deepseek-v4-pro
temperature = 1.0
thinking = false
effort = "high"                 # low | high | max

[files]
mode = "off"                    # off | read | confirm | direct
inline_max_chars = 3000
attachment_dir = "~/.mincli/attachments"

[web]
enabled = true

[recall]
enabled = false

[prompt]
file = "~/.mincli/prompt.md"
```

- 首次启动自动生成带注释的默认文件；提示词文件缺失时写入默认提示词
- 非法值回退默认并静默；`mincli info` 展示生效值
- 任何开关变化立即写回（保留模板注释：按模板重新生成，值从当前状态填入）
- 读取用 `tomllib`（Python 3.11+）；3.10 通过 `tomli` 依赖兜底
- 环境变量 `MINCLI_CONFIG_PATH` 可覆盖配置路径（保留调试能力）

### 3.2 配置到运行时的映射

- `model/temperature/thinking/effort` → `ChatController` 构造参数
- `files.mode` → 工具列表裁剪 + 写确认
- `web.enabled` → 搜索/抓取工具是否下发
- `recall.enabled` → 树查询工具是否下发
- `files.inline_max_chars` → `compose_message` 的截断阈值
- `files.attachment_dir` → 附件正文落盘目录

---

## 4. 工具设计

### 4.1 工具表

| 工具 | 参数 | 可用条件 |
| --- | --- | --- |
| `read_attachment` | `name`, `offset=1`, `limit=200` | 只要有附件就可用，与文件模式无关 |
| `read_file` | `path`, `offset=1`, `limit=200` | 文件模式 ≥ 只读 |
| `write_file` | `path`, `content` | 文件模式 ≥ 写需确认（确认态弹窗） |
| `edit_file` | `path`, `old_string`, `new_string` | 同上 |
| `list_directory` | `path` | 文件模式 ≥ 只读 |
| `web_search` | `query`, `count=5` | 联网开 |
| `web_fetch` | `url` | 联网开 |
| `query_conversation_tree` | `root?`, `search?`, `tree?` | `/recall on` |
| `read_conversation_nodes` | `node_ids`, `tree?` | `/recall on` |

### 4.2 门控矩阵

- `off`：只下发 `read_attachment`
- `read`：`read_attachment` + `read_file` + `list_directory`（+ recall 工具若开启）
- `confirm`：再加 `write_file` / `edit_file`，每次调用弹确认
- `direct`：写工具不再确认
- 联网与 recall 独立于文件四态

### 4.3 分段读取规范

- `offset` 为 1 起的行号，`limit` 为行数，默认 200
- 输出带行号（内联段不带）
- 单行超长文件（如 JSON）按行无法分段时，工具需有单次输出字符上限，并在结果里说明
- `read_attachment` 的 `name` 用引用块里的名字；重名时用一级目录区分（见第 9 节待确认项）

### 4.4 联网接口

- Search：`POST https://api.deepseek.com/v1/tools/search`，字段 `query` / `count` / `need_content`
- Fetch：`POST https://api.deepseek.com/v1/tools/fetch`，字段 `url`；返回 `title` + Markdown `content`
- 鉴权：`Authorization: Bearer <DEEPSEEK_API_KEY>`；用 `httpx`
- 错误：400/401/402/403/404/408/413/429/500/504，失败文案需可读
- 计费：按成功请求次数（Search 5 元/千次，Fetch 5 元/千次），不按 token

---

## 5. 附件与消息

### 5.1 消息形态（最终）

输入框：

```
帮我读读这个[DOCX 报告.docx]，第 3 节和[PDF paper.pdf]的结论对比一下
```

模型收到：

```
帮我读读这个[DOCX 报告.docx]，第 3 节和[PDF paper.pdf]的结论对比一下

---
报告.docx（共 1,234 行，此处显示前 42 行 / 2,980 字符）：
<前 42 行原文>

（已截断；可用 read_attachment 从第 43 行继续读取）
---
paper.pdf（共 96 页 / 3,214 行，此处显示前 300 行 / 12,450 字符）：
<前 300 行原文>

（已截断；可用 read_attachment 从第 301 行继续读取）
```

规则：
- 短文件：`文件名（共 N 行）：` + 全文，无截断提示
- 单行长文：按字符截断，提示不带行号
- 图片不进文本段，仍按多模态块发送；正文里只留 `[IMAGE ...]`
- 同一轮同路径/URL 只附一次内容；引用块出现几次保留几次
- 网页附件用 Fetch API 的正文；引用块标题取 `title`，缺失退化为域名

### 5.2 附件存储（Phase 2 实现）

- 正文按内容哈希落盘：`<attachment_dir>/<id>.txt`
- 元数据（id / kind / name / 原始路径或 URL / 行数 / 字符数）随节点序列化
- `read_attachment` 从当前会话所有节点的附件里按名字解析
- 启动时回收未被任何会话引用的附件文件
- 图片：保留现有 Files API 上传逻辑（`file_id` 优先，失败回退 base64）

### 5.3 输入框引用块（Phase 0 已完成）

- 文本形式 `[TYPE 名称]`，与普通文字混排
- 退格 / Delete：光标贴块边或块内时整块删除
- 左右键：光标跳过块，不落入块内
- 着色：`bright_cyan` on `grey27`，在 `render_line` 里对 Strip 贴样式
- 拖入 / 粘贴路径：光标处插入块，前后补空格
- 提交：`Submitted` 消息携带附件快照（输入框随后立刻清空）

---

## 6. 界面

### 6.1 底部状态栏

- 替代 Textual 自带 Footer（不可点击）
- 左：`Ctrl+R 文件权限   Ctrl+W 联网   /help 帮助`
- 右：`文件 <状态>`、`联网 <开/关>`，两者均可点击
- 点击、`Ctrl+R` / `Ctrl+W`、`/file` `/web` 三条路径修改同一状态
- `Ctrl+W` 用 priority 绑定覆盖 TextArea 的「删前一个词」；`Ctrl+Backspace` / `Alt+Backspace` 仍可删词

### 6.2 聊天区

- 用户消息里的引用块渲染成行内代码样式
- 消息下方一行附件按钮（文件名），点击弹只读全文（大文本用只读 TextArea，带行号）
- 不加自定义渲染器，避免用户 Markdown 降级

### 6.3 新建与欢迎

- 一棵树都没有时显示欢迎页，页面有新建按钮
- 侧栏顶部有新建按钮
- `Ctrl+N` 新建
- 新建即空树，无向导、无能力勾选
- `Delete` 键删除选中树节点（含子树，确认弹窗）；`/delete` 命令同样可用

---

## 7. 进度

### Phase 0 输入块与状态栏：已完成（`6e734ee`）

已实现：
- `mincli/attachments.py`：`Attachment`、`load_attachment`、`truncate_at_line`、`compose_message`
- `ChatInput`：引用块插入、着色、整块删除、光标跳过、提交携带附件快照
- 底部状态栏：文件四态 + 联网开关，点击 / 快捷键 / 命令三路
- 拖入与粘贴路径生成引用块
- 发送时组装 `---` 附件段，聊天区回显用户原话
- 测试：`tests/test_attachments.py`（7）、`tests/test_chips.py`（4）
- 全量 85 个测试通过

未提交（本轮）：
- 修复引用块不上色：`ChatInput.render_line` 重写，剥离 TextArea Theme 的覆盖
- `tests/test_chips.py` 改为断言渲染后的 Strip 颜色

Phase 0 临时状态（后续 Phase 修）：
- 文件四态与联网开关尚未裁剪工具列表
- 聊天区引用块仍是纯文本，无附件按钮
- 配置不持久化（重启回到默认）
- `/import`、`/files`、`/wf`、`/mcp` 仍在
- 无树时仍弹旧建树向导

### Phase 1 配置层：已完成

- `mincli/config.py` 重写：`Config` 数据类 + `load_config/save_config/ensure_config`；
  `tomllib`（3.10 用 `tomli`）；默认值生成、非法值静默回退、写回时按模板重新生成注释
- 提示词收敛为 `~/.mincli/prompt.md` 单份；`system_prompt()` 读取，缺失时回退内置默认
- 删除模型注册 / MCP / 工作流 / shell 执行的常量与配置读写
- `mincli/cli.py` 只剩 `chat` 与 `info`；`info` 只读展示生效值，不打印提示词内容
- 环境变量：`MINCLI_CONFIG_PATH`、`MINCLI_PROMPT_PATH`、`MINCLI_ATTACHMENTS_PATH`（测试/调试）
- 测试：`tests/test_config.py`（12）

### Phase 2 附件管线：已完成

- `Attachment` 增 `stored_path`、`metadata()`、`from_metadata()`；正文按内容哈希落盘
  `<attachment_dir>/<id>.txt`
- `ConversationNode.attachments` 随节点序列化（旧存档缺字段按空处理）
- `read_attachment` 按名字解析本轮/历史附件，重名给候选提示，按行续读
- 启动时 `controller._collect_attachments()` 回收未被任何树引用的正文文件
- 网页附件改走官方 Fetch API（`fetch_page`），chip 标题取 `title`，缺失退化为域名
- `compose_message` 规则不变：短全文、长截断、单行长文按字符截、图片不进文本段、同轮去重
- 测试：`tests/test_attachments.py`（10）

### Phase 3 工具层：已完成

- `mincli/tools/file_tools.py`：`read_file`（按行、行号、续读提示）、`write_file`、
  `edit_file`（唯一匹配，多处报错）、`list_directory`（目录在前、文件带大小）
- `mincli/tools/web_tools.py`：`web_search` / `web_fetch` / `fetch_page`；httpx、30/60s 超时、
  按状态码给出可读错误；Search 结果带标题/链接/摘要，正文按需摘录
- `mincli/tools/builtin.py`：唯一工具表 + `available_tools(cfg, has_attachments)` 门控 +
  `ToolContext` + `dispatch_tool` 进程内分派（无 MCP、无子进程）
- `ChatController._rebuild_llm_tools()` / 发送时按本轮附件重算；写确认走
  `_confirm_write`（仅 `confirm` 模式弹窗）；树查询走 `_recall_dispatch`
- 删除 shell 执行、审核层级、工作区、`tools/execute.py`、`tools/registry.py`、`tools/thinking.py`
- 测试：`tests/test_tools.py`（23，含门控矩阵与联网错误分支）

### Phase 4 输入与聊天区完善：已完成

- 聊天区用户消息里的引用块渲染为行内代码（`_chips_as_code`）
- 用户消息下方附件按钮行（`att-<节点ID>-<序号>`），点击弹出只读 `AttachmentViewScreen`
- 草稿与附件随树持久化：`TreeState.draft` + `draft_attachments`，`ChatInput.restore_chips`
- 节点删除：侧栏选中后 `Delete`（`check_action` 限定焦点在树上）+ `/delete`
- 附件导入改到线程执行（`asyncio.to_thread`），网页抓取不卡 UI

### Phase 5 命令面与界面：已完成

- `mincli/tui/commands.py` 收敛为 15 个一级命令（加 `/a1` 共 16 项）并同步补全与 `/help`
- 删除 `/wf` `/mcp` `/import` `/view` `/info` `/full` `/set`
- 欢迎页（`#welcome` + `#welcome-new`）、侧栏新建、`Ctrl+N`；删除 `tui/tree_wizard.py`
- `/recall on|off`；底部状态栏三路入口一致
- 删除导入提示与悬停弹窗
- 测试：`tests/test_tui.py`

### Phase 6 清理、文档与测试：已完成

- 删除 `mcp_client.py`、`mcp_server.py`、`workflows.py`、`tools/execute.py`、
  `tools/thinking.py`、`tools/registry.py`、`tools/web_fetch.py`、`tui/tree_wizard.py`、
  双份 `system_prompt*.md`
- `pyproject.toml`：移除 `mcp`、`trafilatura`、`requests`；新增 `httpx`、`tomli`（<3.11）；
  `textual>=8`；版本 `0.2.0`
- 测试重写：`test_controller`（25）、`test_tui`（25）、`test_config`（12）、
  `test_tools`（23）、`test_attachments`（10）、`test_images`、`test_trees`；
  删除 `test_web_fetch.py`；`check()` 在 pytest 下失败会抛断言（不再静默通过）
- `readme.md` / `readme.zh.md` 重写
- 全量测试 + `mincli info` 冒烟（临时 HOME）

## 8. 技术结论与坑（已验证）

1. **Textual 版本**：实际安装 `8.2.8`（`pyproject` 仍写 `>=4.0`）。8.x 没有 `SyntaxHighlighter`，TextArea 用 tree-sitter 做语法高亮。
2. **TextArea 着色**：在 `get_line()` 里 `Text.stylize` 会被 TextArea Theme 的 `base_style` 覆盖。正确做法是重写 `render_line(y)`，等 Strip 生成后用 `Strip.crop()` 切段，再 `Segment.apply_style(..., post_style=CHIP_STYLE)` 贴样式；`Strip.apply_style()` 是前置样式，会被段内样式盖掉。
3. **Strip 切分**：`Strip.divide([0, n])` 在起点为 0 时不会返回 3 段，用 `crop` 更稳。
4. **单元格换算**：CJK 文件名要按 `rich.cells.cell_len` 换算单元格；软换行用 `wrapped_document._offset_to_line_info[scroll_y + y]` 得到 `(line_index, section_offset)`。
5. **提交竞态**：`ChatInput.action_submit_message` 先 `post_message` 再 `clear()`，App 收到时正文已空。必须让 `Submitted` 在清空前携带附件快照。
6. **Footer 不可点击**：Textual `Footer` 只是按键提示。可点击状态栏用一个普通 `Horizontal` + 三个 `Static`，在 `App.on_click` 里按 `event.widget.id` 分发；自定义容器子类会在渲染时报 `'NoneType' object has no attribute 'render_strips'`，不要用。
7. **Ctrl+W 冲突**：TextArea 的 `ctrl+w` 是删词。App 级 `priority=True` 绑定可以覆盖；代价是输入框里 Ctrl+W 不再删词。
8. **pytest**：异步测试需要 `asyncio_mode = "auto"`，已写入 `pyproject.toml`；未配置时 14 个 TUI 测试报「async def functions are not natively supported」。
9. **环境**：Python 3.11.9（`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`）；依赖用清华源装进本机，无虚拟环境：`textual 8.2.8`、`rich 15.0.0`、`pytest 9.1.1`、`pytest-asyncio`，`mincli` 为可编辑安装。安装写 site-packages 需要 danger-full-access。
10. **旧测试基线**：改动前 74 个测试在 textual 8.2.8 下全绿；Phase 0 后 85 个全绿。
11. **MCP 现状**：内置 server 通过 `python -m mincli.mcp_server` 子进程提供 `read_file` / `list_directory` / `write_file` / `edit_file` / `execute_command` / `fetch_webpage` / `cancel_command`；Phase 3 起全部改进程内，Phase 6 删文件。
12. **工具调用链**：`controller._run_tool` 目前先分派进程内两个树查询工具与写/编辑/执行，再转发 MCP。Phase 3 改成纯本地分派。

---

## 9. 待最终验收时确认的小项

1. **同名附件**：同一消息里两个不同目录的同名文件，我计划自动带一级目录区分（如 `[DOCX papers/报告.docx]`）；要么改为拒绝导入并提示。
2. **`Ctrl+W` 覆盖删词**：已按你的选择实现，输入框里删词需用 `Ctrl+Backspace` / `Alt+Backspace`；如果不可接受，可换成 `Alt+W`。
3. **CLI 参数**：计划让 `mincli chat` 不再接受 `-m/--temp/--thinking/--effort`，全部走 `config.toml`；如需保留快速覆盖参数，需回退这条。
4. **`/files` 的定位**：保留为 Files API 远端图片管理（list/info/delete/clean）；它不是本地附件缓存的管理入口。
5. **`list_directory` 的返回格式**：Phase 3 定稿（是否显示文件大小、是否递归）。
6. **状态栏外观**：文件态与联网态目前是纯文本；是否加颜色区分（关闭灰、只读白、可写绿）由验收时定。

---

## 10. 运行与测试

```bash
cd /Users/pengyingqi/mincli
python3 -m mincli chat        # 或 mincli chat（可编辑安装）
python3 -m mincli info
python3 -m pytest -q tests/   # 当前 114 passed
```

- 工作区：`/Users/pengyingqi/mincli`
- 分支：`refactor`
- 提交前先跑全量测试；提交信息用中文、不带 emoji

---

## 11. 附录 A：默认提示词草案

写入 `~/.mincli/prompt.md`（文件不存在时生成）：

```markdown
你是 mincli 的 AI 助手，运行在用户的终端里，通过对话帮用户阅读资料、写作与查证。

## 工作方式
- 语言跟随用户；代码、命令、路径、标识符保持原样，不要翻译。
- 用 Markdown 组织回答；数学公式用 LaTeX（行内 $...$，行间 $$...$$）。
- 若用户要把内容写入 Obsidian 笔记，行内公式用 $...$，行间公式用 $$...$$。
- 能用工具确认的事先去确认，不要凭猜测描述文件内容。
- 改文件前先读原文件；小改动用 edit_file，整体重写才用 write_file；改完说明改了什么。
- 工具失败或结果为空时如实说明，不要编造成功或结果。
- 删除、覆盖这类不可逆操作，先说清影响再执行。

## 资料与附件
- 用户消息里的 [PDF ...]、[DOCX ...]、[IMAGE ...]、[WEB ...] 是附件引用。
- 附件正文附在消息末尾的 --- 之后；若标注「已截断」，用 read_attachment 从提示的行继续读取（按行、可多次读）。
- 图片以多模态块单独发送。

## 能力
- 可用工具以本轮实际提供的为准；缺少完成当前任务的工具时，直接告诉用户当前能力状态，不要假装已执行。
- 历史对话以树状保存，用户可以在任意节点继续提问；你没有读取历史节点的工具。

## 客户端
- 以 / 开头的输入由客户端处理，你不会收到；用户想用时直接给出正确的 /命令。
```

---

## 12. 附录 B：Phase 0 关键代码位置

- 附件模型与组装：`mincli/attachments.py`
- 引用块与渲染：`mincli/tui/widgets.py`（`ChatInput.insert_chip` / `chip_spans` / `_chip_cell_ranges` / `render_line` / `action_delete_left` / `action_delete_right` / `action_cursor_left` / `action_cursor_right`）
- 提交携带附件：`ChatInput.action_submit_message`、`ChatInput.Submitted`
- 状态栏：`ChatApp.compose` 的 `#status-footer`、`_refresh_status_footer`、`_set_file_mode`、`_set_web`、`on_click`
- 发送组装：`ChatApp.on_chat_input_submitted`、`ChatApp._send_user_text`
- 拖入：`ChatApp._attach_targets`、`on_chat_input_files_dropped`、`on_paste`
- 样式：`mincli/tui/chat.tcss` 末尾 `#status-footer` 等规则
