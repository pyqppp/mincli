> [English](readme.md)

# mincli

基于 DeepSeek V4 与 **Textual TUI** 的树状对话助手，专注阅读资料与写作。
流式 Markdown、思考过程、对话分支与多棵树；可以读文件（含图片）、改文件、联网搜索与抓取网页，但不执行 shell 命令、不连接外部服务。

---

## 特性

- **树状对话** — 主线 + 分支节点，点击节点切换，`/a1` 这类编号可直接跳转
- **多对话树** — 每棵树编号与固定配色（主题随当前树切换），`Ctrl+1..8` 快速切换；所有树完全一样，新建即空树
- **流式输出** — 实时 Markdown 渲染，支持中断后继续生成
- **思考过程** — DeepSeek 推理链单独成块展示，可随时开关
- **附件导入** — 拖入文件或粘贴路径，输入框里生成 `[DOCX 报告.docx]` 这类引用块；短内容直接内联，长内容截断并可用 `read_attachment` 按行续读
- **三组固定内置工具** — 文件（读/写/改/列目录）、联网（官方 Search / Fetch API）、对话树查询（默认关闭）
- **能力开关** — 文件权限四态与联网开关，底部状态栏点击、快捷键、命令三路修改，立即写回配置
- **上下文压缩** — `/compact` 把当前分支压成详细摘要，长对话继续可用
- **图片理解** — `deepseek-flash` 原生多模态；本地图片自动上传 Files API，`/files` 管理远端文件
- **用量与余额** — 输入栏状态条显示缓存命中率、余额、下次输入 token 与预计花费

---

## 安装

```bash
pip install -e .
```

需要 Python 3.10+（3.11+ 使用内置 `tomllib`，3.10 自动安装 `tomli`）。

配置 API Key（任选其一）：

```bash
export DEEPSEEK_API_KEY=sk-...
# 或写入 ~/.mincli/.env
echo 'DEEPSEEK_API_KEY=sk-...' > ~/.mincli/.env
```

## 运行

```bash
mincli chat     # 启动对话界面
mincli info     # 只读查看当前生效配置
```

首次运行会自动生成 `~/.mincli/config.toml` 与 `~/.mincli/prompt.md`。

---

## 配置

所有设置集中在 `~/.mincli/config.toml`，命令行不接受启动参数。修改后下次启动生效；界面里的开关变化会立即写回（注释按模板重新生成）。

```toml
model = "deepseek-flash"              # deepseek-flash（支持图片）| deepseek-v4-pro
temperature = 1.0                     # 0.0 - 2.0
thinking = false                      # 是否默认开启思考
effort = "high"                       # low | high | max

[files]
mode = "off"                          # off 关闭 | read 只读 | confirm 写需确认 | direct 写直通
inline_max_chars = 3000               # 附件内联进消息的最大字符数
attachment_dir = "~/.mincli/attachments"

[web]
enabled = true                        # 联网搜索与网页抓取

[recall]
enabled = false                       # 对话树检索工具（也可用 /recall on|off）

[prompt]
file = "~/.mincli/prompt.md"          # 系统提示词文件；客户端不显示、不可编辑
```

- 非法值静默回退默认值，不阻塞启动
- 系统提示词只有这一份文件，可直接用编辑器改，客户端里看不到内容
- 环境变量 `MINCLI_CONFIG_PATH` 可覆盖配置路径（调试用）

---

## 能力开关

三组入口修改同一状态，改动立即写回 `config.toml`：

| 入口 | 文件权限 | 联网 |
| --- | --- | --- |
| 底部状态栏 | 点击「文件 …」循环四态 | 点击「联网 开/关」 |
| 快捷键 | `Ctrl+R` | `Ctrl+W` |
| 命令 | `/file off\|read\|confirm\|direct` | `/web on\|off` |

文件权限四态：

| 模式 | 含义 | 下发的工具 |
| --- | --- | --- |
| `off` 关闭 | 默认 | 只有 `read_attachment`（有附件时） |
| `read` 只读 | 可读文件 | 加 `read_file`、`list_directory` |
| `confirm` 写需确认 | 每次写入前弹窗 | 加 `write_file`、`edit_file` |
| `direct` 写直通 | 不弹窗 | 同上，直接写 |

联网与文件权限相互独立；对话树查询单独由 `/recall on|off` 控制，默认关闭，不占状态栏。

---

## 附件

- 拖入文件、或直接粘贴路径，输入框里生成引用块：`[IMAGE x.jpg]`、`[DOCX x.docx]`、`[PDF x.pdf]`、`[WEB example.com/标题]`
- 引用块与文字混排，退格/Delete 整块删除，左右键跳过整块
- 发送时正文附在消息末尾 `---` 之后：
  - 短于 `inline_max_chars`：附全文
  - 更长：按行截断，并提示用 `read_attachment` 从下一行继续读
  - 图片不进文本，按多模态块发送
- 同一轮里同一个文件/网页只附一次正文，引用块出现几次保留几次
- 附件正文落盘到 `attachment_dir`，历史会话里的附件也能续读；启动时回收没有被任何会话引用的正文文件

---

## 工具

| 工具 | 作用 | 可用条件 |
| --- | --- | --- |
| `read_attachment` | 按行续读本轮/历史附件 | 只要有附件 |
| `read_file` | 按行读本地文件（文本、MD、CSV、PDF、DOCX） | 文件权限 ≥ 只读 |
| `list_directory` | 列目录（不递归） | 文件权限 ≥ 只读 |
| `write_file` | 写入/覆盖文件 | 文件权限 ≥ 写需确认 |
| `edit_file` | 替换唯一匹配的一段原文 | 文件权限 ≥ 写需确认 |
| `web_search` | 搜索网页（标题/链接/摘要） | 联网开启 |
| `web_fetch` | 抓取网页正文（Markdown） | 联网开启 |
| `query_conversation_tree` | 查询对话树结构 | `/recall on` |
| `read_conversation_nodes` | 读取历史节点内容 | `/recall on` |

联网走 DeepSeek 官方接口（`POST /v1/tools/search`、`POST /v1/tools/fetch`），按成功请求次数计费。

---

## 命令

| 命令 | 说明 |
| --- | --- |
| `/help` | 显示帮助 |
| `/clear` | 清空当前对话树 |
| `/compact` | 压缩当前分支为摘要节点 |
| `/exit` | 退出（自动保存） |
| `/tree`、`/tree new`、`/tree delete <编号>` | 列出、新建、删除对话树 |
| `/up`、`/home` | 返回父节点、回到根节点 |
| `/delete <节点ID>` | 删除节点及其子树（需确认） |
| `/recall on\|off` | 开关对话树检索工具 |
| `/file <off\|read\|confirm\|direct>` | 设置文件权限 |
| `/files [list\|info\|delete\|clean]` | 管理 Files API 远端图片 |
| `/save [节点ID]` | 导出节点为 Markdown |
| `/web <on\|off>` | 开关联网 |
| `/model [flash\|pro]` | 查看或切换模型 |
| `/a1` 这类写法 | 跳到指定节点 |

输入 `/` 弹出补全列表，`Tab` 循环、`Shift+Tab` 反向、`Enter` 补全。

---

## 快捷键

| 按键 | 作用 |
| --- | --- |
| `Enter` | 发送 |
| `Shift+Enter` | 输入框内换行 |
| `Esc` | 打断生成 |
| `Ctrl+C` | 退出 |
| `Ctrl+N` | 新建对话树 |
| `Ctrl+R` | 循环文件权限 |
| `Ctrl+W` | 开关联网 |
| `Ctrl+1..8` | 切换到第 N 棵树 |
| `Delete` | 侧栏选中节点时删除该节点（需确认） |
| `Ctrl+Backspace` / `Alt+Backspace` | 输入框内删词 |

---

## 数据文件

| 路径 | 内容 |
| --- | --- |
| `~/.mincli/config.toml` | 全部设置 |
| `~/.mincli/prompt.md` | 系统提示词 |
| `~/.mincli/trees/` | 每棵树一个 JSON + 索引 |
| `~/.mincli/attachments/` | 附件正文副本 |
| `~/.mincli/.env` | API Key |
| `~/.mincli/pricing.json` | 可选，覆盖价格与峰谷时段 |
| `~/Documents/mincli_Conversations/` | `/save` 导出的 Markdown |

---

## 与旧版的差异

这是一次收窄定位的重构：从「能挂载能力、能跑命令的编码代理」变成「只做对话与资料阅读/写作的工作台」。

- 移除 shell 执行、全部 MCP、全部工作流、模型注册与多 provider
- 移除客户端内查看/修改提示词、每棵树的能力挂载设置
- 移除 `/import`，改为拖入或粘贴路径生成引用块
- 旧文件 `~/.mincli/workflows.json`、`mcp_servers.json`、`models.json` 保留在磁盘上，但不再读取

## 限制

- 只支持 DeepSeek 的两个模型；`deepseek-v4-pro` 不支持图片
- `.doc` 需先转成 `.docx`
- Search 与 Fetch 按成功请求计费，失败不计费
