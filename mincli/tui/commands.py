"""斜杠命令规格：补全候选、/help、各处用法提示共用同一份数据。

结构是三级命令树：

    /file               一级命令
    /file read          二级子命令
    /tree delete 1      子命令 + 动态候选

运行时才知道的候选项（对话树编号）不写死在这里，由 ``providers`` 提供
（见 ``runtime_providers``），补全时按需展开。

补全、/help、用法提示都从这棵树生成，保证三处格式一致：新增命令只改这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional



@dataclass(frozen=True)
class Cmd:
    """一条命令/子命令/取值。

    name          该级输入的名字（不含前导 "/"）
    desc          一句话说明（补全弹窗与 /help 里的灰字）
    args          该级之后的参数占位（显示用，如 "<提示词>"）
    aliases       等价写法（不含 "/"）
    children      静态下一级
    group         仅一级命令使用：/help 分组
    dyn           运行时候选项来源名（trees）
    dyn_children  动态候选项自身的下一级
    extra         /help 里额外补的 (用法, 说明) 行（静态树表达不了的组合）
    """

    name: str
    desc: str
    args: str = ""
    aliases: tuple[str, ...] = ()
    children: tuple["Cmd", ...] = ()
    group: str = ""
    dyn: str = ""
    dyn_children: tuple["Cmd", ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

    @property
    def takes_more(self) -> bool:
        """接受该候选项后是否还要继续输入（决定补全时是否补一个空格）。"""
        return bool(self.args or self.children or self.dyn)


# 动态候选项在用法里的占位写法
DYN_LABEL = {"trees": "<编号>"}

GROUP_ORDER = ("基本", "对话树", "文件与附件", "联网", "模型", "节点跳转")

# 仅用于 /tree：静态子命令表达不了「/tree <编号> tools」这种组合
CMD_SPEC: tuple[Cmd, ...] = (
    Cmd("help", "显示命令帮助", aliases=("h",), group="基本"),
    Cmd("clear", "清空当前对话树（全部节点）", aliases=("c",), group="基本"),
    Cmd("compact", "把当前分支压缩成摘要并新建摘要节点", group="基本"),
    Cmd("exit", "退出程序（自动保存会话）", aliases=("quit", "q"), group="基本"),
    Cmd("tree", "对话树：列出全部、新建、删除", group="对话树",
        children=(
            Cmd("list", "列出全部对话树（默认，也可只输 /tree）", aliases=("ls",)),
            Cmd("new", "新建一棵空对话树（所有树完全一样）"),
            Cmd("delete", "删除整棵树（需确认，编号不再复用）", args="<编号>",
                aliases=("rm", "del"), dyn="trees"),
        ),
        dyn="trees"),
    Cmd("up", "返回父节点", group="对话树"),
    Cmd("home", "跳回根节点", group="对话树"),
    Cmd("delete", "删除节点及其所有子节点（需确认）", args="<节点ID> [...]", group="对话树"),
    Cmd("recall", "开关对话树检索工具（默认关）", args="<on|off>", group="对话树",
        children=(Cmd("on", "允许模型查询历史节点"), Cmd("off", "关闭历史节点查询"))),
    Cmd("file", "查看/切换文件权限（关闭/只读/写需确认/写直通）",
        args="<off|read|confirm|direct>", group="文件与附件",
        children=(
            Cmd("off", "关闭：只提供附件续读工具"),
            Cmd("read", "只读：允许读取本地文件与列目录"),
            Cmd("confirm", "写需确认：读取 + 每次写文件前确认"),
            Cmd("direct", "写直通：读取 + 写文件不再确认"),
        )),
    Cmd("files", "管理已上传的 Files API 图片文件（默认最新在前）", group="文件与附件",
        children=(
            Cmd("list", "列出最近上传的文件（含序号与容量统计）", args="[条数]",
                aliases=("ls",)),
            Cmd("info", "查询单个文件信息", args="<ID|序号>"),
            Cmd("delete", "删除文件（同步清理引用它的对话树）", args="<ID|序号>",
                aliases=("rm", "del")),
            Cmd("clean", "清理不再被任何对话树引用的文件（需确认）"),
        )),
    Cmd("save", "导出节点为 Markdown 文件", args="[节点ID]", group="文件与附件"),
    Cmd("web", "开关联网（搜索 + 网页抓取）", args="<on|off>", group="联网",
        children=(Cmd("on", "开启联网"), Cmd("off", "关闭联网"))),
    Cmd("model", "在 flash / pro 之间切换", args="<flash|pro>", group="模型",
        children=(
            Cmd("flash", "deepseek-flash：支持图片理解"),
            Cmd("pro", "deepseek-v4-pro：不支持图片"),
        )),
) + (
    Cmd(
        "a1",
        "跳到指定节点（支持 a1 / b2 这类编号，输入 /a1 即进入）",
        group="节点跳转",
    ),
)

TOP_LEVEL: tuple[Cmd, ...] = CMD_SPEC


# ---------------- 运行时候选项 ----------------

def runtime_providers(ctrl) -> dict[str, Callable[[], list[tuple[str, str]]]]:
    """把控制器里的运行时状态包成候选项来源（控制器不可用时返回空）。"""

    def trees() -> list[tuple[str, str]]:
        numbers = ctrl.tree_numbers()
        active = ctrl.active_number
        out = []
        for n in numbers:
            desc = f"对话树 {n}"
            if n == active:
                desc += "（当前）"
            out.append((str(n), desc))
        return out

    return {"trees": trees}


def _dyn_items(source: str, providers: Optional[Mapping[str, Callable[[], list[tuple[str, str]]]]]):
    if not source or providers is None:
        return []
    fn = providers.get(source)
    if fn is None:
        return []
    try:
        return list(fn())
    except Exception:
        # 运行时状态取不到（控制器未就绪等）不该影响输入
        return []


# ---------------- 静态查询（用法提示、/help） ----------------

def _find_in(children: Iterable[Cmd], token: str) -> Optional[Cmd]:
    token = (token or "").lower()
    if not token:
        return None
    for child in children:
        if token == child.name.lower() or token in (a.lower() for a in child.aliases):
            return child
    return None


def node_at(path: str) -> Optional[Cmd]:
    """按 "/file read" 取静态节点（不展开动态候选项）。"""
    parts = [p for p in (path or "").split() if p]
    if not parts:
        return None
    first = parts[0].lstrip("/")
    node = _find_in(TOP_LEVEL, first)
    for part in parts[1:]:
        if node is None:
            return None
        node = _find_in(node.children, part)
    return node


def usage_line(path: str) -> str:
    """一行用法提示（notify 用）。

    含子命令的给「<子命令>（名称清单）」，叶子给完整用法。
    """
    node = node_at(path)
    if node is None:
        return f"用法: {path}"
    names = [c.name for c in node.children]
    if node.dyn and not node.args:
        # args 已写明该位置填什么（如 <编号>），就不再重复列动态占位
        names.append(DYN_LABEL.get(node.dyn, "<值>"))
    if names and node.args:
        # 取值枚举（如 <on|off>）在 args 里已经写全，不再重复列名字
        if not node.dyn and all(n in node.args for n in names):
            return f"用法: {path} {node.args}"
        return f"用法: {path} {node.args} ｜ 子命令：{'/'.join(names)}"
    if names:
        return f"用法: {path} <子命令>（{'/'.join(names)}）"
    args = f" {node.args}" if node.args else ""
    return f"用法: {path}{args}"


def _usage_of(base: str, name: str, args: str = "") -> str:
    """把路径前缀与名字拼成用法文本：base="/file" + "read"。"""
    text = f"{base} {name}".strip() if base else f"/{name}"
    return f"{text} {args}" if args else text


def command_block(path: str) -> str:
    """某条命令的分行说明（Markdown 列表），用于无参命令输出。"""
    node = node_at(path)
    if node is None:
        return usage_line(path)
    base = "/" + path.strip("/")
    lines = [f"**用法**：`{usage_line(path).split('用法: ', 1)[-1]}`", "", node.desc]
    if node.children or node.extra:
        lines.append("")
        for child in node.children:
            lines.append(f"- `{_usage_of(base, child.name, child.args)}` — {child.desc}")
        for usage, desc in node.extra:
            lines.append(f"- `{usage}` — {desc}")
    return "\n".join(lines)


def help_markdown() -> str:
    """/help 正文（从同一份规格生成，保证与补全提示一致）。"""
    lines = ["**帮助**", ""]
    for group in GROUP_ORDER:
        items = [c for c in TOP_LEVEL if c.group == group]
        if not items:
            continue
        lines.append(f"**{group}**")
        for cmd in items:
            names = (cmd.name,) + cmd.aliases
            if cmd.children:
                # 既接受参数又有子命令（如 /import）时两种写法都列出
                usages = [f"/{n} {cmd.args}" for n in names] if cmd.args else []
                usages += [f"/{n} <子命令>" for n in names]
                lines.append("- " + "、".join(f"`{u}`" for u in usages) + f" — {cmd.desc}")
                for child in cmd.children:
                    lines.append(f"  - `{_usage_of('/' + cmd.name, child.name, child.args)}` — {child.desc}")
            else:
                usages = [_usage_of("", n, cmd.args) for n in names]
                lines.append("- " + "、".join(f"`{u}`" for u in usages) + f" — {cmd.desc}")
            for usage, desc in cmd.extra:
                lines.append(f"  - `{usage}` — {desc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------- 补全 ----------------

@dataclass
class Candidate:
    """一个候选：补全后写进输入框的文本 + 展示用的一行。"""

    insert: str
    usage: str
    desc: str


@dataclass
class Completion:
    """当前输入对应的补全状态。"""

    title: str
    desc: str
    candidates: list[Candidate] = field(default_factory=list)


def _expanded(cmd: Optional[Cmd], providers) -> list[Cmd]:
    """静态子命令 + 运行时候选项（动态项继承 dyn_children）。"""
    children: list[Cmd] = list(cmd.children) if cmd is not None else list(TOP_LEVEL)
    source = cmd.dyn if cmd is not None else ""
    for value, desc in _dyn_items(source, providers):
        children.append(Cmd(name=str(value), desc=desc, children=cmd.dyn_children))
    return children


def complete(text: str, providers=None) -> Optional[Completion]:
    """解析输入框内容，返回补全状态；None 表示不该弹窗。"""
    if not text.startswith("/"):
        return None
    tokens = text.split()
    if not tokens:
        return None

    node: Optional[Cmd] = None
    path: list[str] = []
    consumed = 0
    children = list(TOP_LEVEL)
    for token in tokens:
        # 一级命令带前导 "/"（"/file"），往下各级都是裸名字（"read"）
        lookup = token.lstrip("/") if consumed == 0 else token
        child = _find_in(children, lookup)
        if child is None:
            break
        node = child
        path.append(child.name)
        children = _expanded(child, providers)
        consumed += 1

    base = "/" + " ".join(path) if path else ""
    word = "" if consumed == len(tokens) else tokens[consumed]
    if consumed == 0:
        word = word.lstrip("/")
    word = word.lower()

    candidates: list[Candidate] = []
    for child in children:
        if word and not child.name.lower().startswith(word):
            continue
        insert = _usage_of(base, child.name)
        if child.takes_more:
            insert += " "
        candidates.append(
            Candidate(insert=insert, usage=_usage_of(base, child.name, child.args), desc=child.desc)
        )

    if node is None and not candidates:
        return None  # 未知命令：不弹窗
    if node is not None:
        title = base + (f" {node.args}" if node.args else "")
        return Completion(title=title, desc=node.desc, candidates=candidates)
    return Completion(title="命令补全", desc="", candidates=candidates)
