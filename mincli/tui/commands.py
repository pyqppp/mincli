"""斜杠命令规格：补全候选、/help、各处用法提示共用同一份数据。

结构是三级命令树：

    /set                一级命令
    /set thinking       二级子命令
    /set thinking on    三级取值

运行时才知道的候选项（对话树编号、工作流名、MCP server 名）不写死在这里，
由 ``providers`` 提供（见 ``runtime_providers``），补全时按需展开。

补全、/help、用法提示都从这棵树生成，保证三处格式一致：新增命令只改这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

from mincli.config import TEMPERATURE_MAX, TEMPERATURE_MIN


@dataclass(frozen=True)
class Cmd:
    """一条命令/子命令/取值。

    name          该级输入的名字（不含前导 "/"）
    desc          一句话说明（补全弹窗与 /help 里的灰字）
    args          该级之后的参数占位（显示用，如 "<提示词>"）
    aliases       等价写法（不含 "/"）
    children      静态下一级
    group         仅一级命令使用：/help 分组
    dyn           运行时候选项来源名（trees/workflows/servers）
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
DYN_LABEL = {"trees": "<编号>", "workflows": "<名称>", "servers": "<名称>"}

GROUP_ORDER = ("基本", "对话树", "工作流", "多模态", "多模型", "配置", "节点")

# 仅用于 /tree：静态子命令表达不了「/tree <编号> tools」这种组合
_TREE_TOOLS = Cmd("tools", "修改这棵树挂载的能力")

CMD_SPEC: tuple[Cmd, ...] = (
    Cmd("exit", "退出程序（自动保存会话）", aliases=("quit", "q", "e"), group="基本"),
    Cmd("clear", "清空当前会话（当前树的全部节点）", aliases=("c",), group="基本"),
    Cmd("compact", "把当前分支全部对话压缩成摘要并新建摘要节点", group="基本"),
    Cmd("help", "显示命令帮助", aliases=("h",), group="基本"),
    Cmd("view", "用编辑器打开当前回答", group="基本"),
    Cmd("import", "导入文件/网页/图片（图片转为待发送图片，并给出 token 估算）",
        args="<文件路径或URL> [...]", group="多模态",
        children=(
            Cmd("clear", "清除已导入、待发送的内容", aliases=("c",)),
        )),
    Cmd("files", "管理已上传的 Files API 图片文件（默认最新在前）", group="多模态",
        children=(
            Cmd("list", "列出最近上传的文件（含序号与容量统计）", args="[条数]",
                aliases=("ls",)),
            Cmd("info", "查询单个文件信息", args="<ID|序号>"),
            Cmd("delete", "删除文件（同步清理引用它的对话树）", args="<ID|序号>",
                aliases=("rm", "del")),
            Cmd("clean", "清理不再被任何对话树引用的文件（需确认）"),
        )),
    Cmd("tree", "列出全部对话树（编号、颜色、节点数、挂载能力）", group="对话树",
        children=(
            Cmd("new", "新建对话树并选择挂载的能力"),
            Cmd("delete", "删除整棵树（需确认，编号不再复用）", args="<编号>",
                aliases=("rm", "del"), dyn="trees"),
        ),
        dyn="trees", dyn_children=(_TREE_TOOLS,),
        extra=(("/tree <编号> tools", _TREE_TOOLS.desc),)),
    Cmd("wf", "把某次/一连串操作保存为可复用工作流", aliases=("workflow",), group="工作流",
        children=(
            Cmd("list", "列出全部工作流", aliases=("ls",)),
            Cmd("show", "查看工作流规范", args="<名>", dyn="workflows"),
            Cmd("save", "把当前节点提炼为工作流并长期保存", args="<名> [起点节点ID]"),
            Cmd("use", "挂载到下一次输入（一次性）", args="<名>", dyn="workflows"),
            Cmd("stop", "取消已挂载的工作流"),
            Cmd("run", "立即按工作流执行", args="<名> [键=值...]", dyn="workflows"),
            Cmd("edit", "修订工作流规范（不带要求时用编辑器打开）", args="<名> [修改要求]",
                dyn="workflows"),
            Cmd("rename", "重命名工作流", args="<旧名> <新名>"),
            Cmd("delete", "删除工作流", args="<名>", aliases=("rm", "del"), dyn="workflows"),
        )),
    Cmd("model", "查看/注册模型配置", group="多模型",
        children=(
            Cmd("list", "列出内置与已注册模型", aliases=("ls",)),
            Cmd("register", "注册 OpenAI 兼容模型", args="<模型名> <URL> [-p provider] [-k key_var]"),
        )),
    Cmd("mcp", "管理第三方 MCP server", group="配置",
        children=(
            Cmd("list", "列出已配置的 server", aliases=("ls",)),
            Cmd("add", "新增 server（本地命令或远程 URL）", args="<名称> <命令|URL> [参数...]"),
            Cmd("remove", "移除 server", args="<名称>", aliases=("rm", "del"), dyn="servers"),
            Cmd("reload", "重新连接所有 server"),
        )),
    Cmd("set", "修改运行配置", group="配置",
        children=(
            Cmd("system", "修改系统提示词", args="<提示词>"),
            Cmd("temp", f"设置温度（{TEMPERATURE_MIN}~{TEMPERATURE_MAX}）", args="<值>"),
            Cmd("model", "切换模型（flash 支持图片理解）", args="<flash|pro|模型名>"),
            Cmd("thinking", "开关思考模式", args="<on|off>",
                children=(Cmd("on", "开启思考模式"), Cmd("off", "关闭思考模式"))),
            Cmd("effort", "设置推理强度", args="<low|high|max>",
                children=(
                    Cmd("low", "更省时间的低强度推理"),
                    Cmd("high", "默认的高强度推理"),
                    Cmd("max", "最大推理强度（最慢）"),
                )),
            Cmd("audit", "设置审核层级", args="<1-4>"),
            Cmd("workspace", "设置命令执行默认工作目录", args="<路径>"),
            Cmd("detail", "设置图片清晰度（low 走内联真正省 token）",
                args="<low|auto|high|original>",
                children=(
                    Cmd("low", "最低清晰度：本地图片改走内联，file_id 会忽略 detail"),
                    Cmd("auto", "自动选择（默认，优先上传 Files API 复用）"),
                    Cmd("high", "高清晰度（等价 original）"),
                    Cmd("original", "原图，最清晰"),
                )),
            Cmd("file_confirm", "文件写入前是否弹窗确认", args="<on|off>",
                children=(Cmd("on", "写入前弹窗确认"), Cmd("off", "直接写入不确认"))),
            Cmd("show", "显示当前配置"),
        )),
    Cmd("info", "查看节点详情", args="[节点ID]", group="节点"),
    Cmd("up", "返回父节点", group="节点"),
    Cmd("home", "跳回根节点", group="节点"),
    Cmd("full", "切换全览模式（节点树全宽显示）", aliases=("f",), group="节点"),
    Cmd("save", "导出节点为 Markdown 文件", args="[节点ID]", group="节点"),
    Cmd("delete", "删除节点及其所有子节点（需确认）", args="<节点ID> [...]", group="节点"),
)

TOP_LEVEL: tuple[Cmd, ...] = CMD_SPEC


# ---------------- 运行时候选项 ----------------

def runtime_providers(ctrl) -> dict[str, Callable[[], list[tuple[str, str]]]]:
    """把控制器里的运行时状态包成候选项来源（控制器不可用时返回空）。

    每次按键都会调用一次，因此这里只读内存态：尤其不要用 tree_summary()
    （未加载的树会逐个读盘解析），树编号的说明只用编号与「当前」标记。
    """

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

    def workflows() -> list[tuple[str, str]]:
        out = []
        for wf in ctrl.wf_list():
            steps = wf.get("steps", 0)
            desc = f"{wf.get('goal') or '工作流'}（{steps} 步）"
            out.append((str(wf.get("name", "")), desc))
        return [item for item in out if item[0]]

    def servers() -> list[tuple[str, str]]:
        return [(name, "已配置的 MCP server") for name in ctrl.external_servers()]

    return {"trees": trees, "workflows": workflows, "servers": servers}


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
    """按 "/set temp" 取静态节点（不展开动态候选项）。"""
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
    """把路径前缀与名字拼成用法文本：base="/set" + "temp" + "<值>"。"""
    text = f"{base} {name}".strip() if base else f"/{name}"
    return f"{text} {args}" if args else text


def command_block(path: str) -> str:
    """某条命令的分行说明（Markdown 列表），用于 /wf 这类无参输出。"""
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
    trailing = text.endswith(" ")
    tokens = text.split()
    if not tokens:
        return None

    node: Optional[Cmd] = None
    path: list[str] = []
    consumed = 0
    children = list(TOP_LEVEL)
    for token in tokens:
        # 一级命令带前导 "/"（"/set"），往下各级都是裸名字（"thinking"）
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
