import asyncio
import logging
import os
import sys
import threading
from typing import Callable, Dict, List, Optional

from mincli.config import EXEC_MAX_TIMEOUT, load_mcp_servers

# 静默 MCP SDK 的会话终止告警（部分远程 server 不支持 DELETE 会话终止，
# 关闭时会产生 "Session termination failed: 400/404" 噪音，不影响功能）
logging.getLogger("mcp").setLevel(logging.ERROR)

BUNDLED_NAME = "mincli"
CONNECT_TIMEOUT = 15
# 客户端调用超时须大于服务端 execute_command 的 timeout 上限（EXEC_MAX_TIMEOUT），
# 否则命令的“超时返回部分输出”路径会被客户端提前截断成“工具调用失败”
CALL_TIMEOUT = EXEC_MAX_TIMEOUT + 30
# 发送消息前等待后台连接完成的上限（正常情况启动后 1~2 秒内已完成）
READY_TIMEOUT = CONNECT_TIMEOUT + 10

# 内部工具：由 mincli 主进程调用（如打断时 cancel_command），不暴露给模型
INTERNAL_TOOLS = frozenset({"cancel_command"})


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _mcp_sdk():
    """惰性导入 MCP SDK（约 0.2s）。

    SDK 只在连接/调用时用得到，而连接现在跑在后台线程里——放在模块顶层会
    白白拖慢首屏。
    """
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    return Client, StdioServerParameters, stdio_client, streamable_http_client


def _bundled_params():
    """内置 server 的启动参数（命令行）：mincli 自身以 --mcp-server 再跑一个进程。"""
    _, StdioServerParameters, _, _ = _mcp_sdk()
    if _frozen():
        return StdioServerParameters(command=sys.executable, args=["--mcp-server"])
    return StdioServerParameters(command=sys.executable, args=["-m", "mincli.mcp_server"])


def _external_params(server_name: str, cfg: dict, log: Callable[[str], None]):
    """第三方 stdio server 的启动参数；缺少 command 时返回 None 并告警。"""
    command = cfg.get("command")
    if not command:
        log(f"MCP server「{server_name}」缺少 command 配置，已跳过")
        return None
    _, StdioServerParameters, _, _ = _mcp_sdk()
    env = cfg.get("env") or {}
    return StdioServerParameters(
        command=command,
        args=list(cfg.get("args") or []),
        env={**os.environ, **env},
    )


class McpToolClient:
    def __init__(
        self,
        external_config_path: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.external_config_path = external_config_path
        self.ok: bool = False
        self._log_sink: Callable[[str], None] = log or print
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_fut = None  # 后台连接任务（concurrent.futures.Future）
        self._clients: Dict[str, Client] = {}
        self._http_clients: Dict[str, object] = {}  # name -> httpx2.AsyncClient（远程 server 专用）
        self._tool_owner: Dict[str, str] = {}
        self._tool_defs: List[dict] = []

    def _log(self, message: str) -> None:
        """输出一条连接/就绪信息。

        走回调而非 print：TUI 起来之后这些消息来自后台线程，直接写 stdout
        会把界面冲乱（由 UI 侧转成通知）。
        """
        try:
            self._log_sink(message)
        except Exception:
            pass

    def start(self) -> None:
        """启动事件循环线程并**在后台**连接所有 server，立即返回。

        首屏不再等待 MCP：调用方先渲染界面，连接完成前用 ready/connecting
        查询状态，真正要用工具时再 wait_ready()。
        """
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop")
        self._thread.start()
        self._schedule_connect()

    @property
    def connecting(self) -> bool:
        """后台连接是否仍在进行。"""
        return self._ready_fut is not None and not self._ready_fut.done()

    @property
    def ready(self) -> bool:
        """后台连接是否已结束（不等于连接成功，成功与否看 ok）。"""
        return self._ready_fut is not None and self._ready_fut.done()

    def wait_ready(self, timeout: Optional[float] = READY_TIMEOUT) -> bool:
        """阻塞等待后台连接结束，返回是否至少连上一个 server。

        供发送消息的路径调用（在工作线程里执行，不阻塞 UI 事件循环）。
        超时不再等待——本轮退回“无 MCP 工具”，避免把发送卡死。
        """
        fut = self._ready_fut
        if fut is None:
            return False
        try:
            fut.result(timeout=timeout)
        except Exception:
            return self.ok
        return self.ok

    def _schedule_connect(self) -> None:
        """把连接任务投递到事件循环线程（不等待结果）。"""
        self._ready_fut = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)

    def tools(self) -> List[dict]:
        return list(self._tool_defs)

    def tool_names(self) -> set:
        return set(self._tool_owner.keys())

    def tool_owner(self, name: str) -> Optional[str]:
        """工具名 → 提供它的 server 名（未知工具返回 None）。"""
        return self._tool_owner.get(name)

    def tools_by_server(self) -> Dict[str, List[str]]:
        """server 名 → 该 server 暴露给模型的工具名（不含内部工具）。

        多对话树的能力勾选界面按这个分组列出可选工具；内置 server 用
        BUNDLED_NAME（"mincli"）作为键，对应「系统工具」那一组。
        """
        grouped: Dict[str, List[str]] = {}
        for d in self._tool_defs:
            name = d.get("function", {}).get("name")
            if not name:
                continue
            owner = self._tool_owner.get(name) or BUNDLED_NAME
            grouped.setdefault(owner, []).append(name)
        return grouped

    def configured_servers(self) -> List[str]:
        """已配置的第三方 server 名（不含内置），按配置顺序。"""
        return list(load_mcp_servers().keys())

    def call(self, name: str, arguments: dict, timeout: int = CALL_TIMEOUT) -> str:
        owner = self._tool_owner.get(name)
        if owner is None:
            return f"未知工具: {name}"
        client = self._clients.get(owner)
        if client is None or not self.ok:
            return "MCP 客户端未就绪，无法调用工具"
        try:
            result = self._run_coro(client.call_tool(name, arguments), timeout=timeout)
        except Exception as e:
            return f"工具调用失败: {e}"
        texts = []
        for block in getattr(result, "content", []) or []:
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", ""))
            else:
                texts.append(str(block))
        content = "\n".join(texts)
        if getattr(result, "is_error", False):
            content = f"[工具执行失败]\n{content}"
        return content

    def cancel_running(self) -> bool:
        """请求内置 server 终止当前正在执行的命令（供用户打断）。

        第三方 server 没有该内部工具，返回 False（上层仍会停止后续流程，
        但无法杀掉第三方 server 自己派生的进程）。
        """
        owner = self._tool_owner.get("cancel_command")
        if owner is None or not self.ok:
            return False
        client = self._clients.get(owner)
        if client is None:
            return False
        try:
            # 独立超时：打断路径不能被长命令拖住（server 端在另一线程杀进程）
            self._run_coro(client.call_tool("cancel_command", {}), timeout=10)
            return True
        except Exception:
            return False

    def server_status(self) -> dict:
        """返回所有 server（内置 + 配置的第三方）的连接状态与工具数。"""
        names = {"mincli": "内置"}
        names.update({n: n for n in load_mcp_servers()})
        status = {}
        for name in names:
            client = self._clients.get(name)
            if client is not None:
                count = sum(
                    1 for n in self._tool_owner
                    if self._tool_owner[n] == name and n not in INTERNAL_TOOLS
                )
                status[name] = {"connected": True, "tools": count}
            else:
                status[name] = {"connected": False, "tools": 0}
        return status

    def reload(self) -> None:
        """断开全部连接并重新加载（读取最新配置文件）。

        同样在后台执行：重连要花几秒，不能卡住 UI 线程；期间 connecting 为
        True，调用方据此提示「正在重连」。
        """
        if self._loop is None:
            raise RuntimeError("MCP 客户端未启动")
        self._ready_fut = asyncio.run_coroutine_threadsafe(self._reconnect(), self._loop)

    async def _reconnect(self) -> None:
        """关闭旧连接后重新连接全部 server（运行在事件循环线程）。"""
        try:
            await self._close_all()
        except Exception:
            pass
        self._clients = {}
        self._http_clients = {}
        self._tool_owner = {}
        self._tool_defs = []
        self.ok = False
        await self._connect_all()

    def close(self) -> None:
        if self._loop is None:
            return
        # 连接可能仍在后台进行：先取消再关，避免 _close_all 与 _connect_one
        # 并发改动客户端表（遍历时被改会抛 RuntimeError）
        fut = self._ready_fut
        if fut is not None and not fut.done():
            fut.cancel()
            try:
                fut.result(timeout=2)
            except Exception:
                pass
        try:
            self._run_coro(self._close_all(), timeout=10)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=2)
        except Exception:
            pass
        self._loop = None
        self._ready_fut = None
        self.ok = False
        self._clients.clear()
        self._http_clients.clear()
        self._tool_owner.clear()
        self._tool_defs.clear()

    def _run_coro(self, coro, timeout: Optional[float] = None):
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("MCP 事件循环未运行")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    async def _connect_all(self) -> None:
        """并发连接所有 server，再并发拉取工具列表。

        连接是纯 I/O 等待（子进程握手 / 远程 HTTP），串行时四个 server 要 4s，
        并发后可省掉大部分等待。任一 server 慢或失败都不影响其他 server。
        """
        servers: Dict[str, tuple] = {}
        for name, cfg in load_mcp_servers().items():
            if isinstance(cfg, dict) and cfg.get("url"):
                servers[name] = ("http", cfg["url"], cfg.get("headers") or {})
            else:
                params = _external_params(name, cfg, self._log)
                if params:
                    servers[name] = ("stdio", params, None)
        # 内置 server 排最前：工具重名时以内置为准（与串行版顺序一致）
        servers = {"mincli": ("stdio", _bundled_params(), None), **servers}

        try:
            await asyncio.gather(*[
                self._connect_guarded(name, kind, target, headers)
                for name, (kind, target, headers) in servers.items()
            ])
            self.ok = bool(self._clients)
            await self._register_tools()
        except Exception as e:  # pragma: no cover - 兜底，避免后台任务静默失败
            self._log(f"MCP 连接失败: {e}")
            self.ok = False

    async def _connect_guarded(
        self, name: str, kind: str, target, headers: Optional[dict] = None
    ) -> None:
        """连接单个 server，超时/异常都只告警不抛出（不影响其他 server）。"""
        try:
            await asyncio.wait_for(
                self._connect_one(name, kind, target, headers), timeout=CONNECT_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._log(f"连接 MCP server「{name}」超时，已跳过")
        except Exception as e:
            self._log(f"连接 MCP server「{name}」失败: {e}，已跳过")

    async def _connect_one(
        self, name: str, kind: str, target, headers: Optional[dict] = None
    ) -> None:
        Client, _, stdio_client, streamable_http_client = _mcp_sdk()
        if kind == "http":
            if headers:
                try:
                    import httpx2
                except ImportError as e:
                    raise RuntimeError(
                        f"需要请求头但未安装 httpx2（请升级 mcp SDK）: {e}"
                    ) from e
                http_client = httpx2.AsyncClient(headers=headers)
                self._http_clients[name] = http_client
                client = Client(streamable_http_client(target, http_client=http_client))
            else:
                client = Client(streamable_http_client(target))
        else:
            client = Client(stdio_client(target))
        await client.__aenter__()
        self._clients[name] = client

    async def _list_tools(self, name: str):
        client = self._clients[name]
        return await asyncio.wait_for(client.list_tools(), timeout=CONNECT_TIMEOUT)

    async def _register_tools(self) -> None:
        """并发拉取各 server 的工具列表，按 server 顺序登记（先到先得）。

        结果用「整体替换」的方式落库：发送线程可能正在读 self.llm_tools，
        就地 append 会让它读到半成品列表。
        """
        names = list(self._clients)
        results = await asyncio.gather(
            *[self._list_tools(name) for name in names], return_exceptions=True
        )
        owner: Dict[str, str] = {}
        defs: List[dict] = []
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                self._log(f"获取「{name}」工具列表失败: {result}")
                continue
            for t in result.tools:
                if t.name in owner:
                    self._log(f"工具「{t.name}」与已有工具重名，来自「{name}」的工具已忽略")
                    continue
                owner[t.name] = name
                if t.name in INTERNAL_TOOLS:
                    # 内部工具：保留 owner 映射供客户端调用，但不加入模型工具列表
                    continue
                defs.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.input_schema,
                    },
                })
        self._tool_owner = owner
        self._tool_defs = defs
        self._log(f"MCP 就绪：{len(names)} 个 server，{len(defs)} 个工具")

    async def _close_all(self) -> None:
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        # 释放远程 server 专用的 http 客户端（streamable_http_client 不接管外部传入的 client）
        for http_client in self._http_clients.values():
            try:
                await http_client.aclose()
            except Exception:
                pass
        self._http_clients.clear()
