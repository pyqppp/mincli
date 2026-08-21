import asyncio
import logging
import os
import sys
import threading
from typing import Dict, List, Optional

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from mincli.config import EXEC_MAX_TIMEOUT, load_mcp_servers

# 静默 MCP SDK 的会话终止告警（部分远程 server 不支持 DELETE 会话终止，
# 关闭时会产生 "Session termination failed: 400/404" 噪音，不影响功能）
logging.getLogger("mcp").setLevel(logging.ERROR)

BUNDLED_NAME = "mincli"
CONNECT_TIMEOUT = 15
# 客户端调用超时须大于服务端 execute_command 的 timeout 上限（EXEC_MAX_TIMEOUT），
# 否则命令的“超时返回部分输出”路径会被客户端提前截断成“工具调用失败”
CALL_TIMEOUT = EXEC_MAX_TIMEOUT + 30


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _bundled_params() -> StdioServerParameters:
    if _frozen():
        return StdioServerParameters(command=sys.executable, args=["--mcp-server"])
    return StdioServerParameters(command=sys.executable, args=["-m", "mincli.mcp_server"])


def _external_params(server_name: str, cfg: dict) -> Optional[StdioServerParameters]:
    command = cfg.get("command")
    if not command:
        print(f"⚠ MCP server「{server_name}」缺少 command 配置，已跳过")
        return None
    env = cfg.get("env") or {}
    return StdioServerParameters(
        command=command,
        args=list(cfg.get("args") or []),
        env={**os.environ, **env},
    )


class McpToolClient:
    def __init__(self, external_config_path: Optional[str] = None):
        self.external_config_path = external_config_path
        self.ok: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Dict[str, Client] = {}
        self._http_clients: Dict[str, object] = {}  # name -> httpx2.AsyncClient（远程 server 专用）
        self._tool_owner: Dict[str, str] = {}
        self._tool_defs: List[dict] = []

    def start(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop")
        self._thread.start()
        try:
            self._run_coro(self._connect_all())
            self.ok = bool(self._clients)
        except Exception as e:
            print(f"MCP 连接失败: {e}")
            self.ok = False

    def tools(self) -> List[dict]:
        return list(self._tool_defs)

    def tool_names(self) -> set:
        return set(self._tool_owner.keys())

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

    def server_status(self) -> dict:
        """返回所有 server（内置 + 配置的第三方）的连接状态与工具数。"""
        names = {"mincli": "内置"}
        names.update({n: n for n in load_mcp_servers()})
        status = {}
        for name in names:
            client = self._clients.get(name)
            if client is not None:
                count = sum(1 for n in self._tool_owner if self._tool_owner[n] == name)
                status[name] = {"connected": True, "tools": count}
            else:
                status[name] = {"connected": False, "tools": 0}
        return status

    def reload(self) -> None:
        """断开全部连接并重新加载（读取最新配置文件）。"""
        if self._loop is None:
            raise RuntimeError("MCP 客户端未启动")
        try:
            self._run_coro(self._close_all(), timeout=10)
        except Exception:
            pass
        self._clients.clear()
        self._http_clients.clear()
        self._tool_owner.clear()
        self._tool_defs.clear()
        try:
            self._run_coro(self._connect_all())
            self.ok = bool(self._clients)
        except Exception as e:
            print(f"MCP 重连失败: {e}")
            self.ok = False

    def close(self) -> None:
        if self._loop is None:
            return
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
        servers = {"mincli": ("stdio", _bundled_params(), None)}
        for name, cfg in load_mcp_servers().items():
            if isinstance(cfg, dict) and cfg.get("url"):
                servers[name] = ("http", cfg["url"], cfg.get("headers") or {})
            else:
                params = _external_params(name, cfg)
                if params:
                    servers[name] = ("stdio", params, None)

        for name, (kind, target, headers) in servers.items():
            try:
                await asyncio.wait_for(
                    self._connect_one(name, kind, target, headers), timeout=CONNECT_TIMEOUT
                )
            except asyncio.TimeoutError:
                print(f"⚠ 连接 MCP server「{name}」超时，已跳过")
            except Exception as e:
                print(f"⚠ 连接 MCP server「{name}」失败: {e}，已跳过")

        await self._register_tools()

    async def _connect_one(
        self, name: str, kind: str, target, headers: Optional[dict] = None
    ) -> None:
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

    async def _register_tools(self) -> None:
        for name, client in self._clients.items():
            try:
                result = await asyncio.wait_for(client.list_tools(), timeout=CONNECT_TIMEOUT)
            except Exception as e:
                print(f"⚠ 获取「{name}」工具列表失败: {e}")
                continue
            for t in result.tools:
                if t.name in self._tool_owner:
                    print(f"⚠ 工具「{t.name}」与已有工具重名，来自「{name}」的工具已忽略")
                    continue
                self._tool_owner[t.name] = name
                self._tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.input_schema,
                    },
                })
        count = len(self._tool_defs)
        print(f"MCP 就绪：{len(self._clients)} 个 server，{count} 个工具")

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
