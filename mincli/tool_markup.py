"""清理模型在正文里泄漏的工具调用标记。

背景：DeepSeek 在某个工具没有被下发给模型时，偶尔会把内部的工具调用语法
当成正文输出。例如关闭联网后：

    <|DSML| | calls> <|DSML| | invoke name="web_search"> ...
    </|DSML| | invoke> </|DSML| | calls>

这些字符对用户没有意义，还会写进历史、被下一轮当成上下文。这里做两层兜底：

- :class:`ToolMarkupFilter`：流式过滤，普通正文立即透传，检测到标记后丢弃到
  行尾，不会出现「先显示一段乱码再消失」；
- :func:`strip_tool_markup`：对最终回答做同样的整段清理，保证落盘内容干净。

提示词与「本轮能力状态」能避免绝大多数情况，这里是最后一道保险。
"""

from __future__ import annotations

import re

# 标记开头：<|DSML|、</|DSML|、<|/DSML|、<|tool_calls|>、<|function_call|> 等
_START_RE = re.compile(
    r"</?\|[\s|]*(?:/)?\s*DSML|<\|[\s|]*(?:tool_calls?|function_call)[\s|]*>",
    re.IGNORECASE,
)
# 标记里可能出现的工具名：invoke name="web_search"（parameter name=... 不算）
_NAME_RE = re.compile(
    r'(?:invoke|function_call|function|tool)\s+[^>\n]{0,40}?name\s*=\s*"([A-Za-z_][\w]*)"',
    re.IGNORECASE,
)

# 尾巴里可能是「不完整标记」的部分，需要留到下一个 chunk 再判断
_TAIL_RE = re.compile(r"</?\|[^\n]{0,24}$")

# 整段回答只有标记时的替代文案
MARKUP_FALLBACK = "模型尝试调用当前未提供的工具，这次调用已被忽略。"


def _split_tail(data: str) -> tuple[str, str]:
    """拆出「可以立即输出」与「可能是标记开头、需要等待」的两部分。"""
    match = _TAIL_RE.search(data)
    if match:
        return data[: match.start()], data[match.start():]
    if data.endswith("</"):
        return data[:-2], data[-2:]
    if data.endswith("<"):
        return data[:-1], data[-1:]
    return data, ""


def _collect_names(text: str, names: list[str]) -> None:
    for name in _NAME_RE.findall(text):
        if name not in names:
            names.append(name)


class ToolMarkupFilter:
    """流式过滤器：把泄漏的工具调用标记从正文里去掉。"""

    def __init__(self, max_suppress: int = 20_000) -> None:
        self._buf = ""
        self._suppress = False
        self._suppressed = 0
        self._max = max_suppress
        self.leaked = False
        self.names: list[str] = []

    def feed(self, text: str) -> str:
        """喂入一个增量，返回可以展示的正文。"""
        if not text:
            return ""
        data = self._buf + text
        self._buf = ""
        out: list[str] = []

        while data:
            if self._suppress:
                newline = data.find("\n")
                if newline == -1:
                    self._suppressed += len(data)
                    if self._suppressed > self._max:
                        # 异常长的块：放弃抑制，避免吞掉整篇正文
                        self._suppress = False
                    return "".join(out)
                self._suppressed += newline
                data = data[newline + 1 :]
                self._suppress = False
                continue

            match = _START_RE.search(data)
            if match is None:
                head, tail = _split_tail(data)
                out.append(head)
                self._buf = tail
                return "".join(out)

            head = data[: match.start()]
            out.append(head)
            self.leaked = True
            _collect_names(data[match.start():], self.names)

            newline = data.find("\n", match.start())
            if newline == -1:
                self._suppress = True
                self._suppressed = len(data) - match.start()
                return "".join(out)
            data = data[newline + 1 :]

        return "".join(out)

    def flush(self) -> str:
        """结束本轮：输出暂存尾巴（若它是标记则丢弃）。"""
        tail, self._buf = self._buf, ""
        if tail and _START_RE.search(tail):
            self.leaked = True
            _collect_names(tail, self.names)
            return ""
        return tail


def strip_tool_markup(text: str) -> tuple[str, bool, list[str]]:
    """整段清理，返回 ``(干净文本, 是否含标记, 标记里出现的工具名)``。"""
    filter_ = ToolMarkupFilter()
    cleaned = filter_.feed(text or "") + filter_.flush()
    return cleaned, filter_.leaked, list(filter_.names)
