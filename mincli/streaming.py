"""流式调用 DeepSeek API。

本模块不依赖 Rich：渲染完全交给调用方（Textual TUI / 纯文本前端），
通过 on_chunk 回调逐增量接收内容增量，由调用方决定如何展示。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from openai import OpenAI

from mincli.helpers import estimate_tokens
from mincli.models import StreamResult


def stream_response(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    user_question: str = "",
    thinking_enabled: bool = False,
    reasoning_effort: str = "high",
    tools: Optional[List[Dict]] = None,
    silent: bool = False,
    on_chunk: Optional[Callable[[str, str], None]] = None,
) -> StreamResult:
    """流式请求 DeepSeek，返回聚合结果。

    on_chunk(content_delta, reasoning_delta) 每收到一个增量回调一次，
    用于 UI 实时渲染。silent=True 或未传 on_chunk 时只聚合、不回调。

    出错时返回 StreamResult(error=...)，不抛异常。
    """
    estimated_input = estimate_tokens(messages)
    full_content = ""
    reasoning_text = ""
    usage_input = 0
    usage_output = 0
    accumulated_tool_calls: Dict[int, Dict] = {}

    def _process_chunk(chunk):
        nonlocal full_content, reasoning_text, usage_input, usage_output
        if getattr(chunk, "usage", None):
            usage_input = chunk.usage.prompt_tokens
            usage_output = chunk.usage.completion_tokens
        delta = chunk.choices[0].delta
        content_delta = delta.content or ""
        reasoning_delta = getattr(delta, "reasoning_content", None) or ""
        if reasoning_delta:
            reasoning_text += reasoning_delta
        if content_delta:
            full_content += content_delta
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": "",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.id:
                    accumulated_tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
        if on_chunk is not None:
            on_chunk(content_delta, reasoning_delta)

    try:
        extra_body: Dict = {}
        if thinking_enabled:
            extra_body["thinking"] = {"type": "enabled"}
            extra_body["reasoning_effort"] = reasoning_effort
        else:
            extra_body["thinking"] = {"type": "disabled"}

        kwargs = dict(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            extra_body=extra_body,
        )
        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)
        for chunk in response:
            _process_chunk(chunk)

        if accumulated_tool_calls:
            return StreamResult(
                tool_calls=list(accumulated_tool_calls.values()),
                reasoning=reasoning_text,
                input_tokens=usage_input,
                output_tokens=usage_output,
            )

        if usage_input == 0 and usage_output == 0:
            input_tokens = estimated_input
            output_tokens = estimate_tokens(
                [{"role": "assistant", "content": full_content}]
            )
        else:
            input_tokens = usage_input
            output_tokens = usage_output

        return StreamResult(
            content=full_content,
            reasoning=reasoning_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except Exception as e:
        return StreamResult(error=str(e))
