from typing import Optional, List, Dict

from openai import OpenAI
from rich.live import Live
from rich.markdown import Markdown

from mincli.helpers import estimate_tokens
from mincli.models import StreamResult
from mincli.render import console


def stream_response(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float,
    user_question: str,
    thinking_enabled: bool = False,
    reasoning_effort: str = "high",
    tools: Optional[List[Dict]] = None,
    silent: bool = False,
) -> StreamResult:
    estimated_input = estimate_tokens(messages)
    full_content = ""
    reasoning_text = ""
    usage_input = 0
    usage_output = 0
    accumulated_tool_calls: Dict[int, Dict] = {}

    def _process_chunk(chunk):
        nonlocal full_content, reasoning_text, usage_input, usage_output
        if hasattr(chunk, "usage") and chunk.usage:
            usage_input = chunk.usage.prompt_tokens
            usage_output = chunk.usage.completion_tokens
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
        if delta.content:
            full_content += delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                if tc.id:
                    accumulated_tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

    try:
        extra_body = {}
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

        if silent:
            for chunk in response:
                _process_chunk(chunk)
        else:
            with Live(auto_refresh=False, console=console) as live:
                header = f"**你:**\n{user_question}\n\n"
                initial_display = header + "**DeepSeek:** "
                live.update(Markdown(initial_display), refresh=True)

                for chunk in response:
                    _process_chunk(chunk)

                    display = header
                    if reasoning_text:
                        display += "[dim]**DeepSeek 思考过程:**\n " + reasoning_text + "[/dim]\n\n"
                    display += f"**DeepSeek:** {full_content}"
                    live.update(Markdown(display), refresh=True)

                final_display = header
                if reasoning_text:
                    final_display += "[dim]**DeepSeek 思考过程:**\n " + reasoning_text + "[/dim]\n\n"
                final_display += f"**DeepSeek:** {full_content}"
                live.update(Markdown(final_display), refresh=True)

        if accumulated_tool_calls:
            return StreamResult(tool_calls=list(accumulated_tool_calls.values()), reasoning=reasoning_text,
                                input_tokens=usage_input, output_tokens=usage_output)

        if usage_input == 0 and usage_output == 0:
            input_tokens = estimated_input
            output_tokens = estimate_tokens([{"role": "assistant", "content": full_content}])
        else:
            input_tokens = usage_input
            output_tokens = usage_output

        return StreamResult(content=full_content, reasoning=reasoning_text,
                            input_tokens=input_tokens, output_tokens=output_tokens)

    except Exception as e:
        console.print(f"[red]API 调用失败: {e}[/red]")
        return StreamResult()
