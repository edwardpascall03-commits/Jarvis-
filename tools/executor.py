def run_with_tools(client, model, messages, system, tools, handle_tool, max_tokens=1000):
    """
    Shared tool-use loop. Works with any client that implements the Anthropic
    messages.create interface — swap in an Ollama or OpenAI-compatible adapter
    and this function needs no changes.

    Returns (reply_text, input_tokens, output_tokens).
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=tools,
        messages=messages
    )

    total_in = response.usage.input_tokens
    total_out = response.usage.output_tokens

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[Jarvis using tool: {block.name}]")
                result = handle_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results}
        ]

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages
        )
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens

    reply = "".join(block.text for block in response.content if hasattr(block, "text"))
    return reply, total_in, total_out
