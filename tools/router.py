import anthropic
import os
from tools.executor import run_with_tools

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ROUTING_PROMPT = """Classify this message into exactly one category. Respond with ONLY the category name.

SIMPLE: greetings, reminders, basic facts, casual chat
COMPLEX: reasoning, planning, analysis, advice, multi-step problems
CODE: writing, debugging, or explaining code
RETRIEVAL: asking about past conversations or notes

Bias toward SIMPLE for anything answerable in one sentence.

Message: {message}

Category:"""

def classify(message: str) -> str:
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": ROUTING_PROMPT.format(message=message)}]
        )
        category = response.content[0].text.strip().upper()
        if category not in ["SIMPLE", "COMPLEX", "CODE", "RETRIEVAL"]:
            return "COMPLEX"
        return category
    except Exception as e:
        print(f"[Classifier error: {e}]")
        return "COMPLEX"

def haiku_chat(conversation_history: list, system_prompt: str, tools=None, handle_tool=None) -> str:
    try:
        history = []
        for m in conversation_history:
            if isinstance(m["content"], str) and m["content"].strip():
                history.append(m)
            elif isinstance(m["content"], list):
                # Keep tool results, skip raw tool_use blocks
                text_parts = [b for b in m["content"] 
                             if isinstance(b, dict) and b.get("type") == "tool_result"]
                if text_parts:
                    history.append({"role": m["role"], "content": text_parts})

        if tools and handle_tool:
            reply, _, _ = run_with_tools(
                client=_client,
                model="claude-haiku-4-5-20251001",
                messages=history,
                system=system_prompt,
                tools=tools,
                handle_tool=handle_tool,
                max_tokens=500
            )
            return reply

        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_prompt,
            messages=history
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[Haiku error: {e}]")
        return None

def should_use_claude(category: str) -> bool:
    return category in ["COMPLEX", "CODE"]