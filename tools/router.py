import anthropic
import os
from tools.executor import run_with_tools

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ROUTING_PROMPT = """Classify this message into exactly one category. Respond with ONLY the category name.

SIMPLE: greetings, basic facts, casual chat, short writes, reading files, brief summaries
COMPLEX: reasoning, planning, analysis, advice, multi-step problems, detailed writing,
         anything asking to analyse/explain/evaluate/critique/compare/give feedback on content
CODE: writing, debugging, or explaining code
RETRIEVAL: asking about past conversations or notes

If the message contains verbs like: analyse, explain, evaluate, critique, compare, give feedback,
tell me what's missing/wrong/good — classify as COMPLEX regardless of how simple the rest seems.

Bias toward SIMPLE for anything answerable in one sentence or requiring only a short action.

Message: {message}

Category:"""

# ─── Action Detection ─────────────────────────────────────────────────────────

READ_KEYWORDS = ["read", "show me", "what's in", "open", "find", "search", "list", "what does", "tell me about", "look at", "check","what files", "do you have any", "is there a"]
WRITE_KEYWORDS = ["append", "log", "save", "write", "record", "add to", "note down", "update", "put in", "store"]

def detect_action(message: str) -> str:
    """
    Detect the action type from the message.
    Returns: READ, WRITE, or CHAT
    
    READ  → always Haiku, always vault read tools
    WRITE → complexity decides model, always write tools
    CHAT  → complexity decides model, Chroma decides tools
    """
    msg = message.lower()
    if any(w in msg for w in READ_KEYWORDS):
        return "READ"
    if any(w in msg for w in WRITE_KEYWORDS):
        return "WRITE"
    return "CHAT"

# ─── Classification ───────────────────────────────────────────────────────────

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

# ─── Haiku chat ───────────────────────────────────────────────────────────────

def haiku_chat(conversation_history: list, system_prompt: str, tools=None, handle_tool=None) -> tuple:
    """Returns (reply, input_tokens, output_tokens)."""
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
            reply, in_tok, out_tok = run_with_tools(
                client=_client,
                model="claude-haiku-4-5-20251001",
                messages=history,
                system=system_prompt,
                tools=tools,
                handle_tool=handle_tool,
                max_tokens=500
            )
            return reply, in_tok, out_tok

        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_prompt,
            messages=history
        )
        return response.content[0].text.strip(), response.usage.input_tokens, response.usage.output_tokens
    except Exception as e:
        print(f"[Haiku error: {e}]")
        return None, 0, 0

# ─── Routing decision ─────────────────────────────────────────────────────────

def should_use_claude(category: str) -> bool:
    """Returns True if Sonnet should handle this based on complexity alone."""
    return category in ["COMPLEX", "CODE"]