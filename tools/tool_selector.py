import time
import anthropic
import os

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Tool TTL in seconds — drop from context after this long
TOOL_TTL = 600  # 10 minutes

# Tool history — stores recent tool calls with timestamps
_tool_history = []

SELECTOR_PROMPT = """You are a tool selector for an AI assistant. Given a user message, select which tools are needed to respond.

Available tools:
- append_to_today: writing/logging something to today's note
- read_today: reading today's note or schedule
- write_note: creating a new note in the vault
- read_note: reading a specific note by exact filename
- read_vault_file: reading any vault file by relative path
- search_vault: finding files when filename is unknown
- list_vault: listing all available vault files

Respond with ONLY a comma-separated list of tool names needed, or NONE if no tools are needed.

Examples:
"hello how are you" -> NONE
"what are my tasks today" -> read_today
"find my dissertation notes" -> search_vault,read_vault_file
"save this to my notes" -> write_note
"what files do you have" -> list_vault
"log that I trained today" -> append_to_today
"read jarvis_status_april2026.md" -> read_vault_file
"find jarvis_security_layer.md" -> search_vault,read_vault_file
"jarvis_security_layer.md" -> search_vault,read_vault_file

Message: {message}

Tools needed:"""

def select_tools(message: str, all_tools: list) -> list:
    """Return only the tools needed for this message."""
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": SELECTOR_PROMPT.format(message=message)}]
        )
        result = response.content[0].text.strip().upper()

        if result == "NONE" or not result:
            return []

        selected_names = [n.strip().lower() for n in result.split(",")]
        selected_tools = [t for t in all_tools if t["name"] in selected_names]

        if selected_tools:
            print(f"[Tools selected: {[t['name'] for t in selected_tools]}]")

        return selected_tools

    except Exception as e:
        print(f"[Tool selector error: {e}] — using all tools")
        return all_tools


def log_tool_call(name: str, inputs: dict, result: str):
    """Log a tool call to history with timestamp."""
    _tool_history.append({
        "tool": name,
        "inputs": inputs,
        "result": result[:500],  # cap result size
        "timestamp": time.time()
    })


def get_recent_tool_context(ttl: int = TOOL_TTL) -> str:
    """Return recently used tools formatted for system prompt injection."""
    now = time.time()
    recent = [t for t in _tool_history if now - t["timestamp"] < ttl]

    if not recent:
        return ""

    lines = []
    for t in recent[-5:]:  # max 5 recent tool results
        age_mins = int((now - t["timestamp"]) / 60)
        lines.append(f"- [{age_mins}m ago] {t['tool']}({t['inputs']}) → {t['result'][:200]}")

    return "\n\nRecently used tools:\n" + "\n".join(lines)


def clear_expired_tools(ttl: int = TOOL_TTL):
    """Remove tool history entries older than TTL."""
    global _tool_history
    now = time.time()
    _tool_history = [t for t in _tool_history if now - t["timestamp"] < ttl]
