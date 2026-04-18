import anthropic
import os
import json
import readline
import threading
import time
from dotenv import load_dotenv
from tools.obsidian import read_note, write_note, append_to_today, read_today, search_vault, list_vault
from tools.voice import listen_and_transcribe, speak
from tools.retrieval import (
    store, retrieve, format_for_prompt,
    store_session, retrieve_session, format_session_for_prompt,
    delete_session_collection
)
from tools.router import classify, haiku_chat, should_use_claude
from tools.executor import run_with_tools
from tools.memory import save_session, load_last_session, get_daily_dir
from tools.topic_manager import TopicManager
from tools.tool_registry import register_tools, get_tools_for_message
from tools.memory_curator import register_profile_chunks, get_relevant_chunks, format_profile_for_prompt, curate_context
from tools.router import classify, haiku_chat, should_use_claude, detect_action

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

topic_manager = TopicManager(daily_dir=get_daily_dir())

# Cost tracking
total_input_tokens = 0
total_output_tokens = 0
total_haiku_input_tokens = 0
total_haiku_output_tokens = 0
SONNET_INPUT_COST = 3.00 / 1_000_000
SONNET_OUTPUT_COST = 15.00 / 1_000_000
HAIKU_INPUT_COST = 0.80 / 1_000_000
HAIKU_OUTPUT_COST = 4.00 / 1_000_000
MONTHLY_BUDGET = 5.00
show_tokens = False

tools = [
    {
        "name": "append_to_today",
        "description": "Append content to today's daily note. Use when the user says: 'append to today', 'log this', 'add to today', 'record that'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to append."}
            },
            "required": ["content"]
        }
    },
    {
        "name": "read_today",
        "description": "Read today's daily note from Obsidian.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "write_note",
        "description": "Create and save a new markdown file to the Obsidian vault. Use only when the user explicitly asks to save, create, or write something new.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename including .md extension."},
                "content": {"type": "string", "description": "Content to write."}
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "read_note",
        "description": "Read an existing note from the Obsidian vault by filename.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename including .md extension."}
            },
            "required": ["filename"]
        }
    },
    {
        "name": "read_vault_file",
        "description": "Read any file from the Obsidian vault by relative path. Use for project status, goals, training notes, dissertation ideas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Relative path from vault root including .md extension e.g. 'jarvis/jarvis_status_april2026.md'"}
            },
            "required": ["filename"]
        }
    },
    {
        "name": "search_vault",
        "description": "Search the Obsidian vault for files matching a query when the exact filename is unknown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term to find matching files"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_vault",
        "description": "List all files in the Obsidian vault.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# Register tools into Chroma vector DB after tools list is defined
register_tools(tools)
register_profile_chunks()

def handle_tool(name, inputs):
    if name == "append_to_today":
        result = append_to_today(inputs["content"])
    elif name == "read_today":
        result = read_today()
    elif name == "write_note":
        result = write_note(inputs["filename"], inputs["content"])
    elif name == "read_note":
        result = read_note(inputs["filename"])
    elif name == "read_vault_file":
        result = read_note(inputs["filename"])
    elif name == "search_vault":
        result = search_vault(inputs["query"])
    elif name == "list_vault":
        result = list_vault()
    else:
        result = "Unknown tool."
    return result

def load_profile(query: str = ""):
    if not query:
        # No query — just inject identity and communication
        chunks = get_relevant_chunks("general greeting")
        return format_profile_for_prompt(chunks)

    # Get relevant profile chunks semantically
    chunks = get_relevant_chunks(query)
    profile_text = format_profile_for_prompt(chunks)

    # Get memory context
    session_memories = retrieve_session(query)
    long_term_memories = retrieve(query)

    # Curate — Haiku decides what actually gets injected
    curated = curate_context(query, profile_text, session_memories, long_term_memories)

    return f"You are Jarvis, a personal AI assistant. Always address the user as Sir. You have tools to read and write files in the user's Obsidian vault — use them when asked, never claim you cannot access files.\n\n{curated}"

conversation_history = []

def chat(message):
    global total_input_tokens, total_output_tokens, total_haiku_input_tokens, total_haiku_output_tokens

    conversation_history.append({"role": "user", "content": message})
    topic_manager.process_message("user", message, message_for_detection=message)

    # Keep only last 4 entries (2 exchanges) for immediate API coherence
    # Session Chroma handles deeper context now
    capped_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history

    # Layer 1 — complexity classification
    category = classify(message)
    print(f"[Router: {category}]")

    # Layer 2 — action detection (free, keyword-based)
    action = detect_action(message)
    print(f"[Action: {action}]")

    # Layer 3 — tool selection based on action type
    if action == "READ":
        # Reading never needs Sonnet — force Haiku, inject vault tools
        active_tools = [t for t in tools if t["name"] in [
            "search_vault", "read_vault_file", "read_note", "read_today", "list_vault"
        ]]
        force_haiku = True

    elif action == "WRITE":
        # Writing always needs write tools — complexity decides the model
        active_tools = [t for t in tools if t["name"] in ["append_to_today", "write_note"]]
        force_haiku = False

    else:  # CHAT
        # Ambiguous — use Chroma semantic selection, fallback for RETRIEVAL
        active_tools = get_tools_for_message(message, n_results=3)
        if not active_tools and category == "RETRIEVAL":
            active_tools = [t for t in tools if t["name"] in ["search_vault", "read_vault_file", "read_note"]]
            print(f"[Tool fallback: vault tools injected for RETRIEVAL]")
        force_haiku = False

    system = load_profile(query=message)

    # Route to Haiku or Sonnet
    if force_haiku or not should_use_claude(category):
        result = haiku_chat(
            capped_history,
            system,
            tools=active_tools if active_tools else [],
            handle_tool=handle_tool
        )
        if result:
            reply, haiku_in, haiku_out = result
            total_haiku_input_tokens += haiku_in
            total_haiku_output_tokens += haiku_out

            if show_tokens:
                cost = haiku_in * HAIKU_INPUT_COST + haiku_out * HAIKU_OUTPUT_COST
                print(f"[Haiku tokens — in: {haiku_in} out: {haiku_out} cost: £{cost:.5f}]")

            conversation_history.append({"role": "assistant", "content": reply})
            topic_manager.process_message("assistant", reply)
            exchange = f"User: {message}\nJarvis: {reply}"
            store_session(exchange, metadata={"source": "conversation"})
            store(exchange, metadata={"source": "conversation"})
            if len(conversation_history) % 10 == 0:
                save_session(conversation_history)
            return reply
        print("[Router: Haiku failed, falling back to Sonnet]")

    # Sonnet path — complex reasoning, code, or failed Haiku
    reply, in_tokens, out_tokens = run_with_tools(
        client=client,
        model="claude-sonnet-4-6",
        messages=capped_history,
        system=system,
        tools=active_tools if active_tools else tools,
        handle_tool=handle_tool
    )

    total_input_tokens += in_tokens
    total_output_tokens += out_tokens

    if show_tokens:
        cost = in_tokens * SONNET_INPUT_COST + out_tokens * SONNET_OUTPUT_COST
        print(f"[Tokens — in: {in_tokens} out: {out_tokens} cost: £{cost:.5f}]")

    conversation_history.append({"role": "assistant", "content": reply})
    topic_manager.process_message("assistant", reply)
    exchange = f"User: {message}\nJarvis: {reply}"
    store_session(exchange, metadata={"source": "conversation"})
    store(exchange, metadata={"source": "conversation"})
    if len(conversation_history) % 10 == 0:
        save_session(conversation_history)
    return reply

def shutdown():
    """Clean shutdown — save session, clear temp memory."""
    save_session(conversation_history)
    topic_manager.close_session()
    delete_session_collection()
    sonnet_cost = total_input_tokens * SONNET_INPUT_COST + total_output_tokens * SONNET_OUTPUT_COST
    haiku_cost = total_haiku_input_tokens * HAIKU_INPUT_COST + total_haiku_output_tokens * HAIKU_OUTPUT_COST
    total_cost = sonnet_cost + haiku_cost
    total_tokens = total_input_tokens + total_output_tokens + total_haiku_input_tokens + total_haiku_output_tokens
    print(f"[Session cost: £{total_cost:.4f} (Sonnet £{sonnet_cost:.4f} / Haiku £{haiku_cost:.4f}) | Total tokens: {total_tokens}]")
    print("Jarvis: Shutting down, Sir.")

voice_mode = False

print("Jarvis online. Type 'quit' to exit, 'voice' to toggle voice mode.\n")

while True:
    if voice_mode:
        user_input = listen_and_transcribe()
        if user_input == "__TEXT_MODE__":
            voice_mode = False
            print("[Switched to text mode]")
            continue
        print(f"You said: {user_input}")
        if not user_input:
            print("[Nothing detected, try again]")
            continue
    else:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            shutdown()
            break

    if user_input.lower() in ["quit", "exit"]:
        shutdown()
        break

    if user_input.lower() == "voice":
        voice_mode = not voice_mode
        status = "on" if voice_mode else "off"
        print(f"[Voice mode {status}]")
        continue

    if user_input.lower() == "tokens":
        show_tokens = not show_tokens
        print(f"[Token display {'on' if show_tokens else 'off'}]")
        continue

    if user_input.lower() == "cost":
        sonnet_cost = total_input_tokens * SONNET_INPUT_COST + total_output_tokens * SONNET_OUTPUT_COST
        haiku_cost = total_haiku_input_tokens * HAIKU_INPUT_COST + total_haiku_output_tokens * HAIKU_OUTPUT_COST
        total_cost = sonnet_cost + haiku_cost
        budget_used = (total_cost / MONTHLY_BUDGET) * 100
        print(f"[Session — Sonnet in: {total_input_tokens} out: {total_output_tokens} £{sonnet_cost:.4f} | Haiku in: {total_haiku_input_tokens} out: {total_haiku_output_tokens} £{haiku_cost:.4f} | Total: £{total_cost:.4f} | Budget: {budget_used:.1f}%]")
        continue

    if user_input.lower() == "reset":
        conversation_history.clear()
        total_input_tokens = 0
        total_output_tokens = 0
        total_haiku_input_tokens = 0
        total_haiku_output_tokens = 0
        print("[Session reset — memory cleared, costs zeroed]")
        continue

    if not user_input:
        continue

    reply = chat(user_input)
    print(f"Jarvis: {reply}\n")

    if voice_mode:
        speak(reply)