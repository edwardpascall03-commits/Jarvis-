import anthropic
import os
import json
import readline
import threading
import time
from dotenv import load_dotenv
from tools.obsidian import read_note, write_note, append_to_today, read_today, search_vault, list_vault
from tools.voice import listen_and_transcribe, speak
from tools.retrieval import store, retrieve, format_for_prompt
from tools.router import classify, haiku_chat, should_use_claude
from tools.executor import run_with_tools
from tools.memory import save_session, load_last_session, get_daily_dir
from tools.topic_manager import TopicManager
from tools.tool_selector import select_tools, log_tool_call, get_recent_tool_context, clear_expired_tools

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

topic_manager = TopicManager(daily_dir=get_daily_dir())

def load_profile(query: str = ""):
    with open("profile.json", "r") as f:
        profile = json.load(f)
    base = f"You are Jarvis, a personal AI assistant for {profile['name']}. Here is everything you need to know about them: {json.dumps(profile, indent=2)}"
    
    if query:
        memories = retrieve(query)
        base += format_for_prompt(memories)
    
    # Inject rolling window context
    base += topic_manager.get_rolling_context()
    base += get_recent_tool_context()

    return base

last_interaction = time.time()

def body_double_timer(interval_minutes=25):
    while True:
        time.sleep(60)  # check every minute
        inactive_minutes = (time.time() - last_interaction) / 60
        if inactive_minutes >= interval_minutes:
            print("\n[Jarvis]: Hey Edward — still on track? You've been quiet for a while.")

# Cost tracking
total_input_tokens = 0
total_output_tokens = 0
SONNET_INPUT_COST = 3.00 / 1_000_000
SONNET_OUTPUT_COST = 15.00 / 1_000_000
HAIKU_INPUT_COST = 0.80 / 1_000_000
HAIKU_OUTPUT_COST = 4.00 / 1_000_000
show_tokens = False


tools = [
    {
        "name": "append_to_today",
        "description": "Append a note or log entry to today's daily note in Obsidian.",
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
        "description": "Write a new note to the Obsidian vault.",
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
        "description": "Read any file from the Obsidian vault. Use when the user asks about project status, goals, training notes, dissertation ideas, or any document they've stored. Pass the relative path from vault root e.g. 'jarvis/jarvis_status_april2026.md'",
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
        "description": "Search the Obsidian vault for files matching a query when you don't know the exact filename. Use when the user asks for a document but doesn't give the exact name.",
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
        "description": "List all files in the Obsidian vault. Use when the user wants to know what documents are available or when search returns no results.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

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
    
    log_tool_call(name, inputs, result)
    return result

conversation_history = []

def chat(message):
    global total_input_tokens, total_output_tokens

    clear_expired_tools()

    conversation_history.append({"role": "user", "content": message})
    topic_manager.process_message("user", message, message_for_detection=message)

    # Cap history to last 12 entries (6 exchanges) for cost control
    # Full history still stored for autosave and Chroma
    capped_history = conversation_history[-12:] if len(conversation_history) > 12 else conversation_history

    category = classify(message)
    print(f"[Router: {category}]")

    # Select only tools needed for this message
    active_tools = select_tools(message, tools)

    system = load_profile(query=message)

    # Simple/retrieval tasks go to Haiku
    if not should_use_claude(category):
        reply = haiku_chat(
            capped_history,
            system,
            tools=active_tools if active_tools else None,
            handle_tool=handle_tool if active_tools else None
        )
        if reply:
            conversation_history.append({"role": "assistant", "content": reply})
            topic_manager.process_message("assistant", reply)
            store(f"User: {message}\nJarvis: {reply}", metadata={"source": "conversation"})
            if len(conversation_history) % 10 == 0:
                save_session(conversation_history)
            return reply
        print("[Router: Haiku failed, falling back to Sonnet]")

    # Complex/code tasks go to Sonnet
    reply, in_tokens, out_tokens = run_with_tools(
        client=client,
        model="claude-sonnet-4-6",
        messages=capped_history,
        system=system,
        tools=active_tools if active_tools else tools,  # fallback to all tools if selector returns empty
        handle_tool=handle_tool
    )

    total_input_tokens += in_tokens
    total_output_tokens += out_tokens

    if show_tokens:
        cost = in_tokens * SONNET_INPUT_COST + out_tokens * SONNET_OUTPUT_COST
        print(f"[Tokens — in: {in_tokens} out: {out_tokens} cost: £{cost:.5f}]")

    conversation_history.append({"role": "assistant", "content": reply})
    topic_manager.process_message("assistant", reply)
    store(f"User: {message}\nJarvis: {reply}", metadata={"source": "conversation"})

    if len(conversation_history) % 10 == 0:
        save_session(conversation_history)

    return reply

voice_mode = False

print("Jarvis online. Type 'quit' to exit, 'voice' to toggle voice mode.\n")

timer_thread = threading.Thread(target=body_double_timer, daemon=True)
timer_thread.start()

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
            save_session(conversation_history)
            topic_manager.close_session()
            print("\nJarvis: Shutting down, Sir.")
            break

    if user_input.lower() in ["quit", "exit"]:
        save_session(conversation_history)
        topic_manager.close_session() 
        total_cost = (total_input_tokens * SONNET_INPUT_COST +
                      total_output_tokens * SONNET_OUTPUT_COST)
        print(f"[Session cost: £{total_cost:.4f} | Total tokens: {total_input_tokens + total_output_tokens}]")
        print("Jarvis: Shutting down.")
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
        total_cost = (total_input_tokens * SONNET_INPUT_COST +
                      total_output_tokens * SONNET_OUTPUT_COST)
        print(f"[Session — input: {total_input_tokens} output: {total_output_tokens} total: £{total_cost:.4f}]")
        continue
    
    if user_input.lower() == "reset":
        conversation_history.clear()
        total_input_tokens = 0
        total_output_tokens = 0
        last_interaction = time.time()
        print("[Session reset — memory cleared, costs zeroed]")
        continue

    if not user_input:
        continue

    reply = chat(user_input)
    print(f"Jarvis: {reply}\n")

    if voice_mode:
        speak(reply)