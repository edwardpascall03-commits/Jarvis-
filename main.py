import anthropic
import os
import json
from dotenv import load_dotenv
from tools.memory import save_session, load_last_session
from tools.obsidian import read_note, write_note, append_to_today, read_today
from tools.voice import listen_and_transcribe, speak
from tools.retrieval import store, retrieve, format_for_prompt

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def load_profile(query: str = ""):
    with open("profile.json", "r") as f:
        profile = json.load(f)
    base = f"You are Jarvis, a personal AI assistant for {profile['name']}. Here is everything you need to know about them: {json.dumps(profile, indent=2)}"
    
    if query:
        memories = retrieve(query)
        base += format_for_prompt(memories)
    
    return base
    
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
    }
]

def handle_tool(name, inputs):
    if name == "append_to_today":
        return append_to_today(inputs["content"])
    elif name == "read_today":
        return read_today()
    elif name == "write_note":
        return write_note(inputs["filename"], inputs["content"])
    elif name == "read_note":
        return read_note(inputs["filename"])
    return "Unknown tool."

conversation_history = []

def chat(message):
    conversation_history.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=load_profile(query=message), 
        tools=tools,
        messages=conversation_history
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        assistant_message = {"role": "assistant", "content": response.content}
        conversation_history.append(assistant_message)

        for block in response.content:
            if block.type == "tool_use":
                print(f"[Jarvis using tool: {block.name}]")
                result = handle_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        conversation_history.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=load_profile(),
            tools=tools,
            messages=conversation_history
        )

    reply = ""
    for block in response.content:
        if hasattr(block, "text"):
            reply += block.text

    conversation_history.append({"role": "assistant", "content": reply})
    
    # ← store the exchange after every reply
    store(f"User: {message}\nJarvis: {reply}", metadata={"source": "conversation"})
    
    return reply

from tools.voice import listen_and_transcribe, speak

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
        user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        save_session(conversation_history)
        print("Jarvis: Shutting down.")
        break

    if user_input.lower() == "voice":
        voice_mode = not voice_mode
        status = "on" if voice_mode else "off"
        print(f"[Voice mode {status}]")
        continue

    if not user_input:
        continue

    reply = chat(user_input)
    print(f"Jarvis: {reply}\n")

    if voice_mode:
        speak(reply)