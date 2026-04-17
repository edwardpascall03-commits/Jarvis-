import json
import os
from datetime import datetime

MEMORY_DIR = "memory"

def get_daily_dir():
    """Get today's log directory, creating it if needed."""
    today = datetime.now().strftime("%Y-%m-%d")
    daily_dir = os.path.join(MEMORY_DIR, today)
    os.makedirs(daily_dir, exist_ok=True)
    return daily_dir

def save_session(conversation_history):
    daily_dir = get_daily_dir()
    timestamp = datetime.now().strftime("%H-%M-%S")

    # JSON save (existing behaviour, now in daily folder)
    json_filename = os.path.join(daily_dir, f"session_{timestamp}.json")
    serialisable = []
    for message in conversation_history:
        if isinstance(message["content"], str):
            serialisable.append(message)
        elif isinstance(message["content"], list):
            cleaned = []
            for block in message["content"]:
                if hasattr(block, "type"):
                    cleaned.append({"type": block.type, "text": getattr(block, "text", "")})
                else:
                    cleaned.append(block)
            serialisable.append({"role": message["role"], "content": cleaned})

    with open(json_filename, "w") as f:
        json.dump(serialisable, f, indent=2)

    # Markdown log — human readable, goes into Obsidian vault later
    md_filename = os.path.join(daily_dir, f"session_{timestamp}.md")
    with open(md_filename, "w") as f:
        f.write(f"# Jarvis Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for message in serialisable:
            if isinstance(message["content"], str) and message["content"].strip():
                role = "**You**" if message["role"] == "user" else "**Jarvis**"
                f.write(f"{role}: {message['content']}\n\n")

    print(f"[Session saved — {daily_dir}/session_{timestamp}]")

def load_last_session():
    if not os.path.exists(MEMORY_DIR):
        return []

    # Find most recent daily folder
    daily_dirs = sorted([
        d for d in os.listdir(MEMORY_DIR)
        if os.path.isdir(os.path.join(MEMORY_DIR, d))
    ])

    if not daily_dirs:
        return []

    latest_dir = os.path.join(MEMORY_DIR, daily_dirs[-1])

    # Find most recent JSON session in that folder
    sessions = sorted([
        f for f in os.listdir(latest_dir)
        if f.endswith(".json")
    ])

    if not sessions:
        return []

    with open(os.path.join(latest_dir, sessions[-1]), "r") as f:
        return json.load(f)