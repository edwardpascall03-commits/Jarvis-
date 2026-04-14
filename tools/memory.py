import json
import os
from datetime import datetime

MEMORY_DIR = "memory"

def ensure_memory_dir():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)

def save_session(conversation_history):
    ensure_memory_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{MEMORY_DIR}/session_{timestamp}.json"
    
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
    
    with open(filename, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Session saved to {filename}")
    
def load_last_session():
    ensure_memory_dir()
    sessions = sorted(os.listdir(MEMORY_DIR))
    if not sessions:
        return []
    with open(f"{MEMORY_DIR}/{sessions[-1]}", "r") as f:
        return json.load(f)