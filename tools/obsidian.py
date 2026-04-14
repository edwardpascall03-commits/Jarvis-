import os
from datetime import datetime

VAULT_PATH = "/home/eddy/Documents/Ai-personal/Jarvis/jarvis-vault"

def get_today_path():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(VAULT_PATH, f"{today}.md")

def read_note(filename):
    path = os.path.join(VAULT_PATH, filename)
    if not os.path.exists(path):
        return f"Note '{filename}' not found."
    with open(path, "r") as f:
        return f.read()

def write_note(filename, content):
    path = os.path.join(VAULT_PATH, filename)
    with open(path, "w") as f:
        f.write(content)
    return f"Note '{filename}' saved."

def append_to_today(content):
    path = get_today_path()
    timestamp = datetime.now().strftime("%H:%M")
    with open(path, "a") as f:
        f.write(f"\n## {timestamp}\n{content}\n")
    return f"Appended to today's log."

def read_today():
    path = get_today_path()
    if not os.path.exists(path):
        return "No log for today yet."
    with open(path, "r") as f:
        return f.read()