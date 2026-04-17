import os
from datetime import datetime

VAULT_PATH = "/home/eddy/Documents/Ai-personal/Jarvis/jarvis-vault"

def get_today_path():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(VAULT_PATH, "daily", f"{today}.md")

def read_note(filename: str) -> str:
    """Read a note by relative path from vault root."""
    path = os.path.join(VAULT_PATH, filename)
    if not os.path.exists(path):
        return f"Note '{filename}' not found."
    with open(path, "r") as f:
        return f.read()

def write_note(filename: str, content: str) -> str:
    """Write a note to the vault, creating subdirectories if needed."""
    path = os.path.join(VAULT_PATH, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"Note '{filename}' saved."

def append_to_today(content: str) -> str:
    path = get_today_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M")
    with open(path, "a") as f:
        f.write(f"\n## {timestamp}\n{content}\n")
    return "Appended to today's log."

def read_today() -> str:
    path = get_today_path()
    if not os.path.exists(path):
        return "No log for today yet."
    with open(path, "r") as f:
        return f.read()

def search_vault(query: str) -> str:
    """
    Search vault for files matching a query string.
    Searches filenames and folder names.
    Returns a list of matching relative paths.
    """
    query_lower = query.lower().replace(" ", "_")
    matches = []

    for root, dirs, files in os.walk(VAULT_PATH):
        # Skip hidden folders like .obsidian
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if not filename.endswith(".md"):
                continue
            if query_lower in filename.lower() or query.lower() in filename.lower():
                full_path = os.path.join(root, filename)
                relative = os.path.relpath(full_path, VAULT_PATH)
                matches.append(relative)

    if not matches:
        return f"No files found matching '{query}'."
    return "Found files:\n" + "\n".join(f"- {m}" for m in matches)

def list_vault() -> str:
    """List all markdown files in the vault with their relative paths."""
    files = []
    for root, dirs, filenames in os.walk(VAULT_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in filenames:
            if filename.endswith(".md"):
                full_path = os.path.join(root, filename)
                relative = os.path.relpath(full_path, VAULT_PATH)
                files.append(relative)

    if not files:
        return "Vault is empty."
    return "Vault contents:\n" + "\n".join(f"- {f}" for f in sorted(files))