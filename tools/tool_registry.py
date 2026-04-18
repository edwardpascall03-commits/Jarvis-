import chromadb
import json
import os
import hashlib

# Separate Chroma collection for tools only
_client = chromadb.PersistentClient(path="./chroma_db")
_tool_collection = _client.get_or_create_collection(
    name="jarvis_tools",
    metadata={"hnsw:space": "cosine"}
)

def _tools_hash(tools: list) -> str:
    return hashlib.md5(json.dumps(tools, sort_keys=True).encode()).hexdigest()

def register_tools(tools: list):
    """
    Embed all tool definitions into the tool vector DB.
    Only re-embeds if tool definitions have changed since last startup.
    """
    current_hash = _tools_hash(tools)

    # Check if already embedded with same content
    existing = _tool_collection.get(include=["metadatas"])
    if existing["ids"] and existing["metadatas"][0].get("content_hash") == current_hash:
        print(f"[Tool registry: up to date ({len(tools)} tools)]")
        return

    # Content changed or first run — re-embed
    if existing["ids"]:
        _tool_collection.delete(ids=existing["ids"])

    documents = []
    metadatas = []
    ids = []

    for tool in tools:
        doc = f"{tool['name']}: {tool['description']}"
        documents.append(doc)
        metadatas.append({
            "name": tool["name"],
            "schema": json.dumps(tool["input_schema"]),
            "full_definition": json.dumps(tool),
            "content_hash": current_hash
        })
        ids.append(tool["name"])

    _tool_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"[Tool registry: {len(tools)} tools embedded]")


def get_tools_for_message(message: str, n_results: int = 3) -> list:
    """
    Semantically retrieve the most relevant tool definitions for a message.
    Returns empty list if message is unlikely to need tools.
    """
    count = _tool_collection.count()
    if count == 0:
        return []

    n = min(n_results, count)

    results = _tool_collection.query(
        query_texts=[message],
        n_results=n,
        include=["metadatas", "distances"]
    )

    tools = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        print(f"[Debug — {meta['name']}: {distance:.2f}]")
        if distance < 0.6:
            tool_def = json.loads(meta["full_definition"])
            tools.append(tool_def)
            print(f"[Tool selected: {meta['name']} (distance: {distance:.2f})]")

    return tools


def get_tool_by_name(name: str) -> dict | None:
    """Fetch a specific tool definition by name — used as fallback."""
    try:
        result = _tool_collection.get(ids=[name], include=["metadatas"])
        if result["ids"]:
            return json.loads(result["metadatas"][0]["full_definition"])
    except Exception:
        pass
    return None