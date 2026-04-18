import chromadb
import json
import os

# Separate Chroma collection for tools only
_client = chromadb.PersistentClient(path="./chroma_db")
_tool_collection = _client.get_or_create_collection(
    name="jarvis_tools",
    metadata={"hnsw:space": "cosine"}
)

def register_tools(tools: list):
    """
    Embed all tool definitions into the tool vector DB.
    Call once on startup. Re-registers if definitions have changed.
    """
    # Clear and re-register to pick up any definition changes
    existing = _tool_collection.get()
    if existing["ids"]:
        _tool_collection.delete(ids=existing["ids"])

    documents = []
    metadatas = []
    ids = []

    for tool in tools:
        # Embed the description + name so semantic search finds the right tool
        doc = f"{tool['name']}: {tool['description']}"
        documents.append(doc)
        metadatas.append({
            "name": tool["name"],
            "schema": json.dumps(tool["input_schema"]),
            "full_definition": json.dumps(tool)
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
    Returns a list of tool dicts ready to pass to the API.
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
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for meta, distance in zip(metadatas, distances):
        # Cosine distance threshold — only include tools that are actually relevant
        # Distance of 0 = identical, 1 = completely unrelated
        # Threshold of 0.6 means "reasonably relevant"
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
