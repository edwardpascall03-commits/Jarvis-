import chromadb
from datetime import datetime
import hashlib

# Persistent client — stores to disk so memory survives restarts
_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection(
    name="jarvis_memory",
    metadata={"hnsw:space": "cosine"}
)

def store(text: str, metadata: dict = None) -> None:
    """Store a piece of text in Chroma with optional metadata."""
    if not text or not text.strip():
        return

    doc_id = hashlib.md5(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()

    meta = {"timestamp": datetime.now().isoformat()}
    if metadata:
        meta.update(metadata)

    _collection.add(
        documents=[text],
        metadatas=[meta],
        ids=[doc_id]
    )

def retrieve(query: str, n_results: int = 3) -> list[str]:
    """Retrieve the most semantically relevant stored memories for a query."""
    count = _collection.count()
    if count == 0:
        return []

    # Don't request more results than exist
    n = min(n_results, count)

    results = _collection.query(
        query_texts=[query],
        n_results=n
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    # Return docs paired with their timestamp for context
    output = []
    for doc, meta in zip(docs, metas):
        ts = meta.get("timestamp", "unknown time")
        output.append(f"[{ts[:10]}] {doc}")

    return output

def format_for_prompt(memories: list[str]) -> str:
    """Format retrieved memories into a system prompt block."""
    if not memories:
        return ""
    joined = "\n".join(f"- {m}" for m in memories)
    return f"\n\nRelevant context from past sessions:\n{joined}"