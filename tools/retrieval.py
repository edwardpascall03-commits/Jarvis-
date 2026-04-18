import chromadb
from datetime import datetime
import hashlib
import uuid

# Persistent client — stores to disk so memory survives restarts
_client = chromadb.PersistentClient(path="./chroma_db")

# Long-term memory collection — persists across sessions
_collection = _client.get_or_create_collection(
    name="jarvis_memory",
    metadata={"hnsw:space": "cosine"}
)

# Per-session collection — deleted on quit
_SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"
_session_collection = _client.get_or_create_collection(
    name=_SESSION_ID,
    metadata={"hnsw:space": "cosine"}
)

# ─── Long-term memory ────────────────────────────────────────────────────────

def store(text: str, metadata: dict = None) -> None:
    """Store a piece of text in long-term Chroma memory."""
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
    """Retrieve relevant memories from long-term storage."""
    count = _collection.count()
    if count == 0:
        return []

    n = min(n_results, count)
    results = _collection.query(query_texts=[query], n_results=n)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

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

# ─── Session memory ───────────────────────────────────────────────────────────

def store_session(text: str, metadata: dict = None) -> None:
    """Store a piece of text in the current session collection."""
    if not text or not text.strip():
        return

    doc_id = hashlib.md5(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()
    meta = {"timestamp": datetime.now().isoformat()}
    if metadata:
        meta.update(metadata)

    _session_collection.add(
        documents=[text],
        metadatas=[meta],
        ids=[doc_id]
    )

def retrieve_session(query: str, n_results: int = 3) -> list[str]:
    """Retrieve relevant context from the current session only."""
    count = _session_collection.count()
    if count == 0:
        return []

    n = min(n_results, count)
    results = _session_collection.query(query_texts=[query], n_results=n)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    output = []
    for doc, meta in zip(docs, metas):
        ts = meta.get("timestamp", "unknown time")
        # Show time only for session context, not full date
        output.append(f"[{ts[11:16]}] {doc}")

    return output

def format_session_for_prompt(memories: list[str]) -> str:
    """Format session memories into a system prompt block."""
    if not memories:
        return ""
    joined = "\n".join(f"- {m}" for m in memories)
    return f"\n\nRelevant context from this session:\n{joined}"

def delete_session_collection() -> None:
    """Delete the session collection on quit — cleans up temporary memory."""
    try:
        _client.delete_collection(_SESSION_ID)
        print(f"[Session memory cleared: {_SESSION_ID}]")
    except Exception as e:
        print(f"[Session cleanup error: {e}]")