import os
import hashlib
import json
import anthropic
import chromadb
from datetime import datetime

VAULT_PATH = "/home/eddy/Documents/Ai-personal/Jarvis/jarvis-vault"
FILE_READ_LIMIT = 3000  # characters per chunk
SUMMARY_MAX_TOKENS = 200  # Haiku summary length

_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma_client.get_or_create_collection(
    name="jarvis_memory",
    metadata={"hnsw:space": "cosine"}
)
_api_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─── Hashing ──────────────────────────────────────────────────────────────────

def get_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_stored_hash(relative_path: str) -> str | None:
    """Get the stored hash for a file from Chroma metadata."""
    try:
        results = _collection.get(
            where={"$and": [{"type": {"$eq": "file_summary"}}, {"source": {"$eq": relative_path}}]},
            include=["metadatas"]
        )
        if results["ids"]:
            return results["metadatas"][0].get("hash")
    except Exception:
        pass
    return None

def get_stored_summary_id(relative_path: str) -> str | None:
    """Get the Chroma document ID for a stored file summary."""
    try:
        results = _collection.get(
            where={"$and": [{"type": {"$eq": "file_summary"}}, {"source": {"$eq": relative_path}}]},
            include=["metadatas"]
        )
        if results["ids"]:
            return results["ids"][0]
    except Exception:
        pass
    return None

# ─── Summarisation ────────────────────────────────────────────────────────────

def summarise_file(content: str, filename: str) -> str:
    """Use Haiku to generate a concise summary of a vault file."""
    # Truncate very large files before summarising
    truncated = content[:6000] if len(content) > 6000 else content
    truncation_note = " [file truncated for summary]" if len(content) > 6000 else ""

    try:
        response = _api_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": f"Summarise this file in 2-4 sentences. Be specific about what it contains — dates, decisions, topics covered. File: {filename}\n\n{truncated}"
            }]
        )
        return response.content[0].text.strip() + truncation_note
    except Exception as e:
        print(f"[Vault sync: summarisation error for {filename}: {e}]")
        return f"File: {filename} — summary unavailable"

# ─── Sync ─────────────────────────────────────────────────────────────────────

def walk_vault():
    """Yield all markdown files in the vault as relative paths."""
    for root, dirs, files in os.walk(VAULT_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.endswith(".md"):
                full_path = os.path.join(root, filename)
                relative = os.path.relpath(full_path, VAULT_PATH)
                yield relative, full_path

def sync_vault():
    """
    Scan vault and update Chroma with summaries of new or modified files.
    Only processes files that have changed since last sync.
    Removes summaries for deleted files.
    """
    current_files = set()
    new_count = 0
    updated_count = 0

    for relative_path, full_path in walk_vault():
        current_files.add(relative_path)

        # Skip daily logs — too volatile, not useful as static summaries
        if relative_path.startswith("daily/"):
            continue

        current_hash = get_file_hash(full_path)
        stored_hash = get_stored_hash(relative_path)

        if stored_hash == current_hash:
            continue  # unchanged — skip

        # File is new or modified
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            continue  # skip empty files

        summary = summarise_file(content, relative_path)

        doc_id = f"file_summary_{hashlib.md5(relative_path.encode()).hexdigest()}"

        # Upsert — handles both new and updated files
        try:
            _collection.upsert(
                documents=[summary],
                metadatas=[{
                    "type": "file_summary",
                    "source": relative_path,
                    "hash": current_hash,
                    "date_synced": datetime.now().isoformat(),
                    "full_content_available": "true"
                }],
                ids=[doc_id]
            )
            if stored_hash is None:
                new_count += 1
            else:
                updated_count += 1
        except Exception as e:
            print(f"[Vault sync: storage error for {relative_path}: {e}]")

    # Remove summaries for deleted files
    removed_count = remove_deleted_files(current_files)

    total = new_count + updated_count + removed_count
    if total > 0:
        print(f"[Vault sync: {new_count} new, {updated_count} updated, {removed_count} removed]")
    else:
        print(f"[Vault sync: up to date]")

def remove_deleted_files(current_files: set):
    """Remove Chroma entries for files that no longer exist in the vault."""
    removed = 0
    try:
        results = _collection.get(
            where={"type": {"$eq": "file_summary"}},
            include=["metadatas"]
        )
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            source = meta.get("source", "")
            if source and source not in current_files:
                _collection.delete(ids=[doc_id])
                removed += 1
    except Exception as e:
        print(f"[Vault sync: cleanup error: {e}]")
    return removed

# ─── Chunked reading ──────────────────────────────────────────────────────────

def read_note_chunked(filename: str, chunk: int = 0) -> str:
    """
    Read a file in chunks to avoid massive token injections.
    Returns the chunk content plus a continuation hint if more exists.
    """
    if not filename.endswith(".md"):
        filename += ".md"

    path = os.path.join(VAULT_PATH, filename)
    if not os.path.exists(path):
        return f"Note '{filename}' not found."

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    start = chunk * FILE_READ_LIMIT
    end = start + FILE_READ_LIMIT
    excerpt = content[start:end]

    if not excerpt:
        return f"No more content in '{filename}'."

    remaining = len(content) - end
    if remaining > 0:
        excerpt += f"\n\n---\n[{remaining} characters remaining. Ask to 'continue reading' for the next section.]"

    return excerpt

# ─── Summary retrieval ────────────────────────────────────────────────────────

def get_file_summary(relative_path: str) -> str | None:
    """Retrieve a stored file summary from Chroma."""
    try:
        results = _collection.get(
            where={"$and": [{"type": {"$eq": "file_summary"}}, {"source": {"$eq": relative_path}}]},
            include=["documents"]
        )
        if results["documents"]:
            return results["documents"][0]
    except Exception:
        pass
    return None

def search_file_summaries(query: str, n_results: int = 3) -> list[dict]:
    """
    Semantic search across file summaries.
    Returns list of {path, summary} dicts.
    """
    try:
        count = _collection.count()
        if count == 0:
            return []

        results = _collection.query(
            query_texts=[query],
            where={"type": {"$eq": "file_summary"}},
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if dist < 0.7:  # only relevant results
                output.append({
                    "path": meta.get("source", "unknown"),
                    "summary": doc,
                    "distance": dist
                })
        return output
    except Exception as e:
        print(f"[Vault sync: search error: {e}]")
        return []
