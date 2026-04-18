import chromadb
import json
import anthropic
import os
from datetime import datetime

_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_profile_collection = _chroma_client.get_or_create_collection(
    name="jarvis_profile_chunks",
    metadata={"hnsw:space": "cosine"}
)
_api_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Identity chunk is always injected — name, address preference, core facts
ALWAYS_INJECT = ["identity", "communication"]

def load_profile_chunks(path: str = "profile_chunks.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

def register_profile_chunks(path: str = "profile_chunks.json"):
    """
    Embed each profile chunk into Chroma.
    Called once on startup. Re-registers if chunks have changed.
    """
    chunks = load_profile_chunks(path)

    existing = _profile_collection.get()
    if existing["ids"]:
        _profile_collection.delete(ids=existing["ids"])

    documents = []
    metadatas = []
    ids = []

    for key, value in chunks.items():
        documents.append(value)
        metadatas.append({"key": key, "content": value})
        ids.append(key)

    _profile_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"[Profile chunks registered: {len(chunks)} sections]")

def get_relevant_chunks(query: str, n_results: int = 3) -> dict:
    """
    Retrieve the most relevant profile chunks for a given message.
    Always includes identity and communication chunks.
    Returns dict of {chunk_key: chunk_content}.
    """
    chunks = load_profile_chunks()
    result = {}

    # Always inject these
    for key in ALWAYS_INJECT:
        if key in chunks:
            result[key] = chunks[key]

    # Semantic retrieval for remaining chunks
    count = _profile_collection.count()
    if count == 0:
        return result

    n = min(n_results, count)
    results = _profile_collection.query(
        query_texts=[query],
        n_results=n,
        include=["metadatas", "distances"]
    )

    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        key = meta["key"]
        # Only add if relevant enough and not already included
        if distance < 0.55 and key not in result:
            result[key] = meta["content"]

    return result

def format_profile_for_prompt(chunks: dict) -> str:
    """Format retrieved profile chunks into a system prompt."""
    if not chunks:
        return ""

    lines = [f"- {key}: {content}" for key, content in chunks.items()]
    return "You are Jarvis, a personal AI assistant. Here is relevant context about the user:\n" + "\n".join(lines)

CURATOR_PROMPT = """You are a memory curator for an AI assistant. Given a user message and available context, 
select only what's genuinely needed to answer well.

User message: {message}

Available context:
{context}

Return ONLY the relevant parts, rewritten concisely. Remove anything not needed for this specific message.
If the context is already minimal and relevant, return it as-is.
Keep your response under 300 words."""

def curate_context(message: str, profile_text: str, session_memories: list, long_term_memories: list) -> str:
    """
    Use Haiku to curate what context actually gets injected.
    Strips irrelevant profile sections and memory chunks before they reach Sonnet.
    """
    # Build the full available context
    context_parts = []

    if profile_text:
        context_parts.append(f"Profile:\n{profile_text}")

    if session_memories:
        session_text = "\n".join(f"- {m}" for m in session_memories)
        context_parts.append(f"This session:\n{session_text}")

    if long_term_memories:
        lt_text = "\n".join(f"- {m}" for m in long_term_memories)
        context_parts.append(f"Past sessions:\n{lt_text}")

    if not context_parts:
        return ""

    full_context = "\n\n".join(context_parts)

    # For very short messages or simple greetings, skip curation
    if len(message.strip()) < 20:
        return full_context

    try:
        response = _api_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": CURATOR_PROMPT.format(
                    message=message,
                    context=full_context
                )
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[Curator error: {e}] — using full context")
        return full_context
