"""
Lightweight RAG pipeline using ChromaDB + SentenceTransformers.
Designed for FYP — no cloud services, no API costs.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "counselling_kb"

_collection = None
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedding model loaded")
    return _embedder


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _chunk_text(text: str, max_words: int = 200, overlap_words: int = 30) -> list[str]:
    """Split document into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    step = max_words - overlap_words
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def build_index(force_rebuild: bool = False):
    """
    Index all markdown files in knowledge_base/.
    Call once at startup; safe to call repeatedly (skips if already indexed).
    Returns number of chunks indexed.
    """
    global _collection
    md_files = sorted(KB_DIR.glob("*.md"))
    collection = _get_collection()

    if collection.count() > 0 and not force_rebuild:
        log.info("RAG index already built (%d chunks)", collection.count())
        return collection.count()

    if force_rebuild and collection.count() > 0:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None
        collection = _get_collection()

    embedder = _get_embedder()
    all_docs, all_ids, all_meta = [], [], []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{md_file.stem}_{i}"
            all_docs.append(chunk)
            all_ids.append(doc_id)
            all_meta.append({"source": md_file.name, "chunk": i})

    if not all_docs:
        log.warning("No documents found in %s", KB_DIR)
        return 0

    embeddings = embedder.encode(all_docs, show_progress_bar=False).tolist()
    collection.add(
        documents=all_docs,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_meta,
    )
    log.info(
        "RAG index built: %d chunks from %d files",
        len(all_docs),
        len(md_files),
    )
    return len(all_docs)


def get_index_stats() -> dict:
    """Return current RAG index chunk and source file counts."""
    md_files = list(KB_DIR.glob("*.md"))
    try:
        count = _get_collection().count()
    except Exception:
        count = 0
    return {"chunks": count, "files": len(md_files)}


def get_rag_status() -> dict:
    """Status payload for /api/rag-status."""
    enabled = os.getenv("RAG_ENABLED", "true").lower() in {"1", "true", "yes"}
    stats = get_index_stats()
    ready = stats["chunks"] > 0
    return {
        "rag_enabled": enabled,
        "indexed": ready,
        "chunk_count": stats["chunks"],
        "file_count": stats["files"],
        "ready": ready and enabled,
        "message": (
            f"RAG ready — {stats['chunks']} chunks from {stats['files']} files."
            if ready
            else "Index empty. Run build_index() or restart app."
        ),
    }


def retrieve(query: str, top_k: int = 3) -> Optional[str]:
    """Return formatted RAG context string, or None if nothing relevant."""
    context, _ = retrieve_with_meta(query, top_k=top_k)
    return context


def retrieve_with_meta(query: str, top_k: int = 3) -> tuple[Optional[str], int]:
    """
    Retrieve relevant knowledge chunks.
    Returns (formatted_context, chunk_count).
    """
    if os.getenv("RAG_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return None, 0

    try:
        collection = _get_collection()
        if collection.count() == 0:
            build_index()

        top_k = int(os.getenv("RAG_TOP_K", top_k))
        embedder = _get_embedder()
        query_embedding = embedder.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        distances = results["distances"][0]
        metas = results["metadatas"][0]

        relevant = [
            (doc, meta["source"], dist)
            for doc, meta, dist in zip(docs, metas, distances)
            if dist < 0.7
        ]

        if not relevant:
            return None, 0

        parts = []
        for doc, source, _ in relevant:
            parts.append(f"[{source}]\n{doc}")

        return "\n\n---\n\n".join(parts), len(relevant)

    except Exception as exc:
        log.warning("RAG retrieval failed: %s", exc)
        return None, 0


__all__ = ["retrieve", "retrieve_with_meta", "build_index", "get_index_stats", "get_rag_status"]
