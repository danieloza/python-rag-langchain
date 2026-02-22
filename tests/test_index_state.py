from __future__ import annotations

from pathlib import Path

from index_state import (
    build_index_fingerprint,
    has_saved_index,
    load_index_metadata,
    save_index_metadata,
)


def test_fingerprint_changes_when_knowledge_changes(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "knowledge.txt"
    knowledge_path.write_text("first version", encoding="utf-8")

    first = build_index_fingerprint(
        knowledge_path=knowledge_path,
        embedding_model="embedding-model-v1",
        chunk_size=450,
        chunk_overlap=80,
    )

    knowledge_path.write_text("second version", encoding="utf-8")
    second = build_index_fingerprint(
        knowledge_path=knowledge_path,
        embedding_model="embedding-model-v1",
        chunk_size=450,
        chunk_overlap=80,
    )

    assert first["knowledge_sha256"] != second["knowledge_sha256"]


def test_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = {
        "knowledge_path": "C:/repo/knowledge.txt",
        "knowledge_sha256": "abc123",
        "embedding_model": "embedding-model-v1",
        "chunk_size": 450,
        "chunk_overlap": 80,
    }

    save_index_metadata(tmp_path, metadata)
    loaded = load_index_metadata(tmp_path)

    assert loaded == metadata


def test_load_metadata_returns_none_when_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "index_metadata.json").write_text("{not-json", encoding="utf-8")
    assert load_index_metadata(tmp_path) is None


def test_has_saved_index_requires_faiss_and_pickle(tmp_path: Path) -> None:
    assert has_saved_index(tmp_path) is False

    (tmp_path / "index.faiss").write_bytes(b"faiss")
    assert has_saved_index(tmp_path) is False

    (tmp_path / "index.pkl").write_bytes(b"pickle")
    assert has_saved_index(tmp_path) is True
