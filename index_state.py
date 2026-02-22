from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

INDEX_METADATA_FILE = "index_metadata.json"


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index_fingerprint(
    knowledge_path: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    return {
        "knowledge_path": str(knowledge_path.resolve()),
        "knowledge_sha256": compute_file_sha256(knowledge_path),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def get_index_metadata_path(index_dir: Path) -> Path:
    return index_dir / INDEX_METADATA_FILE


def load_index_metadata(index_dir: Path) -> dict[str, Any] | None:
    metadata_path = get_index_metadata_path(index_dir)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_index_metadata(index_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = get_index_metadata_path(index_dir)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def has_saved_index(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()
