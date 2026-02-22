from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

from index_state import build_index_fingerprint, has_saved_index, load_index_metadata, save_index_metadata

KNOWLEDGE_FILE = Path("knowledge.txt")
INDEX_DIR = Path("faiss_index")
OBSERVABILITY_LOG = Path("observability_logs.jsonl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80


def load_and_split_documents(knowledge_path: Path):
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Missing knowledge base file: {knowledge_path}")

    loader = TextLoader(str(knowledge_path), encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def rebuild_vector_store(
    embeddings: HuggingFaceEmbeddings,
    fingerprint: dict[str, object],
) -> FAISS:
    chunks = load_and_split_documents(KNOWLEDGE_FILE)
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    save_index_metadata(INDEX_DIR, fingerprint)
    return vector_store


def build_or_load_vector_store(trust_local_index: bool):
    embeddings = create_embeddings()
    fingerprint = build_index_fingerprint(
        knowledge_path=KNOWLEDGE_FILE,
        embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    metadata = load_index_metadata(INDEX_DIR)
    metadata_matches = metadata == fingerprint

    if not has_saved_index(INDEX_DIR):
        print("[index] No local index found. Building a fresh FAISS index.")
        return rebuild_vector_store(embeddings, fingerprint)

    if not metadata_matches:
        print("[index] Knowledge/config changed. Rebuilding FAISS index.")
        return rebuild_vector_store(embeddings, fingerprint)

    if trust_local_index:
        print("[index] Loading trusted local index (pickle deserialization enabled).")
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print(
        "[index] Safe mode active. Rebuilding index instead of loading pickled state. "
        "Use --trust-local-index to reuse persisted index."
    )
    return rebuild_vector_store(embeddings, fingerprint)


def build_llm() -> HuggingFacePipeline:
    generation_pipeline = pipeline(
        task="text2text-generation",
        model=GENERATION_MODEL_NAME,
        tokenizer=GENERATION_MODEL_NAME,
        max_new_tokens=128,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=generation_pipeline)


def build_qa_chain(vector_store: FAISS) -> RetrievalQA:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return RetrievalQA.from_chain_type(
        llm=build_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )


def log_observability(question: str, answer: str, sources: list[str]) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "source_documents": sources,
    }
    with OBSERVABILITY_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_queries(qa_chain: RetrievalQA, questions: Iterable[str]) -> None:
    for question in questions:
        result = qa_chain.invoke({"query": question})
        answer = result["result"].strip()
        source_documents = result.get("source_documents", [])
        sources = [doc.metadata.get("source", "unknown") for doc in source_documents]
        log_observability(question, answer, sources)

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Sources: {', '.join(sources) if sources else 'none'}")
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Retrieval-Augmented Generation demo with LangChain and FAISS."
    )
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="Ask one or more custom questions. Repeat -q to add multiple questions.",
    )
    parser.add_argument(
        "--trust-local-index",
        action="store_true",
        help=(
            "Reuse local FAISS index by enabling pickle deserialization. "
            "Use only for trusted local files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = args.question or [
        "What programming languages does Daniel use?",
        "Where does Daniel study?",
        "What are Daniel's career goals?",
        "How does this project demonstrate RAG skills?",
    ]

    print("[1/3] Preparing vector store (FAISS + local embeddings)")
    vector_store = build_or_load_vector_store(
        trust_local_index=args.trust_local_index,
    )
    print("[2/3] Initializing local LLM pipeline")
    qa_chain = build_qa_chain(vector_store)
    print("[3/3] Running RetrievalQA queries")
    run_queries(qa_chain, questions)
    print(f"Observability log written to: {OBSERVABILITY_LOG}")


if __name__ == "__main__":
    main()
