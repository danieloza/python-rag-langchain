from __future__ import annotations

import argparse
import json
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

KNOWLEDGE_FILE = Path("knowledge.txt")
INDEX_DIR = Path("faiss_index")
OBSERVABILITY_LOG = Path("observability_logs.jsonl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"


def load_and_split_documents(knowledge_path: Path):
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Missing knowledge base file: {knowledge_path}")

    loader = TextLoader(str(knowledge_path), encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=80)
    return splitter.split_documents(documents)


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_vector_store():
    embeddings = create_embeddings()

    if INDEX_DIR.exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    chunks = load_and_split_documents(KNOWLEDGE_FILE)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


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
    vector_store = build_or_load_vector_store()
    print("[2/3] Initializing local LLM pipeline")
    qa_chain = build_qa_chain(vector_store)
    print("[3/3] Running RetrievalQA queries")
    run_queries(qa_chain, questions)
    print(f"Observability log written to: {OBSERVABILITY_LOG}")


if __name__ == "__main__":
    main()
