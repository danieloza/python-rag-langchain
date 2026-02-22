# python-rag-langchain

A production-style showcase project that implements a complete local Retrieval-Augmented Generation (RAG) pipeline using LangChain.

## Overview
This repository demonstrates how to orchestrate a practical GenAI system that:
- ingests local data from `knowledge.txt`
- transforms raw text into chunks and embeddings
- loads vectors into FAISS for semantic retrieval
- generates grounded answers with a local Hugging Face model (`google/flan-t5-small`)

The project is fully local and does not require paid API keys.

## Technical Highlights
- `LangChain` for end-to-end AI workflow orchestration
- `FAISS` as the vector database for fast similarity search
- `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`
- `RetrievalQA` chain for retrieval + generation
- local LLM inference through `HuggingFacePipeline`

## LLM Observability
The demo includes lightweight observability for QA runs:
- chain-level verbose execution (`verbose=True`)
- structured JSONL run logs in `observability_logs.jsonl`
- captured fields: timestamp, question, answer, source documents

## AI Data Pipeline (ETL) Perspective
This repository explicitly reflects an ETL-style AI workflow:
- **Extract**: load unstructured knowledge from `knowledge.txt`
- **Transform**: split text and create vector embeddings
- **Load**: persist vectors in a FAISS index (`faiss_index/`)

Then the online inference path performs retrieval and generation over the indexed data.

## Project Structure
- `rag_demo.py`: complete RAG implementation
- `knowledge.txt`: local knowledge base
- `requirements.txt`: Python dependencies
- `docs/qa-evaluation.md`: quality and evaluation notes

## Quick Start
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python rag_demo.py
```

## Example Custom Query
```bash
python rag_demo.py -q "What are Daniel's main career goals?"
```

This project showcases practical skills in GenAI orchestration, RAG architecture design, and AI data pipeline engineering.
