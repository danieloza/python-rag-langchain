# Python RAG LangChain

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)
![Local LLM](https://img.shields.io/badge/LLM-Local%20Inference-2E8B57)

> PL: Repo jest prowadzone po angielsku (dla szerszej widocznosci), ale przyklady i use-case sa osadzone w realnym kontekscie lokalnym.

Production-style showcase project that implements a complete local Retrieval-Augmented Generation (RAG) pipeline with LangChain.

## TL;DR (60s)
Run locally (no paid API keys required):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python rag_demo.py
```

Run custom query:
```bash
python rag_demo.py -q "What are Daniel's main career goals?"
```

Reuse an existing local FAISS index (trusted files only):
```bash
python rag_demo.py --trust-local-index -q "Where does Daniel study?"
```

## What It Demonstrates
- end-to-end local RAG setup (ingestion -> chunking -> embeddings -> retrieval -> answer)
- practical semantic search with FAISS
- grounded QA over a custom local knowledge base (`knowledge.txt`)
- lightweight observability for debugging and evaluation

## Architecture
- embeddings model: `sentence-transformers/all-MiniLM-L6-v2`
- vector store: FAISS (`faiss_index/`)
- generation model: `google/flan-t5-small` via Hugging Face pipeline
- orchestration: LangChain `RetrievalQA`

## Technical Highlights
- `LangChain` for workflow orchestration
- `FAISS` for fast similarity search
- `HuggingFaceEmbeddings` for vector representation
- `RetrievalQA` for retrieval + generation
- local LLM inference (`HuggingFacePipeline`)

## Observability
The demo includes basic QA run observability:
- chain-level verbose execution (`verbose=True`)
- structured JSONL logs in `observability_logs.jsonl`
- captured fields: timestamp, question, answer, source documents

## Index Safety And Freshness
- the index fingerprint is based on `knowledge.txt` hash + chunking config + embedding model
- metadata is stored in `faiss_index/index_metadata.json`
- when knowledge/config changes, the script automatically rebuilds the FAISS index
- default mode is safe: it rebuilds instead of loading pickled index state
- use `--trust-local-index` only for trusted local files to reuse persisted index faster

## AI Data Pipeline (ETL View)
- **Extract**: load unstructured knowledge from `knowledge.txt`
- **Transform**: split text and build vector embeddings
- **Load**: persist vectors into a FAISS index (`faiss_index/`)

Then the online path executes retrieval and generation over the indexed data.

## Project Structure
- `rag_demo.py`: complete RAG implementation
- `knowledge.txt`: local knowledge base
- `requirements.txt`: Python dependencies
- `docs/qa-evaluation.md`: quality and evaluation notes

## Demo Session
Run multiple questions in one command:
```bash
python rag_demo.py -q "Where does Daniel study?" -q "What technologies are used in this RAG pipeline?"
```

Expected answer pattern:
```text
Question: Where does Daniel study?
Answer: Daniel studies Computer Science at the University of Wroclaw (UWr).
Sources: knowledge.txt
```

After execution, inspect:
- `observability_logs.jsonl` for structured run events
- `faiss_index/` for persisted vector artifacts

## Portfolio Positioning
This repository showcases practical skills in:
- GenAI orchestration
- RAG architecture design
- retrieval quality mindset
- AI data pipeline engineering
