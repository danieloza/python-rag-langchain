# QA Notes for `python-rag-langchain`

## Quality Goals
- Grounded answers should be based on retrieved knowledge chunks.
- Responses should avoid unsupported claims outside `knowledge.txt`.
- The same question should produce stable answers (`do_sample=False`).

## Suggested Test Prompts
1. What programming languages does Daniel use?
2. Where does Daniel study?
3. What are Daniel's career goals?
4. Which technologies are used in this RAG pipeline?

## Observability Review
- Inspect `observability_logs.jsonl` after each run.
- Validate that `source_documents` are present.
- Check whether answer content aligns with the retrieved context.

## Improvement Ideas
- Add retrieval metrics (`precision@k`, hit-rate over labeled questions).
- Expand knowledge base coverage and compare answer quality before/after.
- Introduce prompt templates with stricter citation formatting.
