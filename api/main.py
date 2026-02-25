from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval.engine import get_retriever
from src.generation.chain import get_qa_chain
from src.config import settings

app = FastAPI(title="Senior RAG API")

# Global instances for efficiency
retriever = get_retriever()
qa_chain = get_qa_chain(retriever)

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(query: Query):
    try:
        # In LCEL, we directly get the string answer
        answer = qa_chain.invoke(query.text)
        return {
            "answer": answer,
            "sources": "retrieved_via_lcel"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.LLM_MODEL}
