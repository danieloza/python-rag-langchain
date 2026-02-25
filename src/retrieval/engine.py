import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_retriever(trust_local=True):
    embeddings = get_embeddings()
    if trust_local and settings.INDEX_DIR.exists():
        vector_store = FAISS.load_local(
            str(settings.INDEX_DIR), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        from src.ingestion.loader import load_and_split
        docs = load_and_split()
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(str(settings.INDEX_DIR))
        
    return vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})
