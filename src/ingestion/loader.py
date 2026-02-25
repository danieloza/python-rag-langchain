from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

def load_and_split():
    if not settings.KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {settings.KNOWLEDGE_FILE}")
        
    loader = TextLoader(str(settings.KNOWLEDGE_FILE), encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    return loader.load_and_split(splitter)
