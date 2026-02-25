from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    KNOWLEDGE_FILE: Path = BASE_DIR / "knowledge.txt"
    INDEX_DIR: Path = BASE_DIR / "faiss_index"
    
    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-small"
    
    # RAG Config
    CHUNK_SIZE: int = 450
    CHUNK_OVERLAP: int = 80
    RETRIEVAL_K: int = 3
    
    # Hardware
    DEVICE: str = "cpu" # Will be auto-detected in code
    
    # Observability
    ENABLE_TRACING: bool = False
    LANGCHAIN_API_KEY: str | None = None

settings = Settings()
