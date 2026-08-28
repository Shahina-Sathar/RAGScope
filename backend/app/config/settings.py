from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    pinecone_api_key: str
    pinecone_index_name: str = "ragscope"
    pinecone_host: str

    database_url: str = "postgresql+asyncpg://ragscope:ragscope@localhost:5432/ragscope"

    upload_dir: Path = Path("./data/uploads")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5


settings = Settings()
