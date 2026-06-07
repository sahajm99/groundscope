"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (agent brain) — OpenAI-compatible
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embed_provider: str = "local"  # "local" (fastembed) | "openai" (compatible API)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_api_key: str = ""
    embed_base_url: str = ""
    vector_dim: int = 384

    # DB
    database_url: str = ""

    # Web search
    tavily_api_key: str = ""

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Chunking
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # Hardening
    session_ttl_seconds: int = 3600
    max_upload_mb: int = 10
    max_pages: int = 120
    rate_limit_per_min: int = 12
    global_daily_cap: int = 500

    # Agent
    max_tool_rounds: int = 2
    relevance_distance_threshold: float = 0.5  # calibrated for bge-small: relevant ~0.44, irrelevant ~0.61
    agent_engine: str = "loop"  # "loop" (explicit, proven) | "langgraph" (StateGraph)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def db_configured(self) -> bool:
        return bool(self.database_url)

    @property
    def web_search_configured(self) -> bool:
        return bool(self.tavily_api_key)


settings = Settings()
