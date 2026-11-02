"""Central configuration (pydantic-settings). Every threshold and path lives here."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RetrievalMode = Literal["bm25", "vector", "hybrid"]


class Settings(BaseSettings):
    """Application settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        env_ignore_empty=True,  # `FOO=` in .env means "unset", not ""
    )

    # --- Models (Section 2.4) ---
    AGENT_MODEL: str = "claude-haiku-5-5"
    JUDGE_MODEL: str = "claude-sonnet-5-5"
    ANTHROPIC_API_KEY: SecretStr | None = None
    LLM_MAX_TOKENS: int = 8000  # thinking is on by default on current models; leave headroom
    LLM_MAX_RETRIES: int = 4  # SDK-level retries (429 / 5xx / connection errors)
    LLM_TIMEOUT_S: float = 120.0
    LLM_STRUCTURED_METHOD: Literal["json_schema", "function_calling"] = "json_schema"

    # --- Thresholds (Section 4.5) ---
    AUTO_APPROVE_REFUND_LIMIT: float = 100.0
    CONFIDENCE_THRESHOLD: float = Field(default=0.80, ge=0.0, le=1.0)
    MAX_TOOL_STEPS: int = Field(default=4, ge=1)
    RETRIEVAL_TOP_K: int = Field(default=5, ge=1)
    RRF_K: int = Field(default=60, ge=1)
    RETRIEVAL_MODE: RetrievalMode = "hybrid"
    VALIDATION_ENABLED: bool = True
    SENSITIVE_CATEGORIES: str = "other_sensitive"

    # --- Policy windows (Section 4.4; kept here so code never hard-codes them) ---
    DOMESTIC_RETURN_DAYS: int = 30
    INTERNATIONAL_RETURN_DAYS: int = 14
    PEAK_RETURN_DAYS: int = 60
    DAMAGED_REPORT_DAYS: int = 14
    WRONG_ITEM_REPORT_DAYS: int = 14
    HOME_COUNTRY: str = "US"

    # --- Data / storage ---
    BUSINESS_DB_PATH: str = "data/ticketpilot.db"
    CHECKPOINT_DB_PATH: str = "data/checkpoints.db"
    REFERENCE_DATE: date = date(2026, 10, 1)
    KB_DIR: str = "data/kb"

    # --- Vector store / embeddings ---
    QDRANT_URL: str = ""
    QDRANT_COLLECTION: str = "kestrel_kb"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    # Docker bakes the embedding model here so containers start offline
    FASTEMBED_CACHE_DIR: str | None = None

    # --- Observability (optional) ---
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_HOST: str | None = None
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # --- Pricing (optional; USD per million tokens). Unset → tokens only. ---
    AGENT_PRICE_INPUT_PER_MTOK: float | None = None
    AGENT_PRICE_OUTPUT_PER_MTOK: float | None = None
    JUDGE_PRICE_INPUT_PER_MTOK: float | None = None
    JUDGE_PRICE_OUTPUT_PER_MTOK: float | None = None

    # --- Eval harness ---
    EVAL_CONCURRENCY: int = Field(default=4, ge=1)

    # --- Services ---
    API_URL: str = "http://localhost:8000"

    @property
    def sensitive_categories(self) -> set[str]:
        return {c.strip() for c in self.SENSITIVE_CATEGORIES.split(",") if c.strip()}

    def resolve_path(self, value: str) -> Path:
        """Resolve a configured path relative to the project root when not absolute."""
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def business_db_path(self) -> Path:
        return self.resolve_path(self.BUSINESS_DB_PATH)

    @property
    def checkpoint_db_path(self) -> Path:
        return self.resolve_path(self.CHECKPOINT_DB_PATH)

    @property
    def kb_dir(self) -> Path:
        return self.resolve_path(self.KB_DIR)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call `get_settings.cache_clear()` in tests after env changes."""
    return Settings()
