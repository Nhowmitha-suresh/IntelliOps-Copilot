import os
from dotenv import load_dotenv

load_dotenv()

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        openai_api_key: str = ""
        gemini_api_key: str = ""
        mongo_uri: str = "mongodb://root:mongopassword@localhost:27017/intelliops?authSource=admin"
        postgres_dsn: str = "postgresql://postgres:postgrespassword@localhost:5432/intelliops"
        llm_primary_provider: str = "openai"
        llm_fallback_provider: str = "google"
        max_retries: int = 3
        request_timeout_seconds: int = 30
        similarity_threshold: float = 0.7

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
except ImportError:
    from pydantic import BaseSettings

    class Settings(BaseSettings):
        openai_api_key: str = ""
        gemini_api_key: str = ""
        mongo_uri: str = "mongodb://root:mongopassword@localhost:27017/intelliops?authSource=admin"
        postgres_dsn: str = "postgresql://postgres:postgrespassword@localhost:5432/intelliops"
        llm_primary_provider: str = "openai"
        llm_fallback_provider: str = "google"
        max_retries: int = 3
        request_timeout_seconds: int = 30
        similarity_threshold: float = 0.7

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"


settings = Settings()
