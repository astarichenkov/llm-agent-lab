"""Application configuration, sourced from environment variables / .env file.

Every value can be overridden with an environment variable (case-insensitive):

* ``DEEPSEEK_API_KEY``            -> ``deepseek_api_key``
* ``DEEPSEEK_BASE_URL``           -> ``deepseek_base_url``
* ``DEEPSEEK_MODEL``              -> ``deepseek_model``
* ``DEEPSEEK_TIMEOUT_SECONDS``    -> ``deepseek_timeout_seconds``
* ``SYSTEM_PROMPT``               -> ``system_prompt``
* ``MAX_MESSAGE_LENGTH``          -> ``max_message_length``
* ``APP_NAME``                    -> ``app_name``
* ``ENVIRONMENT``                 -> ``environment``
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.chat import MAX_MESSAGE_LENGTH


class Settings(BaseSettings):
    """Runtime configuration. Never commit secrets; the API key must come
    from the environment or the local (git-ignored) ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DeepSeek Study Assistant"
    environment: str = "development"

    # DeepSeek / OpenAI-compatible client settings
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: float = 30.0

    # OpenRouter (Day 5 model comparison) — used ONLY for Day 5.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: float = 90.0

    system_prompt: str = (
        "You are a helpful educational assistant. "
        "Explain concepts clearly, concisely and accurately."
    )

    # Input validation limit shared with the Pydantic schema
    max_message_length: int = MAX_MESSAGE_LENGTH

    # Persistent application log file (mounted ./logs -> /app/logs in Docker).
    app_log_file: str = "logs/app.log"

    @property
    def has_deepseek_api_key(self) -> bool:
        """True when a real key was provided via the environment."""
        return bool(self.deepseek_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor used by FastAPI dependencies."""
    return Settings()
