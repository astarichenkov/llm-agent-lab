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

    app_name: str = "LLM Agent Lab"
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
    # Default OpenRouter model used when an Agent selects provider="openrouter".
    openrouter_model: str = "openai/gpt-4o-mini"

    # Day 7 — persistent Agent context (SQLite). Relative paths resolve from
    # the process working directory; in Docker ``./data`` is bind-mounted so
    # the database survives restarts and container recreation.
    agent_db_path: str = "data/agents.db"
    agent_default_provider: str = "deepseek"

    system_prompt: str = (
        "You are a helpful educational assistant. "
        "Explain concepts clearly, concisely and accurately."
    )

    # Input validation limit shared with the Pydantic schema
    max_message_length: int = MAX_MESSAGE_LENGTH

    # Persistent application log file (mounted ./logs -> /app/logs in Docker).
    app_log_file: str = "logs/app.log"

    # Day 9 — context compression. Messages older than the recent window are
    # progressively folded into a separate summary. These are Agent-level
    # defaults; the Day 9 API can override them per request.
    day9_recent_messages_limit: int = 6
    day9_compression_batch_size: int = 10

    # Day 10 — context-management strategies. Day 10 never summarises history:
    # each strategy only decides WHICH past messages reach the model.
    # ``day10_window_size`` is the number of user/assistant history items kept
    # by the Sliding Window strategy (not message pairs).
    day10_window_size: int = 6
    day10_recent_messages_limit: int = 6

    # Day 11 — agent memory layers (short-term / working / long-term).
    # ``day11_window_size`` is the short-term window used by the Context
    # Builder via the existing Day 10 SlidingWindowStrategy.
    # ``day11_long_term_path`` is the small JSON file that persists ONLY
    # long-term memory (it must survive a new session and a restart).
    day11_window_size: int = 4
    day11_long_term_path: str = "data/day11_long_term_memory.json"
    day11_classifier_max_tokens: int = 800

    # Day 12 — user profile (personalization). The profile is stored
    # SEPARATELY from the dialog and from memory: a small JSON file.
    day12_profile_path: str = "data/day12_user_profile.json"

    # Day 13 — Task State Machine. The structured task state (stage,
    # current_step, expected_action, plan, pause flag) is persisted to its own
    # small JSON file so it survives separate HTTP requests and restarts.
    day13_task_path: str = "data/day13_task_state.json"

    # Day 14 — mandatory invariants. They are stored in their OWN JSON file,
    # completely separate from the conversation history and from the Day 13
    # task state.
    day14_invariants_path: str = "data/day14_invariants.json"

    # Day 15 — controlled state transitions. The full lifecycle state (state,
    # plan_approved, validation_passed, pause bookkeeping and the transition
    # history including blocked attempts) is persisted to its own JSON file.
    day15_task_path: str = "data/day15_lifecycle_state.json"

    @property
    def has_deepseek_api_key(self) -> bool:
        """True when a real key was provided via the environment."""
        return bool(self.deepseek_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor used by FastAPI dependencies."""
    return Settings()
