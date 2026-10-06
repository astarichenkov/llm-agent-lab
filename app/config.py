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

    # Day 17 — VictoriaLogs MCP tool. The base URL is REQUIRED only when the
    # Day 17 feature is used; its absence must never break application startup
    # or Day 16 (see ``VictoriaLogsClient`` which fails lazily).
    #
    # ``victoria_logs_service_field`` is CONFIGURABLE on purpose: the exact
    # field that stores the service name is deployment-specific. The default
    # ``service`` is an assumption that MUST be verified against real data
    # (the provided value can be overridden with VICTORIA_LOGS_SERVICE_FIELD).
    victoria_logs_base_url: str = ""
    victoria_logs_service_field: str = "service"
    victoria_logs_timeout_seconds: float = 5.0
    # Internal corporate CAs are often absent from the public trust store; the
    # flag stays ``True`` by default (never disable TLS verification silently).
    victoria_logs_verify_ssl: bool = True
    # Preferred secure alternative to disabling verification: path to a CA
    # bundle (PEM) that validates the VictoriaLogs certificate.
    victoria_logs_ca_bundle: str = ""

    # Day 18 — scheduled monitoring. Jobs and run aggregates are stored in
    # this SQLite database (created lazily on first use). The directory is
    # bind-mounted in Docker (./data -> /app/data), so the DB survives
    # restarts and container recreation.
    day18_monitoring_db_path: str = "data/day18/monitoring.db"

    # Day 19 — MCP tool-composition pipeline. Every generated report lives
    # under this single root (git-ignored ``data/``). The run id is always
    # validated and the resolved directory must stay inside the root, so a
    # caller can never choose an arbitrary filesystem path.
    day19_artifact_root: str = "data/day19/runs"

    # Day 20 — multi-server MCP orchestration (VictoriaLogs + Gitea + Reports).
    #
    # Gitea is READ-ONLY by design: only GET endpoints are ever called and the
    # repository is fixed in configuration (the LLM can never choose an
    # arbitrary owner/repository). The token is optional for local discovery
    # and must NEVER be committed; it is only sent as an Authorization header.
    gitea_base_url: str = ""
    gitea_token: str = ""
    gitea_repository_owner: str = ""
    gitea_repository_name: str = ""
    gitea_default_branch: str = "main"
    gitea_timeout_seconds: float = 10.0
    # Hard cap on the diff text handed back to the LLM (and stored in traces).
    gitea_max_diff_chars: int = 15000
    # Hard cap on commits returned by one ``recent_commits`` call.
    gitea_max_commits: int = 20

    # Day 20 — investigation report storage. Like Day 19 the investigation id
    # is validated server-side and the resolved path must stay inside the
    # root, so no caller can choose an arbitrary filesystem path.
    day20_artifact_root: str = "data/day20/investigations"
    # Orchestration safety limits (see app/services/day20/orchestrator.py).
    day20_max_tool_iterations: int = 16
    day20_max_total_tool_calls: int = 24
    day20_tool_timeout_seconds: float = 30.0

    # Day 21 — local RAG knowledge-base ingestion/indexing. All paths are
    # relative to the process working directory and can be overridden with
    # environment variables (RAG_MANUAL_PATH, RAG_TELEGRAM_EXPORT_PATH, ...).
    # The defaults are dev/demo locations only — the Python code is never
    # bound to an absolute machine path.
    rag_manual_path: str = "docs/xpander/manual"
    # A single result.json OR a directory with many *.json exports.
    rag_telegram_export_path: str = "docs/xpander/ChatExport"
    # Day 24 — optional Telegram permalink base for clickable source links.
    # When set it has highest priority and is combined as
    # ``<base>/<message_id>`` (e.g. ``https://t.me/xpanderclub`` or
    # ``https://t.me/c/1234567890``). Leave empty to fall back to the public
    # username found in the export, then to the ``/c/<id>`` supergroup form.
    rag_telegram_chat_link_base: str = ""
    # Chunking parameters (shared by both strategies; no magic numbers in
    # code — see app/services/rag/chunking).
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    rag_structural_max_chars: int = 1800
    # Minimum size before a structural boundary (new heading/section) may
    # split the current chunk. Keeps (mis)detected short headings from
    # producing tiny chunks.
    rag_structural_min_chars: int = 400
    # Telegram conversation grouping knobs (deterministic strategy).
    rag_telegram_gap_minutes: int = 30
    rag_telegram_max_group_messages: int = 12
    # Embeddings: auto -> Ollama (if reachable) -> sentence-transformers (if
    # installed) -> dependency-free local hashing embedder. Never a paid
    # cloud embedding API by default.
    rag_embedding_provider: str = "auto"
    rag_embedding_model: str = "bge-m3"
    rag_ollama_base_url: str = "http://localhost:11434"
    rag_ollama_timeout_seconds: float = 120.0
    # Dimension used by the built-in hashing fallback embedder.
    rag_embedding_dimension: int = 384
    # Persistent local vector index (SQLite). Git-ignored via data/.
    # ``rag_index_path`` is the primary (structural) index; the fixed-size
    # index used by the ``compare-chunking`` demo lives in a sibling file.
    rag_index_path: str = "data/day21/rag_index.sqlite3"
    rag_fixed_index_path: str = "data/day21/rag_index_fixed.sqlite3"

    # Day 22 — first end-to-end RAG request (retrieval + generation).
    #
    # Generation is a SEPARATE role from embedding: ``bge-m3`` only produces
    # vectors and must never be used to generate answers. The provider is
    # ``deepseek`` (reusing the existing client) or ``ollama`` (local chat
    # model); the model name is resolved in ONE place (the generation
    # factory) so it is never hard-coded across modules.
    rag_generation_provider: str = "deepseek"
    rag_generation_model: str = ""
    # Low temperature keeps the No-RAG vs RAG comparison reproducible.
    rag_generation_temperature: float = 0.0
    rag_generation_max_tokens: int = 1024
    rag_generation_timeout_seconds: float = 120.0
    # Default number of retrieved chunks for the UI (validated 1..20 by the
    # API schema).
    rag_top_k: int = 5
    # Hard cap on the assembled RAG context so a pathological retrieval can
    # never overflow the generation model's window.
    rag_context_max_chars: int = 12000
    # Day 22 evaluation dataset + manually assigned PASS/PARTIAL/FAIL grades.
    # Both live under the git-ignored ``data/`` directory.
    rag_day22_evaluation_path: str = "data/day22/evaluation_questions.json"
    rag_day22_results_path: str = "data/day22/evaluation_results.json"

    # Day 23 — improved retrieval: query rewrite + wider candidate search +
    # similarity-threshold filtering. ``retrieval_top_k`` is the WIDE initial
    # set; ``final_top_k`` is how many chunks actually reach the LLM.
    # The default threshold 0.50 was chosen from the real Day 22 control
    # questions: the lowest best-expected-source score was 0.5102, so 0.50
    # keeps every expected source while dropping the weak tail (see
    # docs/week5/day23.md).
    rag_retrieval_top_k: int = 20
    rag_final_top_k: int = 5
    rag_similarity_threshold: float = 0.50
    # Query rewrite is a second, cheap call to the SAME generation model.
    rag_rewrite_max_tokens: int = 200
    # Day 23 evaluation grades live separately from the Day 22 grades so the
    # two experiments can be graded independently. The dataset is shared.
    rag_day23_results_path: str = "data/day23/evaluation_results.json"
    # Chosen from the real smoke run: this question shows a wide 20-candidate
    # retrieval being reduced to 2 clean official/owner chunks (18 rejected).
    rag_day23_demo_question: str = (
        "Какая жидкость ATF и какой объём предусмотрены для автоматической "
        "коробки передач Xpander?"
    )

    # Day 24 — grounded RAG (citations + anti-hallucination).
    #
    # The grounding gate deliberately reuses ``rag_similarity_threshold`` as
    # the answer threshold (Option A, documented in docs/week5/day24.md): a
    # chunk that is not relevant enough for retrieval must not be relevant
    # enough to justify an answer. No second threshold is introduced.
    #
    # The 10 control questions and 5 negative questions are both stored under
    # git-ignored ``data/day24/``. The negative dataset was produced by
    # running real retrieval (see the note field of every entry).
    rag_day24_results_path: str = "data/day24/evaluation_results.json"
    rag_day24_negative_path: str = "data/day24/negative_questions.json"
    # Demo questions chosen from real retrieval runs (Day 24 README).
    rag_day24_demo_question_manual: str = (
        "Какое давление должно быть в шинах Mitsubishi Xpander размера "
        "205/55R16 при нагрузке 1–5 человек + груз?"
    )
    rag_day24_demo_question_telegram: str = (
        "Что владельцы в Telegram называют возможной причиной того, что "
        "пластик на Xpander «поплыл»?"
    )
    rag_day24_demo_question_insufficient: str = (
        "Как выполняется адаптация вариатора Xpander диагностическим сканером?"
    )

    # Day 25 — stateful mini-chat with RAG + task memory.
    #
    # The chat database lives SEPARATELY from the Day 21 vector index (a chat
    # is private application state; it is never embedded). Task state is
    # stored as a JSON column in the same chat DB.
    day25_chat_db_path: str = "data/day25/chat.sqlite3"
    # Number of recent raw messages sent to the LLM. Long-lived facts live in
    # the structured task state instead of the transcript.
    day25_recent_messages: int = 6
    # "llm" (default) uses the generation model to extract state operations;
    # "rules" forces the deterministic offline extractor.
    day25_state_extractor: str = "llm"
    # Two reproducible multi-turn scenarios (diagnostics + maintenance).
    day25_scenarios_path: str = "data/day25/scenarios"
    day25_eval_results_path: str = "data/day25/evaluation_results.json"
    day25_demo_opening_question: str = (
        "Хочу разобраться, почему Mitsubishi Xpander вибрирует при движении "
        "на D."
    )
    # Day 25 — controlled manual/PDF context expansion: a strong manual hit
    # brings in its same-document neighbours (e.g. title page -> fasteners
    # page). Only real indexed chunks are ever added.
    rag_context_expansion_enabled: bool = True
    rag_context_expansion_radius: int = 1
    rag_context_expansion_max: int = 4
    rag_context_expansion_min_score: float = 0.55
    # Optional separate threshold for the chat pipeline. Defaults to the Day
    # 24 threshold; only change it AFTER the contextual query fix if the
    # regression scenarios still show a systematic gap.
    rag_chat_similarity_threshold: float = 0.50

    @property
    def has_deepseek_api_key(self) -> bool:
        """True when a real key was provided via the environment."""
        return bool(self.deepseek_api_key)

    @property
    def gitea_repository(self) -> str:
        """Return the configured ``owner/name`` repository (may be empty)."""
        owner = (self.gitea_repository_owner or "").strip()
        name = (self.gitea_repository_name or "").strip()
        if owner and name:
            return f"{owner}/{name}"
        return ""


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor used by FastAPI dependencies."""
    return Settings()
