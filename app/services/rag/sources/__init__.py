"""Day 24 clickable source links (manual PDFs + Telegram permalinks)."""
from app.services.rag.sources.links import SourceLinkResolver
from app.services.rag.sources.manual import (
    MANUAL_SOURCE_ROUTE,
    is_safe_manual_filename,
    manual_source_url,
    resolve_manual_path,
)
from app.services.rag.sources.models import ResolvedSourceLink
from app.services.rag.sources.telegram import (
    TelegramSourceIndex,
    TelegramSourceResolver,
    internal_chat_id,
    is_valid_username,
)

__all__ = [
    "MANUAL_SOURCE_ROUTE",
    "ResolvedSourceLink",
    "SourceLinkResolver",
    "TelegramSourceIndex",
    "TelegramSourceResolver",
    "internal_chat_id",
    "is_safe_manual_filename",
    "is_valid_username",
    "manual_source_url",
    "resolve_manual_path",
]
