"""Backend-only source-link resolver (Day 24 clickable sources).

     LLM -> chunk_id + quote -> Citation Validator -> validated chunk
              -> SourceLinkResolver -> url -> Web UI

The resolver is the ONLY place a source URL is built. It never trusts a URL
from the generation output and returns ``None`` whenever a link cannot be
built reliably.
"""
from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.rag.sources.manual import (
    MANUAL_SOURCE_ROUTE,
    manual_source_url,
    resolve_manual_path,
)
from app.services.rag.sources.models import ResolvedMessageLink, ResolvedSourceLink
from app.services.rag.sources.telegram import (
    TelegramSourceIndex,
    TelegramSourceResolver,
)


class SourceLinkResolver:
    """Resolve a validated citation into a clickable source link."""

    def __init__(
        self,
        settings: Settings,
        *,
        manual_root: str | Path | None = None,
        telegram_path: str | Path | None = None,
        telegram_link_base: str | None = None,
        telegram_index: TelegramSourceIndex | None = None,
        manual_route: str = MANUAL_SOURCE_ROUTE,
    ) -> None:
        self.settings = settings
        self.manual_root = Path(manual_root or settings.rag_manual_path)
        self.manual_route = manual_route
        base = (
            settings.rag_telegram_chat_link_base
            if telegram_link_base is None
            else telegram_link_base
        )
        self.telegram = TelegramSourceResolver(
            telegram_index
            or TelegramSourceIndex(telegram_path or settings.rag_telegram_export_path),
            link_base=base or "",
        )

    # ------------------------------------------------------------------
    def resolve(self, citation) -> ResolvedSourceLink:
        """Return a link for a *validated* citation (invalid -> no link)."""
        if citation is None or not getattr(citation, "quote_valid", False):
            return ResolvedSourceLink()
        source_type = getattr(citation, "source_type", "")
        if source_type == "manual":
            return self._resolve_manual(citation)
        if source_type == "telegram":
            return self._resolve_telegram(citation)
        return ResolvedSourceLink()

    # ------------------------------------------------------------------
    def _resolve_manual(self, citation) -> ResolvedSourceLink:
        filename = str(getattr(citation, "source", "") or "")
        # Build a link only for a real file inside the manual root.
        if resolve_manual_path(self.manual_root, filename) is None:
            return ResolvedSourceLink()
        page = getattr(citation, "page", None)
        page_from = getattr(citation, "page_from", None)
        url = manual_source_url(
            filename, page=page, page_from=page_from, route=self.manual_route
        )
        return ResolvedSourceLink(url=url, link_kind="document")

    def _resolve_telegram(self, citation) -> ResolvedSourceLink:
        message_ids = [
            int(value)
            for value in (getattr(citation, "message_ids", None) or [])
            if str(value).strip().lstrip("-").isdigit()
        ]
        quote = str(getattr(citation, "quote", "") or "")
        topic_id = getattr(citation, "topic_id", None)
        cited = self.telegram.index.find_message_for_quote(message_ids, quote)
        anchor = message_ids[0] if message_ids else None
        target = cited if cited is not None else anchor
        if target is None:
            return ResolvedSourceLink()

        # Build a permalink for every message id of the chunk so the UI can
        # show the main cited message plus context links. The URL is built
        # ONLY from validated message ids + configured base.
        context_links = []
        main_url = None
        for message_id in message_ids:
            link_url, _ = self.telegram.build_url(message_id, topic_id=topic_id)
            is_cited = cited is not None and message_id == cited
            context_links.append(
                ResolvedMessageLink(
                    message_id=message_id, url=link_url, is_cited=is_cited
                )
            )
            if message_id == target:
                main_url = link_url

        url, note = self.telegram.build_url(target, topic_id=topic_id)
        if url is None:
            # No reliable permalink can be built: report the cited message but
            # never claim a link.
            return ResolvedSourceLink(
                cited_message_id=cited,
                topic_id=topic_id,
                context_links=context_links,
            )
        if cited is not None:
            return ResolvedSourceLink(
                url=main_url or url,
                cited_message_id=cited,
                topic_id=topic_id,
                link_kind="message",
                note=note,
                context_links=context_links,
            )
        # Quote could not be mapped to one message: link the conversation
        # anchor and label it honestly (never "Source quote: message X").
        return ResolvedSourceLink(
            url=main_url or url,
            cited_message_id=None,
            topic_id=topic_id,
            link_kind="conversation",
            note=note or (f"Open discussion from message {anchor}" if anchor else ""),
            context_links=context_links,
        )
