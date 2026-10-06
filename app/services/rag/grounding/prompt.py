"""Grounded prompt + context formatting for Day 24.

The prompt is the *soft* anti-hallucination layer. It is intentionally not
trusted on its own: whatever the model returns is validated by
:mod:`app.services.rag.grounding.validator` against the real retrieved
chunks. The prompt's job is only to make the model express its answer as
``chunk_id`` + exact quote instead of free-floating metadata.
"""
from __future__ import annotations

# DeepSeek JSON mode requires the word "json" in the messages; Ollama only
# needs ``format="json"``. This schema is the shared contract.
RESPONSE_FORMAT = {"type": "json_object"}

GROUNDED_SYSTEM_PROMPT = (
    "Ты — технический ассистент по автомобилю Mitsubishi Xpander. "
    "Отвечай на русском языке, понятно и по существу.\n\n"
    "Ты отвечаешь на вопрос пользователя ТОЛЬКО на основании "
    "предоставленного CONTEXT.\n"
    "Не используй внешние знания, чтобы заполнить пробелы в CONTEXT.\n\n"
    "Для каждого существенного утверждения выбери один или несколько "
    "chunk_id из CONTEXT.\n"
    "Для каждого выбранного chunk укажи КОРОТКУЮ точную цитату (обычно "
    "3–12 слов), которая подтверждает утверждение.\n"
    "Цитата — это ТОЧНАЯ подстрока из TEXT, скопированная посимвольно, "
    "вместе с опечатками и пунктуацией источника. НЕ перефразируй, НЕ "
    "переводи и НЕ исправляй текст. Если не можешь скопировать цитату "
    "точно — выбери другую, более короткую, подстроку.\n"
    "Пример правильной цитаты: \"Защита крепится к штатным отверстиям "
    "силовых элементов кузова.\"\n"
    "Используй только chunk_id, которые присутствуют в CONTEXT.\n\n"
    "Никогда не придумывай:\n"
    "- chunk_id;\n"
    "- source / файл;\n"
    "- page / section;\n"
    "- message_ids;\n"
    "- цитаты.\n\n"
    "MANUAL — официальная техническая документация производителя.\n"
    "TELEGRAM — опыт и обсуждение владельцев. Не представляй Telegram "
    "как официальную рекомендацию Mitsubishi: формулируй это как опыт или "
    "обсуждение владельцев.\n\n"
    "Если CONTEXT содержит достаточно информации хотя бы для полезной "
    "ЧАСТИ ответа — дай эту подтверждённую часть ответа. Явно укажи, какая "
    "часть отсутствует в источниках. Не придумывай недостающие факты."
    "\n\n"
    "Верни status=\"insufficient_context\" ТОЛЬКО если CONTEXT вообще не "
    "позволяет дать полезный ответ на вопрос (а не если в базе нет "
    "идеальной исчерпывающей инструкции). Не отказывайся только потому, "
    "что часть деталей (например, точная последовательность затяжки на "
    "схеме) не попала в текст.\n\n"
    "Для каждого подтверждённого утверждения всё равно указывай evidence с "
    "точными chunk_id и цитатами.\n\n"
    "Ответ должен быть ТОЛЬКО одним JSON-объектом такой формы:\n"
    "{\n"
    '  "status": "answered" | "insufficient_context",\n'
    '  "answer": "текст ответа или null",\n'
    '  "evidence": [\n'
    '    {"chunk_id": "точный chunk_id из CONTEXT", "quote": "точная цитата"}\n'
    "  ]\n"
    "}\n"
    "Используй только status, answer и evidence. Никаких дополнительных "
    "полей."
)

GROUNDED_JSON_NOTE = (
    "Верни ответ строго как JSON-объект с полями status, answer, evidence. "
    "Если данных недостаточно — status = \"insufficient_context\"."
)


def _page_value(meta: dict):
    page = meta.get("page")
    if page is not None:
        return page
    page_from, page_to = meta.get("page_from"), meta.get("page_to")
    if page_from is not None and page_to is not None and page_from != page_to:
        return f"{page_from}-{page_to}"
    return page_from


def _as_text(value) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def format_chunk_block(chunk) -> str:
    """Render ONE retrieved chunk with an explicit ``[CHUNK_ID: ...]`` header."""
    meta = dict(getattr(chunk, "metadata", None) or {})
    source_type = str(getattr(chunk, "source_type", "") or meta.get("source_type", ""))
    source = str(getattr(chunk, "source", "") or meta.get("source", ""))

    lines = [f"[CHUNK_ID: {chunk.chunk_id}]"]
    if source_type:
        lines.append(f"SOURCE_TYPE: {source_type}")
    if source:
        lines.append(f"SOURCE: {source}")
    if meta.get("title"):
        lines.append(f"TITLE: {meta['title']}")
    if source_type == "manual":
        page = _page_value(meta)
        if page is not None:
            lines.append(f"PAGE: {page}")
        if meta.get("section"):
            lines.append(f"SECTION: {meta['section']}")
    if source_type == "telegram":
        if meta.get("chat_name"):
            lines.append(f"CHAT: {meta['chat_name']}")
        if meta.get("message_ids"):
            lines.append(f"MESSAGE_IDS: {_as_text(meta['message_ids'])}")
        if meta.get("date_from"):
            lines.append(f"DATE_FROM: {meta['date_from']}")
        if meta.get("date_to"):
            lines.append(f"DATE_TO: {meta['date_to']}")
    lines.append("")
    lines.append("TEXT:")
    lines.append((getattr(chunk, "text", "") or "").strip())
    return "\n".join(lines)


def build_grounded_context(chunks: list) -> str:
    """Format retrieved chunks so each one is identifiable by its chunk_id."""
    if not chunks:
        return ""
    return "\n\n".join(format_chunk_block(chunk) for chunk in chunks)


def _format_recent_messages(messages: list) -> str:
    """Render a short recent-conversation window (Day 25).

    Only the last N messages are ever sent; long-lived facts belong to the
    TASK STATE block, not to this raw transcript.
    """
    lines: list[str] = []
    for message in messages or []:
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", "")
        )
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        content = (content or "").strip().replace("\n", " ")
        if not content:
            continue
        lines.append(f"{str(role).upper()}: {content}")
    return "\n".join(lines)


def build_grounded_messages(
    question: str,
    chunks: list,
    *,
    task_state_text: str = "",
    recent_messages: list | None = None,
) -> list[dict[str, str]]:
    """Build the grounded chat messages (system + context + JSON question).

    The default (no task state, no recent history) keeps the exact Day 24
    format. Day 25 extends the user turn with TASK STATE and RECENT
    CONVERSATION blocks while keeping the SAME grounded contract, prompt and
    citation validation.
    """
    context = build_grounded_context(chunks)
    if task_state_text or recent_messages:
        parts: list[str] = []
        if task_state_text:
            parts.append("TASK STATE:\n" + task_state_text.strip())
        if recent_messages:
            rendered = _format_recent_messages(recent_messages)
            if rendered:
                parts.append("RECENT CONVERSATION:\n" + rendered)
        parts.append("RAG EVIDENCE:\n" + (context or "(нет релевантных chunks)"))
        parts.append("CURRENT USER MESSAGE:\n" + question)
        parts.append(GROUNDED_JSON_NOTE)
        user = "\n\n".join(parts)
    else:
        user = (
            "CONTEXT:\n"
            f"{context}\n\n"
            "ВОПРОС:\n"
            f"{question}\n\n"
            f"{GROUNDED_JSON_NOTE}"
        )
    return [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
