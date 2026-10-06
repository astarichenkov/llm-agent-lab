"""Prompts for the Day 22 No-RAG and RAG modes.

Both modes share the SAME base system prompt. The RAG prompt only appends the
source-handling rules required to use retrieved context honestly. This keeps
the comparison fair: the ONLY difference between the two pipelines is the
presence (or absence) of retrieved context.
"""
from __future__ import annotations

# Shared by BOTH modes -- do not fork this text per mode.
BASE_SYSTEM_PROMPT = (
    "Ты — технический ассистент по автомобилю Mitsubishi Xpander. "
    "Отвечай на русском языке, понятно и по существу. "
    "Не выдумывай факты: если данных не хватает, прямо скажи об этом."
)

# Appended ONLY in RAG mode.
RAG_CONTEXT_INSTRUCTIONS = (
    "\n\n"
    "В сообщении пользователя ниже будет приведён КОНТЕКСТ — фрагменты, "
    "найденные в локальной базе знаний по Mitsubishi Xpander. "
    "Используй этот контекст при формировании ответа.\n"
    "Типы источников:\n"
    "- MANUAL / manual — официальная техническая документация производителя. "
    "Это наиболее авторитетный источник.\n"
    "- TELEGRAM / telegram — обсуждения и пользовательский опыт владельцев. "
    "Если информация пришла из Telegram, формулируй её как опыт/обсуждение "
    "владельцев, а НЕ как официальную рекомендацию Mitsubishi.\n"
    "Не выдавай обсуждение в Telegram за официальную рекомендацию производителя. "
    "Если контекст не содержит ответа, скажи об этом и не додумывай."
)

CONTEXT_EMPTY_NOTE = "[Контекст: релевантных фрагментов в базе знаний не найдено.]"


# ----------------------------------------------------------------------
# Day 23 — query rewrite (retrieval-only)
# ----------------------------------------------------------------------
# The rewritten query is NEVER shown to the generation model as the question;
# it is only used to produce the query embedding. The prompt forbids inventing
# a diagnosis and forbids dropping important conditions so a rewrite can only
# reformulate, not narrow down to a guessed answer.
REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's question into a concise search query for a "
    "Mitsubishi Xpander knowledge base.\n\n"
    "The knowledge base contains:\n"
    "- technical manuals;\n"
    "- Telegram discussions between owners.\n\n"
    "The knowledge base and its questions are mostly in Russian. Preserve the "
    "language of the user's question and keep the original technical terms.\n\n"
    "Preserve:\n"
    "- symptoms;\n"
    "- conditions;\n"
    "- component names;\n"
    "- technical terms;\n"
    "- important user wording.\n\n"
    "Add useful technical synonyms only when reasonable.\n\n"
    "Do not answer the question.\n"
    "Do not explain your reasoning.\n"
    "Do not invent a specific fault or state a diagnosis as a fact.\n"
    "Do not remove important conditions of the original question.\n\n"
    "Return only the search query."
)


def build_rewrite_messages(question: str) -> list[dict[str, str]]:
    """Build the messages that turn *question* into a retrieval query."""
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_messages(
    question: str, context: str | None = None
) -> list[dict[str, str]]:
    """Build the chat messages for one question.

    * ``context is None`` -> No-RAG: base system prompt + question.
    * ``context`` provided -> RAG: base system prompt + context rules, then the
      context and the question in the user message.
    """
    if context is None:
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    system = BASE_SYSTEM_PROMPT + RAG_CONTEXT_INSTRUCTIONS
    context_text = context.strip() if context else CONTEXT_EMPTY_NOTE
    user = (
        "КОНТЕКСТ:\n"
        f"{context_text}\n\n"
        "ВОПРОС:\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
