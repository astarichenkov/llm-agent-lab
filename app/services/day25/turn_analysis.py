"""Pure heuristics for Day 25 conversational retrieval.

The single most important question for multi-turn RAG is:

    Is this message a self-contained question, or a short follow-up whose
    meaning only exists in the current conversation?

A short follow-up ("что делать?", "как установить?", "а дальше?") must NOT be
sent to the vector index as-is — retrieval would lose the topic. It also must
NOT silently replace the user's goal.

Everything here is deterministic and side-effect free so it can be reused by
the contextual query builder, the task-state extractor and the tests without
depending on an LLM.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# A follow-up is usually one of the function words below, or contains an
# anaphoric reference to the previous turn.
_FOLLOWUP_QUESTION_WORDS = (
    "как",
    "что",
    "почему",
    "зачем",
    "куда",
    "где",
    "когда",
    "какой",
    "какая",
    "какие",
    "какую",
    "каком",
    "каким",
    "чем",
    "кто",
    "сколько",
    "можно",
    "нужно",
    "стоит",
)

_FOLLOWUP_PHRASES = (
    "что делать",
    "что дальше",
    "что теперь",
    "что еще",
    "что ещё",
    "как установить",
    "как это",
    "как его",
    "как её",
    "как ее",
    "а дальше",
    "а как",
    "а что",
    "а где",
    "а куда",
    "а почему",
    "а зачем",
    "а когда",
    "а какой",
    "а какая",
    "а какие",
    "а какой",
    "а можно",
    "и что",
    "и как",
    "это подходит",
    "подойдет ли",
    "подойдёт ли",
    "а он",
    "а она",
    "а они",
    "а это",
    "и это",
)

_ANAPHORA = (
    "это",
    "этот",
    "эта",
    "эти",
    "этого",
    "этой",
    "такой",
    "такая",
    "такие",
    "такое",
    "тогда",
    "его",
    "её",
    "ее",
    "их",
    "туда",
    "там",
    "дальше",
    "потом",
    "она",
    "они",
    "такое",
)

# Explicit "let's move to a different task" signals. Only these may reset the
# task memory / change the goal.
_SWITCH_CUES = (
    "с этим закончили",
    "с этим всё",
    "с этим все",
    "с этим разобрались",
    "с этим понятно",
    "закончили с",
    "закончили",
    "другой вопрос",
    "другая тема",
    "новая проблема",
    "новая тема",
    "перейдём к",
    "перейдем к",
    "перейти к",
    "теперь меня интересует",
    "теперь хочу",
    "теперь давай",
    "теперь про",
    "а теперь",
    "поговорим о",
    "хочу обсудить",
    "давай про",
    "хватит про",
    "больше не про",
)

# Pure follow-up cores that must NOT be treated as a topic switch even when the
# message starts with "теперь".
_SWITCH_EXCEPTIONS = {
    "что делать",
    "что делать?",
    "что дальше",
    "что дальше?",
    "что теперь",
    "что теперь?",
    "как",
    "как?",
    "куда",
    "куда?",
    "почему",
    "почему?",
    "какой",
    "какой?",
    "какая",
    "какая?",
    "дальше",
    "дальше?",
}

_MODEL_CODE_RE = re.compile(r"\b\d{4}/\d{4}\b")
_FASTENER_RE = re.compile(r"\bM\d+\s*[xх]\s*\d+\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

SHORT_FOLLOWUP_MAX_WORDS = 7
ANAPHORA_MAX_WORDS = 12


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").replace("\n", " ")).strip()


def is_context_dependent_followup(message: str) -> bool:
    """True when the message only makes sense in the current dialogue.

    Heuristic only: short question + question word, or an anaphoric reference.
    A concrete standalone question ("какая защита двигателя подходит для
    Xpander?") is deliberately NOT treated as a follow-up.
    """
    text = normalize(message).lower()
    if not text:
        return False
    core = text.rstrip("?!. ")
    words = core.split()
    if not words:
        return False

    starts_like_question = (
        core.startswith(tuple(_FOLLOWUP_QUESTION_WORDS))
        or core.startswith(tuple(_FOLLOWUP_PHRASES))
    )
    phrase = any(p in core for p in _FOLLOWUP_PHRASES)
    anaphora = any(re.search(rf"\b{re.escape(a)}\b", core) for a in _ANAPHORA)

    if len(words) <= SHORT_FOLLOWUP_MAX_WORDS and (starts_like_question or phrase or anaphora):
        return True
    if anaphora and len(words) <= ANAPHORA_MAX_WORDS:
        return True
    return False


def is_explicit_topic_switch(message: str) -> bool:
    """True only for an explicit "new task" statement, never for a follow-up."""
    text = normalize(message).lower()
    if not text:
        return False
    if any(cue in text for cue in _SWITCH_CUES):
        return True
    core = text.rstrip("?!. ")
    for prefix in ("а теперь ", "теперь "):
        if core.startswith(prefix):
            remainder = core[len(prefix):].strip()
            if remainder in _SWITCH_EXCEPTIONS:
                return False
            return True
    return False


def is_generic_followup_only(message: str) -> bool:
    """True for a bare continuation phrase that carries no topic of its own.

    Those must never become the goal even when they are the first message.
    """
    core = normalize(message).lower().rstrip("?!. ")
    return core in {
        "что делать",
        "что дальше",
        "что теперь",
        "как",
        "куда",
        "почему",
        "зачем",
        "какой",
        "какая",
        "а дальше",
        "дальше",
        "потом",
        "и что",
        "а как",
        "а что",
        "это подходит",
        "подойдет ли",
        "а этот подойдет",
        "а это",
    }


def is_substantive_question(message: str) -> bool:
    """A question that introduces a topic rather than continuing one."""
    text = normalize(message)
    if not text:
        return False
    if is_context_dependent_followup(message):
        return False
    return len(text.split()) >= 2


def last_meaningful_user_message(
    messages: Iterable[Any], *, exclude: str = ""
) -> str:
    """Return the last substantive user message (usually the previous one)."""
    excluded = normalize(exclude).lower()
    result = ""
    for item in messages or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role = getattr(item, "role", "")
            content = getattr(item, "content", "")
        if role != "user":
            continue
        text = normalize(content)
        if not text or text.lower() == excluded:
            continue
        # Skip only bare continuation phrases ("что делать?"); a short but
        # topical question ("какая защита двигателя подходит?") IS the
        # previous meaningful question we want to carry forward.
        if is_generic_followup_only(text):
            continue
        result = text
    return result


def evidence_topic_hint(entries: Iterable[Any], *, max_len: int = 160) -> str:
    """Compact topic hint from persisted evidence (source / section / codes).

    It is used only to carry the PREVIOUS topic into the next retrieval query.
    It is never treated as knowledge: the grounded answer still has to come
    from freshly retrieved chunks.
    """
    sources: list[str] = []
    sections: list[str] = []
    codes: list[str] = []
    for entry in entries or []:
        if isinstance(entry, dict):
            source = entry.get("source") or ""
            section = entry.get("section") or ""
            chunk_text = entry.get("chunk_text") or ""
        else:
            source = getattr(entry, "source", "") or ""
            section = getattr(entry, "section", "") or ""
            chunk_text = getattr(entry, "chunk_text", "") or ""
        if source:
            stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ")
            if stem and stem not in sources:
                sources.append(stem)
        if section and section not in sections:
            sections.append(section)
        for match in _MODEL_CODE_RE.findall(chunk_text):
            if match not in codes:
                codes.append(match)
        for match in _FASTENER_RE.findall(chunk_text):
            normalized_code = match.upper().replace("Х", "x").replace("X", "x")
            if normalized_code not in codes:
                codes.append(normalized_code)

    parts: list[str] = []
    if sources:
        parts.append(sources[0])
    if sections:
        parts.append(sections[0])
    if codes:
        parts.append(" ".join(codes[:3]))
    hint = ", ".join(parts).strip()
    return hint[:max_len]
