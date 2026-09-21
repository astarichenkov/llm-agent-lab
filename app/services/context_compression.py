"""Day 9 — context compression primitives (pure, dependency-free).

The compression policy is deliberately separated from the service so it can be
unit-tested without any provider call or async runtime.

State model
-----------
A compressed dialog keeps THREE clearly separated things:

* ``full_history``     — every message ever exchanged (never deleted; used by
  the UI, by the "no compression" mode and for diagnostics);
* ``summary``          — a compact text block that stands in for the OLD part
  of the history;
* ``summarized_count`` — how many messages FROM THE START of ``full_history``
  are already represented by ``summary``. This is a single integer instead of
  a fragile "message id" tie-in, so the state stays consistent even if the
  history is reloaded or appended to.

The "recent window" is simply the last ``recent_messages_limit`` messages of
``full_history``; everything before it that is not yet summarized is the
"pending" block eligible for the next compression cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# System prompt used ONLY to build/update a summary. It is intentionally kept
# separate from the agent system prompt (the agent prompt instructs the model
# how to ANSWER; this one instructs the model how to COMPRESS).
SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation-memory compressor. You receive an existing "
    "summary of an earlier conversation plus a block of newer messages. "
    "Produce an UPDATED summary that preserves information useful later.\n\n"
    "Always keep, when present:\n"
    "- facts the user stated (names, identifiers, values, entities);\n"
    "- decisions that were made;\n"
    "- requirements and constraints (including prohibitions);\n"
    "- open / unfinished questions;\n"
    "- important context needed to continue the conversation.\n\n"
    "Do NOT retell the conversation verbatim and do NOT add new facts. "
    "Be compact: a short structured list of durable facts is ideal. "
    "Write the summary in the same language the conversation uses.\n"
    "If there is nothing meaningful to remember, say so briefly."
)


@dataclass
class CompressionConfig:
    """Tunable knobs for the compression policy."""

    # Last N messages always sent verbatim.
    recent_messages_limit: int = 6
    # Minimum number of newly-eligible (older-than-window, not-yet-summarized)
    # messages required before a compression cycle runs.
    compression_batch_size: int = 10


@dataclass
class CompressionState:
    """The compression state of ONE dialog.

    ``full_history`` is authoritative and never trimmed. ``summarized_count``
    must always satisfy ``0 <= summarized_count <= len(full_history)``.
    """

    summary: str | None = None
    summarized_count: int = 0
    compression_cycles: int = 0
    # Token accounting for the last summary build (exact API usage when known).
    last_summary_prompt_tokens: int | None = None
    last_summary_completion_tokens: int | None = None
    full_history: list[dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def recent_window(self, config: CompressionConfig) -> list[dict[str, str]]:
        """The last ``recent_messages_limit`` messages, verbatim.

        These are ALWAYS sent to the model, even if they were somehow already
        summarized (the policy never summarizes into the window, but the
        accessor stays defensive).
        """
        limit = max(config.recent_messages_limit, 0)
        if limit == 0:
            return []
        return self.full_history[-limit:]

    def window_start_index(self, config: CompressionConfig) -> int:
        """Index in ``full_history`` where the recent window begins."""
        limit = max(config.recent_messages_limit, 0)
        return max(len(self.full_history) - limit, 0)

    def pending_messages(self, config: CompressionConfig) -> list[dict[str, str]]:
        """Messages older than the window that are NOT yet in the summary.

        This is the ONLY block a new compression cycle may consume.
        """
        start = self.summarized_count
        end = self.window_start_index(config)
        if end <= start:
            return []
        return self.full_history[start:end]

    def next_summarized_count(self, config: CompressionConfig) -> int:
        """``summarized_count`` after folding the pending block into summary."""
        return self.window_start_index(config)

    def should_compress(self, config: CompressionConfig) -> bool:
        """True when enough new messages accumulated to justify a cycle."""
        return len(self.pending_messages(config)) >= max(
            config.compression_batch_size, 1
        )

    def is_consistent(self) -> bool:
        """Guard used before persisting: count must be within bounds."""
        return 0 <= self.summarized_count <= len(self.full_history)

    # ------------------------------------------------------------------
    def build_working_context(
        self, config: CompressionConfig, system_prompt: str | None
    ) -> list[dict[str, str]]:
        """Build the CONTEXT sent to the model (without the current user turn).

        Order: system prompt, then summary (if any), then the recent window.
        Summarized messages are NEVER re-sent verbatim.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": "Summary of the earlier conversation:\n" + self.summary,
                }
            )
        messages.extend(self.recent_window(config))
        return messages


def build_summary_prompt(
    previous_summary: str | None, block: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Build the ``messages`` payload for ONE summary-build call.

    The payload is the dedicated summary system prompt, the previous summary
    (when it exists) and the NEW block only. It never includes the raw full
    history, so the prompt stays bounded.
    """
    parts: list[str] = []
    if previous_summary:
        parts.append("EXISTING SUMMARY:\n" + previous_summary)
    else:
        parts.append("EXISTING SUMMARY:\n(none yet)")
    parts.append("NEW MESSAGES TO INCORPORATE:")
    for message in block:
        role = message.get("role", "?")
        content = message.get("content", "")
        parts.append(f"[{role}] {content}")
    parts.append(
        "Write the updated summary now. Keep it compact and durable."
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
