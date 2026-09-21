"""Domain models for the Day 10 branching dialog.

Branching is a DIFFERENT axis than Sliding Window / Sticky Facts:

* Sliding Window / Sticky Facts answer "what from the past goes into context?";
* Branching answers "which version of the history is currently active?".

The model here keeps messages OUTSIDE the branches and references them by a
stable ``branch_id``. A branch never copies its parent's messages: the common
prefix is computed by walking ``parent_branch_id`` up to
``checkpoint_message_id``. That guarantees two sibling branches share the
prefix but stay isolated afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


@dataclass
class BranchMessage:
    """One user/assistant turn owned by exactly ONE branch."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    branch_id: str
    created_at: str = field(default_factory=utc_now_iso)

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Branch:
    """A named line of dialog grown from a parent branch checkpoint."""

    id: str
    name: str
    parent_branch_id: str | None = None
    checkpoint_message_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class Conversation:
    """A conversation tree: branches + messages + the active branch."""

    id: str
    branches: dict[str, Branch] = field(default_factory=dict)
    messages: list[BranchMessage] = field(default_factory=list)
    active_branch_id: str = ""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def create_root(cls, main_name: str = "Main") -> "Conversation":
        branch_id = new_id("branch")
        conv = cls(id=new_id("conv"))
        conv.branches[branch_id] = Branch(id=branch_id, name=main_name)
        conv.active_branch_id = branch_id
        return conv

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    @property
    def active_branch(self) -> Branch:
        return self.branches[self.active_branch_id]

    def get_branch(self, branch_id: str) -> Branch:
        if branch_id not in self.branches:
            raise KeyError(branch_id)
        return self.branches[branch_id]

    def own_messages(self, branch_id: str) -> list[BranchMessage]:
        """Messages created IN ``branch_id`` (not inherited)."""
        return [m for m in self.messages if m.branch_id == branch_id]

    def _message_by_id(self, message_id: str) -> BranchMessage | None:
        return next((m for m in self.messages if m.id == message_id), None)

    def _full_history(self, branch_id: str) -> list[BranchMessage]:
        """Inherited prefix + own messages, in order."""
        return self._inherited_prefix(branch_id) + self.own_messages(branch_id)

    def _inherited_prefix(self, branch_id: str) -> list[BranchMessage]:
        branch = self.branches[branch_id]
        if branch.parent_branch_id is None:
            return []
        parent_history = self._full_history(branch.parent_branch_id)
        if branch.checkpoint_message_id:
            for index, message in enumerate(parent_history):
                if message.id == branch.checkpoint_message_id:
                    return parent_history[: index + 1]
        # Defensive: a missing checkpoint falls back to the whole parent path.
        return parent_history

    def branch_history(self, branch_id: str) -> list[BranchMessage]:
        """Everything the model sees for ``branch_id`` (no current turn)."""
        return self._full_history(branch_id)

    def branch_history_payload(self, branch_id: str) -> list[dict[str, str]]:
        return [m.as_payload() for m in self.branch_history(branch_id)]

    def checkpoint_index(self, branch_id: str) -> int | None:
        """1-based position of the branch checkpoint in its parent history."""
        branch = self.branches[branch_id]
        if not branch.checkpoint_message_id or not branch.parent_branch_id:
            return None
        prefix = self._inherited_prefix(branch_id)
        for index, message in enumerate(prefix):
            if message.id == branch.checkpoint_message_id:
                return index + 1
        return None

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def add_message(
        self, role: Literal["user", "assistant"], content: str, branch_id: str | None = None
    ) -> BranchMessage:
        target = branch_id or self.active_branch_id
        self.get_branch(target)  # validate
        message = BranchMessage(
            id=new_id("msg"), role=role, content=content, branch_id=target
        )
        self.messages.append(message)
        return message

    def create_branch(
        self,
        *,
        name: str | None = None,
        parent_branch_id: str | None = None,
        checkpoint_message_id: str | None = None,
    ) -> Branch:
        parent_id = parent_branch_id or self.active_branch_id
        self.get_branch(parent_id)  # validate parent exists

        # Default checkpoint: the LAST message of the parent's full history.
        if checkpoint_message_id is None:
            parent_history = self._full_history(parent_id)
            if parent_history:
                checkpoint_message_id = parent_history[-1].id
        elif self._message_by_id(checkpoint_message_id) is None:
            raise ValueError(f"checkpoint message not found: {checkpoint_message_id}")

        if checkpoint_message_id is not None:
            # The checkpoint must belong to the parent's history.
            parent_ids = {m.id for m in self._full_history(parent_id)}
            if checkpoint_message_id not in parent_ids:
                raise ValueError("checkpoint does not belong to the parent branch")

        auto_name = name.strip() if name and name.strip() else self._auto_name(parent_id)
        branch = Branch(
            id=new_id("branch"),
            name=auto_name,
            parent_branch_id=parent_id,
            checkpoint_message_id=checkpoint_message_id,
        )
        self.branches[branch.id] = branch
        self.active_branch_id = branch.id
        return branch

    def _auto_name(self, parent_id: str) -> str:
        used = sum(
            1 for b in self.branches.values() if b.parent_branch_id == parent_id
        )
        return f"Branch {used + 1}"

    def activate(self, branch_id: str) -> Branch:
        branch = self.get_branch(branch_id)
        self.active_branch_id = branch_id
        return branch

    def reset(self, main_name: str = "Main") -> None:
        fresh = Conversation.create_root(main_name)
        self.id = fresh.id
        self.branches = fresh.branches
        self.messages = fresh.messages
        self.active_branch_id = fresh.active_branch_id
