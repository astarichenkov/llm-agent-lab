"""Day 12 — user profile / personalization tests.

No real DeepSeek call is made: a deterministic provider records every message
payload. The tests assert on the PROFILE STORAGE, the generated instructions
and the fact that the profile block is injected into the model request — never
on the free-form LLM answer (which is non-deterministic).
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.schemas.day11 import LongTermMemory, WorkingMemory
from app.schemas.day12 import (
    ResponseConstraints,
    ResponseFormat,
    UserProfile,
)
from app.services.day11.service import Day11MemoryService
from app.services.day11.store import LongTermMemoryStore
from app.services.day12.profile import (
    DEFAULT_PROFILE,
    PROFILE_PRESETS,
    ProfileStore,
    build_personalization_instructions,
    get_preset,
)
from app.services.day12.service import Day12ProfileService


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class RecordingProvider:
    """Records every provider payload and returns a fixed answer."""

    def __init__(self, answer: str = "OK-answer") -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.answer = answer

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append([dict(m) for m in messages])
        return self.answer, "stop", {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }

    def last(self) -> list[dict[str, str]]:
        return self.calls[-1] if self.calls else []


def _settings(tmp_path) -> Settings:
    return Settings(
        deepseek_api_key="test",
        environment="test",
        day11_long_term_path=str(tmp_path / "long_term.json"),
        day12_profile_path=str(tmp_path / "profile.json"),
    )


def _services(provider, tmp_path):
    settings = _settings(tmp_path)
    memory = Day11MemoryService(
        settings,
        deepseek=provider,
        long_term_store=LongTermMemoryStore(settings.day11_long_term_path),
    )
    profile_service = Day12ProfileService(
        settings,
        memory_service=memory,
        profile_store=ProfileStore(settings.day12_profile_path),
    )
    return settings, memory, profile_service


def _all_content(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages)


# ----------------------------------------------------------------------
# 1) Profile storage
# ----------------------------------------------------------------------
def test_profile_store_saves_and_reloads(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    profile = UserProfile(
        name="Антон",
        style="technical",
        format=ResponseFormat(structured=True, use_code=True),
        constraints=ResponseConstraints(no_intro=True, language="ru"),
    )
    store.save(profile)

    reloaded = ProfileStore(tmp_path / "profile.json").load()
    assert reloaded == profile
    assert reloaded.name == "Антон"


def test_profile_store_roundtrips_custom_instructions(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    profile = UserProfile(custom_instructions="Пиши только транслитом.")
    store.save(profile)

    reloaded = ProfileStore(tmp_path / "profile.json").load()
    assert reloaded.custom_instructions == "Пиши только транслитом."


def test_profile_store_default_when_missing(tmp_path):
    store = ProfileStore(tmp_path / "does-not-exist.json")
    assert store.load() == DEFAULT_PROFILE


def test_profile_update_is_persisted_and_used(tmp_path):
    provider = RecordingProvider()
    settings, memory, service = _services(provider, tmp_path)
    new_profile = UserProfile(style="detailed", constraints=ResponseConstraints(no_intro=False))
    service.update_profile(new_profile)

    assert service.profile.style == "detailed"
    # persisted: a fresh store reads the SAME profile
    assert ProfileStore(settings.day12_profile_path).load().style == "detailed"
    assert "detailed" in service.instructions.lower()


def test_apply_preset_persists(tmp_path):
    provider = RecordingProvider()
    settings, memory, service = _services(provider, tmp_path)
    service.apply_preset("technical")
    assert service.profile.style == "technical"
    assert ProfileStore(settings.day12_profile_path).load().style == "technical"
    with pytest.raises(ValueError):
        service.apply_preset("nope")


# ----------------------------------------------------------------------
# 2) Profile -> instructions (single place)
# ----------------------------------------------------------------------
def test_instruction_builder_style_variants_differ():
    blocks = {
        style: build_personalization_instructions(UserProfile(style=style))
        for style in ("concise", "detailed", "technical")
    }
    assert len(set(blocks.values())) == 3
    assert "concise" in blocks["concise"].lower()
    assert "detailed" in blocks["detailed"].lower()
    assert "technical" in blocks["technical"].lower()
    assert "USER PROFILE" in blocks["concise"]


def test_instruction_builder_format_and_constraints():
    profile = UserProfile(
        format=ResponseFormat(structured=True, use_lists=True, use_examples=True, use_code=True),
        constraints=ResponseConstraints(no_emoji=True, no_intro=True, language="ru"),
    )
    block = build_personalization_instructions(profile)
    assert "sections" in block
    assert "bullet lists" in block
    assert "examples" in block
    assert "code" in block
    assert "emoji" in block
    assert "introductions" in block
    assert "Russian" in block


def test_custom_instructions_land_in_personalization_block():
    profile = UserProfile(custom_instructions="Пиши только транслитом.")
    block = build_personalization_instructions(profile)
    assert "Additional user instructions:" in block
    assert "Пиши только транслитом." in block
    # free-form instructions stay personalization, the priority rule stays last
    assert block.index("Пиши только транслитом.") < block.index(
        "system rules always win"
    )
    assert block.rstrip().endswith("system rules always win.")


def test_no_lists_constraint_overrides_use_lists():
    profile = UserProfile(
        format=ResponseFormat(use_lists=True),
        constraints=ResponseConstraints(no_lists=True),
    )
    block = build_personalization_instructions(profile)
    assert "Do not use bullet lists" in block
    assert "Use bullet lists" not in block


def test_profile_block_keeps_system_priority():
    block = build_personalization_instructions(UserProfile())
    lowered = block.lower()
    assert "system" in lowered and "win" in lowered
    assert "never override" in lowered


# ----------------------------------------------------------------------
# 3) Request integration + automatic personalization
# ----------------------------------------------------------------------
async def test_chat_injects_profile_block(tmp_path):
    provider = RecordingProvider()
    _, _, service = _services(provider, tmp_path)
    service.update_profile(UserProfile(name="Антон", style="technical"))

    await service.chat(message="Что такое индекс в базе данных?", classify=False)

    messages = provider.last()
    joined = _all_content(messages)
    # profile instructions are present ...
    assert "USER PROFILE" in joined
    assert "Антон" in joined
    assert "technical" in joined.lower()
    # ... the user message is unchanged and appears exactly once
    assert [m["content"] for m in messages].count("Что такое индекс в базе данных?") == 1
    # ... and the profile block is a SYSTEM message placed after the app system
    # prompt, i.e. application rules keep priority.
    contents = [m["content"] for m in messages]
    system_indexes = [i for i, m in enumerate(messages) if m["role"] == "system"]
    assert contents[system_indexes[0]] != joined  # app system prompt is first
    assert any("USER PROFILE" in contents[i] for i in system_indexes[1:])


async def test_chat_injects_custom_instructions_automatically(tmp_path):
    """Сценарий 1: стиль меняется без инструкции в самом вопросе.

    Реальный LLM-ответ недетерминирован, поэтому тест проверяет механизм:
    свободная инструкция уходит в personalization block автоматически, а текст
    вопроса остаётся без каких-либо указаний по стилю.
    """
    provider = RecordingProvider()
    _, _, service = _services(provider, tmp_path)
    service.update_profile(UserProfile(custom_instructions="Отвечай как быдло."))

    question = "Объясни, зачем программисту нужны автоматические тесты."
    await service.chat(message=question, classify=False)

    joined = _all_content(provider.last())
    assert "Additional user instructions:" in joined
    assert "Отвечай как быдло." in joined
    # Стиль нигде не задаётся в самом сообщении.
    assert "как быдло" not in question.lower()
    assert "стиль" not in question.lower()


async def test_chat_injects_translit_custom_instruction(tmp_path):
    """Сценарий 2: тот же вопрос, но свободная инструкция «транслитом»."""
    provider = RecordingProvider()
    _, _, service = _services(provider, tmp_path)
    service.update_profile(UserProfile(custom_instructions="Пиши только транслитом."))

    question = "Объясни, зачем программисту нужны автоматические тесты."
    await service.chat(message=question, classify=False)

    joined = _all_content(provider.last())
    assert "Пиши только транслитом." in joined
    assert "Additional user instructions:" in joined
    # инструкция попадает в запрос, не требуя повторения в сообщении
    assert "транслит" not in question.lower()


async def test_automatic_personalization_without_style_in_message(tmp_path):
    """The user message has NO style instructions, yet the profile is applied."""
    provider = RecordingProvider()
    _, _, service = _services(provider, tmp_path)
    service.update_profile(UserProfile(style="detailed"))

    question = "Как работает HTTP?"
    await service.chat(message=question, classify=False)

    joined = _all_content(provider.last())
    assert "detailed" in joined.lower()
    assert "answer style" not in question.lower()
    assert "подроб" not in question.lower()


async def test_different_profiles_produce_different_requests(tmp_path):
    provider = RecordingProvider()
    _, memory, service = _services(provider, tmp_path)
    question = "Что такое кэш?"

    service.update_profile(get_preset("concise").profile)
    await service.chat(message=question, classify=False)
    concise_messages = provider.last()

    # Start a clean short-term window so the next request is comparable.
    memory.reset_short_term()
    service.update_profile(get_preset("technical").profile)
    await service.chat(message=question, classify=False)
    technical_messages = provider.last()

    # Same user message ...
    assert [m["content"] for m in concise_messages].count(question) == 1
    assert [m["content"] for m in technical_messages].count(question) == 1
    # ... but different personalization instructions.
    assert _all_content(concise_messages) != _all_content(technical_messages)
    assert "concise" in _all_content(concise_messages).lower()
    assert "technical" in _all_content(technical_messages).lower()


def test_three_distinct_presets_exist():
    assert [p.id for p in PROFILE_PRESETS] == ["concise", "detailed", "technical"]
    blocks = [build_personalization_instructions(p.profile) for p in PROFILE_PRESETS]
    assert len(set(blocks)) == 3


# ----------------------------------------------------------------------
# 4) Personalization ON TOP of memory
# ----------------------------------------------------------------------
async def test_profile_and_memory_are_both_in_request(tmp_path):
    provider = RecordingProvider()
    _, memory, service = _services(provider, tmp_path)
    memory.set_working(
        WorkingMemory(goal="Спроектировать сервис бронирования", constraints=["Не использовать PostgreSQL"])
    )
    memory.set_long_term(
        LongTermMemory(entries={"preferred_language": "Python", "answer_style": "concise"})
    )
    service.update_profile(UserProfile(name="Антон", style="detailed"))

    response = await service.chat(message="Предложи архитектуру", classify=False)

    joined = _all_content(provider.last())
    # profile
    assert "USER PROFILE" in joined and "detailed" in joined.lower()
    # memory (reused Day 11 layers, not duplicated into the profile)
    assert "LONG-TERM MEMORY" in joined
    assert "preferred_language = Python" in joined
    assert "WORKING MEMORY" in joined
    assert "Не использовать PostgreSQL" in joined
    # diagnostics reflect both
    assert response.context.personalization_included is True
    assert response.context.long_term_included is True
    assert response.context.working_included is True
    assert response.usage.personalization_tokens_estimated > 0

    # priority order inside the payload: system -> profile -> memory
    contents = [m["content"] for m in provider.last()]
    profile_index = next(i for i, c in enumerate(contents) if "USER PROFILE" in c)
    memory_index = next(i for i, c in enumerate(contents) if "LONG-TERM MEMORY" in c)
    assert profile_index < memory_index


async def test_apply_profile_false_omits_block(tmp_path):
    provider = RecordingProvider()
    _, _, service = _services(provider, tmp_path)
    await service.chat(message="x", classify=False, apply_profile=False)
    assert "USER PROFILE" not in _all_content(provider.last())


# ----------------------------------------------------------------------
# 5) Compare (same question, different profiles)
# ----------------------------------------------------------------------
async def test_compare_runs_isolated_and_restores_session(tmp_path):
    provider = RecordingProvider(answer="ответ")
    _, memory, service = _services(provider, tmp_path)
    memory.set_long_term(LongTermMemory(entries={"preferred_language": "Python"}))
    before_session = memory.state().session_id
    before_short_term = memory.state().short_term_count

    result = await service.compare(message="Что такое REST?")

    assert len(result.items) == 3
    assert [item.preset_id for item in result.items] == ["concise", "detailed", "technical"]
    # only the profile differs
    assert len({item.personalization_block for item in result.items}) == 3
    # user's session + memory are untouched
    after = memory.state()
    assert after.session_id == before_session
    assert after.short_term_count == before_short_term
    assert after.long_term.entries == {"preferred_language": "Python"}
    # compare did not bring memory into the isolated requests
    for call in provider.calls:
        assert "LONG-TERM MEMORY" not in _all_content(call)
        assert "USER PROFILE" in _all_content(call)


async def test_compare_selects_requested_presets(tmp_path):
    provider = RecordingProvider()
    _, _, service = _services(provider, tmp_path)
    result = await service.compare(message="x", preset_ids=["concise", "technical"])
    assert [item.preset_id for item in result.items] == ["concise", "technical"]


# ----------------------------------------------------------------------
# 6) API level
# ----------------------------------------------------------------------
def test_api_get_profile(client):
    resp = client.get("/api/day12/profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["style"] == "concise"
    assert "USER PROFILE" in body["instructions"]
    assert [p["id"] for p in body["presets"]] == ["concise", "detailed", "technical"]


def test_api_put_profile_validates_and_persists(client):
    payload = {
        "name": "Антон",
        "style": "technical",
        "format": {"structured": True, "use_lists": True, "use_examples": False, "use_code": True},
        "constraints": {"no_emoji": True, "no_intro": True, "no_lists": False, "language": "ru"},
        "custom_instructions": "Пиши только транслитом",
    }
    resp = client.put("/api/day12/profile", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["style"] == "technical"
    assert body["profile"]["name"] == "Антон"
    assert body["profile"]["custom_instructions"] == "Пиши только транслитом"
    assert "technical" in body["instructions"].lower()
    assert "Additional user instructions:" in body["instructions"]
    assert "Пиши только транслитом" in body["instructions"]

    # reads back the saved profile
    assert client.get("/api/day12/profile").json()["profile"]["style"] == "technical"
    assert (
        client.get("/api/day12/profile").json()["profile"]["custom_instructions"]
        == "Пиши только транслитом"
    )


def test_api_put_profile_rejects_invalid(client):
    resp = client.put("/api/day12/profile", json={"style": "invalid-style"})
    assert resp.status_code == 422


def test_api_apply_preset_and_unknown(client):
    ok = client.post("/api/day12/profile/preset/detailed")
    assert ok.status_code == 200
    assert ok.json()["profile"]["style"] == "detailed"

    missing = client.post("/api/day12/profile/preset/nope")
    assert missing.status_code == 400


def test_api_day12_chat_uses_profile(client, fake_service):
    client.put(
        "/api/day12/profile",
        json={"style": "technical", "constraints": {"no_emoji": True, "language": "ru"}},
    )
    resp = client.post("/api/day12/chat", json={"message": "Что такое индекс?", "classify": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mocked answer from DeepSeek."
    assert body["profile"]["style"] == "technical"
    assert body["profile_applied"] is True
    assert "USER PROFILE" in body["personalization_block"]
    assert body["context"]["personalization_included"] is True

    # the actual provider request really carried the profile block
    sent = fake_service.generate_calls[-1]["messages"]
    assert any("USER PROFILE" in m["content"] for m in sent)


def test_api_day12_compare_returns_three_profiles(client, fake_service):
    resp = client.post("/api/day12/compare", json={"message": "Один и тот же вопрос"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Один и тот же вопрос"
    assert len(body["items"]) == 3
    assert len({item["personalization_block"] for item in body["items"]}) == 3


def test_api_day12_does_not_touch_day7_history(client):
    client.post("/api/day12/chat", json={"message": "hello", "classify": False})
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_old_endpoints_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/day11/state").status_code == 200
    assert client.get("/api/day10/state", params={"strategy": "sliding_window"}).status_code == 200
