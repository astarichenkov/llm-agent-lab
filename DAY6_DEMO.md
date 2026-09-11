# Day 6 demo — первый агент (video cheat sheet)

Короткая шпаргалка для записи видео. Полная документация — в
[README.md](README.md), раздел «День 6 — Первый агент».

## Подготовка

```bash
uvicorn app.main:app --reload --port 8000
# или в Docker
docker compose up -d
```

Открыть UI (локально <http://127.0.0.1:8000>, в Docker <http://localhost>)
и вкладку **«День 6 — Первый агент»**.

## Сценарий записи

1. **Открыть вкладку «День 6 — Первый агент».**
   Показать блок `Agent / Provider / Model / stateless` и чек-лист:
   `✓ отдельный класс Agent`, `✓ собственный config`, `✓ вызывает LLM API`,
   `✗ история между запросами не сохраняется`.
2. Показать в IDE класс `Agent` (`app/agents/agent.py`): метод `chat()`
   собирает `messages`, вызывает `LLMClient` и возвращает ответ.
3. Коротко показать route (`app/api/routes.py`,
   `POST /api/day6/agent/chat`): он только создаёт transient агента и вызывает
   `await agent.chat(payload.message)` — никакого provider-кода в роуте.
4. В UI написать:
   > Объясни в одном предложении, что такое REST API.
5. Показать ответ и строку `provider/model` под ним.
6. Отправить второй независимый запрос, например:
   > Ответь одним словом: столица Франции?
7. Показать, что предыдущий ответ не используется: блок «Ответ агента»
   заменяется, истории нет.
8. Коротко сказать: контекст между вызовами здесь не сохраняется — это
   следующий этап, День 7 (там тот же `Agent` + SQLite persistence).

## Что важно проговорить

- День 6 — это уже `Agent`, а не просто вызов API из роута.
- Логика запроса/ответа инкапсулирована в `Agent.chat()`.
- На каждый запрос создаётся новый transient Agent
  (`AgentManager.create_transient(...)`) с `InMemoryContextRepository`.
- В LLM уходит только `system prompt + текущий user message`.
- Provider DeepSeek/OpenRouter переиспользуется через существующую
  абстракцию `LLMClient`.

## Тесты (показать дополнительно, если останется время)

```bash
pytest tests/test_day6.py -q       # stateless-поведение
pytest -q                          # весь набор
```
