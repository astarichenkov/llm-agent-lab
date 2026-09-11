# Day 7 demo — сохранение контекста (video cheat sheet)

Короткая шпаргалка для записи видео. Полная документация — в
[README.md](README.md), раздел «День 7 — Сохранение контекста».

## Подготовка

```bash
# локально
uvicorn app.main:app --reload --port 8000
# или в Docker
docker compose up -d
```

Открыть UI (локально <http://127.0.0.1:8000>, в Docker <http://localhost>)
и вкладку **«День 7 — Сохранение контекста»**.

## Сценарий записи

1. **Открыть вкладку «День 7 — Сохранение контекста».**
   Показать блок `Agent / Provider / Model / Context` и чек-лист
   (`✓ хранит history`, `✓ SQLite`, `✓ restore после restart`).
2. Написать и отправить:
   > Меня зовут Алексей. Кодовое слово — капибара. Запомни это.
3. Убедиться, что агент ответил; в блоке контекста стало `2 сообщения`.
4. Спросить в том же чате:
   > Как меня зовут и какое кодовое слово?
   Агент отвечает «Алексей» и «капибара» — контекст работает.
5. **Перезапустить приложение:**
   ```bash
   docker compose restart app      # или Ctrl+C и снова uvicorn
   ```
6. **Обновить страницу** в браузере.
7. Показать, что **история восстановилась** (все сообщения на месте,
   `Context: N сообщений`).
8. Спросить:
   > Какое кодовое слово я называл до перезапуска?
   В ответе присутствует «капибара» — старое сообщение реально ушло в
   LLM-запрос (при желании это видно в логах).
9. Показать, где всё это лежит (быстро, по 5–10 секунд на файл):
   - `app/agents/agent.py` — класс `Agent` (`await agent.chat(...)`);
   - `app/agents/config.py` — `AgentConfig`;
   - `app/agents/manager.py` — `AgentManager.get_or_create("default")`;
   - `app/agents/repository.py` — `SQLiteContextRepository`.
10. (Опционально) показать БД:
    ```bash
    ls -l data/agents.db
    ```
11. (Опционально) показать кнопку **«Очистить контекст»** и/или команду:
    ```bash
    curl -u student:password -X DELETE http://localhost/api/chat/history
    ```

## Что важно проговорить

- `Agent` — тот же класс, что и в Дне 6.
- В Дне 7 он дополнительно владеет контекстом и persistence.
- HTTP-роут тонкий: `agent = await manager.get_or_create("default")`,
  `response = await agent.chat(message)`.
- История хранится в SQLite (`data/agents.db`), разделена по `agent_id`.
- `AgentManager` позволяет создать много агентов:
  `manager.create_agent("agent-a", config)`.
- System prompt берётся из `AgentConfig`, а не пишется в БД.
- После рестарта Python-объекты создаются заново, а контекст
  восстанавливается из SQLite.

## Тесты (показать дополнительно, если останется время)

```bash
pytest tests/test_agent.py -q      # persistence + restart + multi-agent
pytest tests/test_day6.py -q       # Day 6 stateless (для контраста)
pytest -q                          # весь набор
```
