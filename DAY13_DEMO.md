# Day 13 demo — Состояние задачи (Task State Machine) (video cheat sheet)

Короткая шпаргалка для записи видео на 4–6 минут. Полная документация — в
[README.md](README.md), раздел «День 13 — Состояние задачи (Task State Machine)».

## Подготовка

```bash
uvicorn app.main:app --reload --port 8000
pytest tests/test_day13.py -q
```

Открыть UI <http://127.0.0.1:8000> и вкладку
**«День 13 — Состояние задачи»**.

## Ключевая идея (проговорить в начале)

Состояние задачи — это **структурированные данные**, а не текст в истории
диалога. Стадию меняет **код** по таблице `ALLOWED_TRANSITIONS`, а не LLM.

```text
planning ──► execution ──► validation ──► done
                 ▲              │
                 └──────────────┘   validation → execution
```

Состояние содержит: `stage`, `current_step`, `expected_action`, `plan`,
`completed_steps`, `paused`. Оно хранится в
`data/day13_task_state.json` и переживает отдельные HTTP-запросы.

## Сценарий записи

1. **Создать задачу.** В поле «Цель задачи» ввести:
   > Спроектируй REST API для небольшого сервиса заметок.
   Нажать **«Создать задачу»**. Показать блок TASK STATE:
   `Stage = PLANNING`, `Current step = 1`,
   `Expected action = Составить план реализации`, `Status = ACTIVE`.
   На диаграмме FSM подсвечен `[PLANNING]`.

2. **Получить план.** В чате отправить `Составь план`.
   Показать переход **кодом** `planning → execution`: стадия стала
   `EXECUTION`, появился план из шагов, `completed_steps` пока пуст.

3. **Выполнить первый шаг.** Отправить `Выполни шаг 1`.
   Показать, что `current_step` изменился на `2`, а выполненный шаг попал в
   `completed_steps`. Открыть **«Показать raw state»** — там реальный JSON.

4. **Pause.** Нажать **Pause**. Показать, что:
   - `Status = PAUSED`;
   - `stage`, `current_step`, `expected_action` **не изменились**;
   - в raw state `"paused": true`.
   Отметить, что `paused` — это флаг, а не отдельная стадия.

5. **Resume без повторного объяснения.** Нажать **Resume**.
   Стадия и шаг те же. В чате отправить только `Продолжай` — **не** повторяя
   задачу, план и текущий шаг. Показать ответ агента: он продолжает с шага 2,
   потому что backend сам подставил `TASK STATE` в запрос к модели.
   Раскрыть «Показать TASK STATE, отправленный в модель».

6. **Дойти до validation.** Отправлять `Продолжай`, пока все шаги плана не
   выполнятся. Код сам переведёт задачу `execution → validation`;
   `Expected action = Проверить полученный результат`.

7. **Validation → done.** Отправить `Проверь результат`. Валидатор отвечает
   verdict `pass`, и **код** выполняет `validation → done`.
   Показать `Status = DONE`. Если бы verdict был `fail`, код вернул бы
   `validation → execution`.

8. **Запрещённый переход.** Показать в терминале (или через API), что
   `planning → done` невозможен:

   ```bash
   curl -X POST http://127.0.0.1:8000/api/day13/task \
     -H "Content-Type: application/json" -d '{"goal":"demo"}'
   curl -X POST http://127.0.0.1:8000/api/day13/transition \
     -H "Content-Type: application/json" -d '{"to_stage":"done"}'
   # -> 400: Недопустимый переход 'planning' -> 'done'
   ```

9. Быстро показать код (по 5–10 секунд):
   - `app/schemas/day13.py` — `TaskState`;
   - `app/services/day13/task_state.py` — `ALLOWED_TRANSITIONS`,
     `TaskStateMachine.transition`;
   - `app/services/day13/service.py` — `_planning_turn` / `_execution_turn` /
     `_validation_turn`, `pause`, `resume`;
   - `app/services/day13/store.py` — persistence.

## Что важно проговорить

- Task State **отделён** от истории сообщений и от памяти.
- LLM генерирует план/ответ/verdict, но **stage ставит код**.
- `expected_action` всегда явно отвечает: «что делать дальше».
- Pause сохраняет место остановки; Resume продолжает с него без объяснений.
- Состояние переживает отдельные HTTP-запросы (JSON-файл). Ограничение:
  задача одна на приложение (учебный single-user сценарий).

## Тесты

```bash
pytest tests/test_day13.py -q
pytest -q
```
