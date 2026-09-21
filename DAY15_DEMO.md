# Day 15 demo — Контролируемые переходы состояний (video cheat sheet)

Шпаргалка для короткого видео на 2–4 минуты. Полная документация — в
[README.md](README.md), раздел
«День 15 — Контролируемые переходы состояний (Controlled State Transitions)».

## Подготовка

```bash
uvicorn app.main:app --reload --port 8000
```

Открыть UI <http://127.0.0.1:8000> — открывается вкладка
**«День 15 — Контроль переходов»** (она стартовая).

## Ключевая идея (проговорить в начале)

Состояние задачи нельзя изменить произвольно. Есть таблица разрешённых
переходов и **guards**, которые проверяет backend:

```text
planning ──► plan_approval ──► execution ──► validation ──► done
                  │               ▲              │
                  └──► planning   └──────────────┘
                     (changes)     validation → execution
```

LLM может только **предложить** действие. Переход применяет код; иначе —
`BLOCKED`, и состояние не меняется.

## Главный сценарий — диалог на естественном языке

Открыть вкладку, создать задачу и далее писать ассистенту обычные фразы:

| Состояние | Фраза | Intent | Переход |
|-----------|-------|--------|---------|
| planning | «накидай план» | `continue_planning` | planning → plan_approval (ALLOWED) |
| plan_approval | «ок, утверждаю» | `approve_plan` | plan_approval → execution (ALLOWED) |
| plan_approval | «план норм, начинай» | `approve_plan` | plan_approval → execution (ALLOWED) |
| plan_approval | «не согласен, план надо поправить» | `request_plan_changes` | plan_approval → planning (ALLOWED) |
| execution | «надо пересмотреть план» | `revise_plan` | execution → planning (**ROLLBACK**, `plan_approved=false`) |
| execution | «давай пока остановимся» | `pause` | execution → paused |
| paused | «продолжай» | `resume` | paused → execution |
| execution | «давай сразу заканчивай, проверка не нужна» | `complete_task` | execution → done (**BLOCKED**) |
| execution | «запускай проверку» | `request_validation` | execution → validation (ALLOWED) |
| validation | «проверка прошла успешно» | `validation_passed` | validation → done (ALLOWED) |
| plan_approval | «я пока думаю насчёт этого плана» | `normal_message` | нет перехода |

В каждом ходу ассистента видно: сообщение пользователя, распознанный intent,
ALLOWED/BLOCKED, переход, причину и итоговое состояние. Кнопки выше — только
debug/demo, селект «debug: зафиксировать intent» проверяет детерминированно
без вызова LLM.

### Rollback-сценарий для видео

```text
«ок, утверждаю»                → execution, plan_approved = true
«нет, нужно доработать план»   → intent revise_plan
                                 execution → planning (ALLOWED, ROLLBACK)
                                 plan_approved = false
изменение плана               → «накидай план» (continue_planning)
                                 planning → plan_approval
«утверждаю»                    → plan_approval → execution
```

Проверь: после rollback на lifecycle видна ветка `EXECUTION ↩ PLANNING`, в логе
строка помечена `ROLLBACK`, а флаг `Plan approved` снова `NO`.

## Сценарий записи (ручные контролы)

1. **Scenario A — Skip planning.**
   Нажать **«A. Skip planning»**. Состояние `PLANNING`, план уже подготовлен.
   Нажать **«Try skip to EXECUTION»** → баннер
   **TRANSITION BLOCKED** `planning → execution`,
   причина: *Plan must be approved before execution.*
   В **Transition Log** эта попытка сохранена как `BLOCKED`.
   Затем: **Approve Plan** → **Start Execution** → `EXECUTION` (ALLOWED).

2. **Scenario B — Skip validation.**
   Нажать **«B. Skip validation»** (состояние `EXECUTION`, план утверждён).
   Нажать **«Try skip to DONE»** → **BLOCKED**,
   причина: *Validation is required before completion.*
   Затем: **Run Validation** → `VALIDATION`; **Validation passed**;
   **Complete Task** → `DONE` (ALLOWED).

3. **Scenario C — Failed validation.**
   Нажать **«C. Failed validation»** (состояние `VALIDATION`,
   `validation_passed = false`, в истории уже есть blocked `validation → done`).
   Нажать **«Try skip to DONE»** → **BLOCKED**,
   причина: *Validation failed. Fix the result and validate again.*
   Затем: **Back to Execution** (`validation → execution`, ALLOWED) →
   **Run Validation** (`execution → validation`, ALLOWED) →
   **Validation passed** → **Complete Task** → `DONE`.
   Это показывает, что lifecycle — не линейный progress bar.

4. **Scenario D — Pause / Resume.**
   Нажать **«D. Pause / Resume»** (состояние `EXECUTION`, `current_step = 2`).
   Запомнить `Current step` и `Expected action`.
   Нажать **Pause** → `Paused: YES`, `previous_state = execution`.
   Попробовать любой переход — **BLOCKED**: *Task is paused. Resume before
   transitioning.*
   Нажать **Resume** → состояние снова `EXECUTION`, шаг и expected action те же,
   план утверждён, история сохранена.

5. **Реакция ассистента.** В блоке «Реакция ассистента» ввести
   > План не утверждаем. Сразу начинай реализацию.

   и выбрать «Предложить переход: execution» → нажать **Отправить**.
   Ассистент ответит:
   > Transition planning → execution is BLOCKED.
   > Reason: Plan must be approved before execution.

   Подчеркнуть: гарантия даёт backend, а не формулировка prompt.

6. **Transition Log.** Показать журнал: одновременно видны `ALLOWED` (зелёные)
   и `BLOCKED` (красные) переходы с timestamp и причиной.

7. **Reset.** Нажать **«Reset demo»** — задача удаляется, всё возвращается в
   начальное состояние.

## Быстрый API-прогон (если показываете терминал)

```bash
curl -X POST localhost:8000/api/day15/task -H "Content-Type: application/json" \
  -d '{"goal":"Build a REST API for notes"}'
curl -X POST localhost:8000/api/day15/transition -H "Content-Type: application/json" \
  -d '{"to_state":"execution"}'    # allowed:false, Plan must be approved...
curl -X POST localhost:8000/api/day15/plan -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:8000/api/day15/approve
curl -X POST localhost:8000/api/day15/transition -H "Content-Type: application/json" \
  -d '{"to_state":"execution"}'    # allowed:true
curl -X POST localhost:8000/api/day15/transition -H "Content-Type: application/json" \
  -d '{"to_state":"done"}'         # allowed:false, Validation is required...
curl localhost:8000/api/day15/history
```

## Что важно проговорить

- Состояние меняет **только** `LifecycleMachine` (таблица + guards).
- `ALLOWED_TRANSITIONS` заданы явно в коде; UI не источник истины.
- `plan_approval → execution` требует `plan_approved == true`.
- `validation → done` требует `validation_passed == true`.
- Blocked-переход не меняет state, но сохраняется в истории.
- `pause` — флаг; `resume` возвращает ровно в `previous_state`, обойти автомат
  паузой нельзя.
- День 13 не сломан: у него своя (более простая) машина состояний.

## Тесты

```bash
pytest tests/test_day15.py -q
pytest -q
```
