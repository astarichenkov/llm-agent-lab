# Day 11 demo — Memory Layers (video cheat sheet)

Короткая шпаргалка для записи видео на 3–5 минут. Полная документация — в
[README.md](README.md), раздел «День 11 — Модель памяти агента».

## Подготовка

```bash
uvicorn app.main:app --reload --port 8000
```

Открыть UI <http://127.0.0.1:8000> и вкладку
**«День 11 — Модель памяти агента»**.

## Сценарий записи

1. **Открыть вкладку «День 11».** Показать схему вверху и три панели:
   SHORT-TERM / WORKING / LONG-TERM, а также `Last memory decision`.
2. Выставить `Context strategy = Sliding Window`, `window N = 4`.
   Нажать **«Новая сессия»**, чтобы начать с чистого short-term.
3. Отправить первое сообщение:
   > Я Антон. Обычно пишу backend на Python и предпочитаю короткие ответы.
   > Сейчас проектируем сервис бронирования. PostgreSQL запрещён.
4. Показать результат:
   - `Last memory decision`: Working: goal / constraint, Long-term:
     `preferred_language = Python`, `answer_style = concise`;
   - **SHORT-TERM** — исходное сообщение;
   - **WORKING** — цель и ограничение;
   - **LONG-TERM** — устойчивые предпочтения.
5. Отправить 4–5 коротких сообщений-подтверждений (`Спасибо`, `Хорошо`,
   `Продолжай`, `Давай обсудим принципы`).
   Показать, что `Last memory decision = nothing_to_save` — память **не**
   засоряется.
6. Обратить внимание на панель `Context Builder`:
   - `Short-term messages всего` растёт, но `Из них отправлено` = 4;
   - `Выпало (sliding window)` увеличивается;
   - раскрыть «Сообщения, выпавшие из short-term контекста» — исходное
     сообщение там.
7. Отправить контрольный вопрос:
   > Предложи архитектуру хранения данных с учётом моих требований.
8. Показать, что ответ продолжает учитывать Python / краткие ответы /
   запрет PostgreSQL, **хотя исходное сообщение выпало из окна** — потому
   что WORKING и LONG-TERM попали в `included_layers` и в фактический
   prompt (раскрыть «Показать actual long-term / working блоки»).
9. **Показать влияние очистки:**
   - нажать **«Очистить Working»** → задать тот же вопрос → ограничение
     PostgreSQL больше не передаётся из working;
   - нажать **«Очистить Long-term»** → предпочтения больше не передаются;
   - нажать **«Очистить Short-term»** → короткая память очищена, но
     working/long-term остались (независимость слоёв).
10. **Показать persistence:**
    - нажать **«Новая сессия»** → SHORT-TERM пуст, LONG-TERM сохранён;
    - перезапустить `uvicorn` и обновить страницу → LONG-TERM по-прежнему
      на месте (`data/day11_long_term_memory.json`).
11. Быстро показать код (по 5–10 секунд):
    - `app/services/day11/classifier.py` — `MemoryClassifier`;
    - `app/services/day11/store.py` — слои и JSON-persistence;
    - `app/services/day11/service.py` — `_build_context` (Context Builder).

## Что важно проговорить

- Memory Layers — **другая ось**, чем стратегии Дня 10: память решает
  «что хранить», стратегия — «что отправить модели».
- Short-term формируется детерминированно, без LLM.
- MemoryClassifier явно решает, что сохранять, и умеет `nothing_to_save`.
- Ошибка классификатора не ломает основной чат (graceful degradation).
- Long-term хранится отдельно и переживает новую сессию и рестарт.

## Тесты

```bash
pytest tests/test_day11.py -q
pytest -q
```
