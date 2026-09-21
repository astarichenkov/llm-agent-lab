/* LLM Agent Lab — Day 15: controlled state transitions.
 * Vanilla JS. Talks to /api/day15/*. The backend owns the lifecycle state and
 * enforces the allowed transitions + guards; this file only visualises the
 * state, the lifecycle chain, the allowed/locked transitions and the log.
 * It never decides a transition itself.
 */
(function () {
  "use strict";

  function initDay15() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day15", "panel-day15",
      "d15-current-state", "d15-current-step", "d15-expected-action",
      "d15-plan-approved", "d15-validation-passed", "d15-paused",
      "d15-lifecycle", "d15-state-planning", "d15-state-plan_approval",
      "d15-state-execution", "d15-state-validation", "d15-state-done",
      "d15-rollback",
      "d15-lifecycle-note", "d15-allowed-title", "d15-allowed-list",
      "d15-goal", "d15-create", "d15-prepare", "d15-approve",
      "d15-start-execution", "d15-run-validation", "d15-back-execution",
      "d15-revise",
      "d15-pass", "d15-fail",
      "d15-complete", "d15-pause", "d15-resume", "d15-reset", "d15-loading",
      "d15-skip-approval", "d15-skip-execution", "d15-skip-done",
      "d15-decision", "d15-decision-status", "d15-decision-text",
      "d15-decision-reason", "d15-error", "d15-note",
      "d15-scenario-happy", "d15-scenario-skip-planning",
      "d15-scenario-skip-validation", "d15-scenario-failed-validation",
      "d15-scenario-pause-resume", "d15-log",
      "d15-messages", "d15-input", "d15-propose", "d15-send", "d15-raw"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day15: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var STAGE_ORDER = ["planning", "plan_approval", "execution", "validation", "done"];
    var STAGE_LABEL = {
      planning: "PLANNING",
      plan_approval: "PLAN_APPROVAL",
      execution: "EXECUTION",
      validation: "VALIDATION",
      done: "DONE"
    };
    var state = { inFlight: false, hasTask: false, task: null };

    /* ---------------- helpers ---------------- */
    function setError(m) {
      $("d15-error").textContent = m || "";
      $("d15-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d15-note").textContent = m || "";
      $("d15-note").classList.toggle("hidden", !m);
    }
    function setLoading(on) {
      state.inFlight = on;
      $("d15-loading").classList.toggle("hidden", !on);
      updateButtons();
    }
    function http(path, opts) {
      return fetch(path, opts).then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, status: r.status, data: d };
        });
      });
    }
    function element(tag, className, text) {
      var el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined && text !== null) el.textContent = text;
      return el;
    }
    function detailMessage(data, fallback) {
      var detail = data && data.detail;
      if (Array.isArray(detail)) {
        detail = detail.map(function (d) { return d.msg || String(d); }).join("; ");
      }
      return detail || fallback;
    }
    function yesNo(value) {
      if (value === true) return "YES";
      if (value === false) return "NO";
      return "—";
    }
    function timeOf(iso) {
      if (!iso) return "";
      var d = new Date(iso);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleTimeString();
    }

    /* ---------------- button availability ---------------- */
    function task() { return state.task; }
    function isState(name) { return !!task() && task().state === name; }
    function updateButtons() {
      var inFlight = state.inFlight;
      var t = task();
      var active = state.hasTask && t && !t.paused && t.state !== "done";
      $("d15-create").disabled = inFlight;
      $("d15-reset").disabled = inFlight;
      $("d15-prepare").disabled = inFlight || !isState("planning");
      $("d15-approve").disabled = inFlight || !isState("plan_approval");
      $("d15-start-execution").disabled = inFlight || !(isState("plan_approval") && t.plan_approved);
      $("d15-run-validation").disabled = inFlight || !isState("execution");
      $("d15-back-execution").disabled = inFlight || !isState("validation");
      $("d15-revise").disabled = inFlight || !isState("execution");
      $("d15-pass").disabled = inFlight || !isState("validation");
      $("d15-fail").disabled = inFlight || !isState("validation");
      $("d15-complete").disabled = inFlight || !(isState("validation") && t.validation_passed === true);
      $("d15-pause").disabled = inFlight || !active;
      $("d15-resume").disabled = inFlight || !(state.hasTask && t && t.paused);
      $("d15-send").disabled = inFlight || !state.hasTask;
      $("d15-skip-approval").disabled = inFlight || !active;
      $("d15-skip-execution").disabled = inFlight || !active;
      $("d15-skip-done").disabled = inFlight || !active;
    }

    /* ---------------- rendering ---------------- */
    function renderLifecycle(t) {
      var currentIndex = t ? STAGE_ORDER.indexOf(t.state) : -1;
      STAGE_ORDER.forEach(function (name, index) {
        var node = $("d15-state-" + name);
        if (!node) return;
        node.classList.remove("active", "passed", "future");
        if (!t) { node.classList.add("future"); return; }
        if (index < currentIndex) node.classList.add("passed");
        else if (index === currentIndex) node.classList.add("active");
        else node.classList.add("future");
      });
      if (!t) {
        $("d15-lifecycle-note").textContent = "Задача ещё не создана.";
      } else if (t.paused) {
        $("d15-lifecycle-note").textContent =
          "PAUSED from " + STAGE_LABEL[t.previous_state || t.state] +
          " — resume вернёт именно это состояние.";
      } else {
        $("d15-lifecycle-note").textContent =
          "Текущее состояние: " + STAGE_LABEL[t.state] +
          (t.plan_approved ? " · план утверждён" : "") +
          (t.validation_passed === true ? " · валидация пройдена" : "");
      }
      var rollback = $("d15-rollback");
      if (rollback) {
        rollback.classList.toggle("available", !!t && t.state === "execution");
        rollback.classList.remove("active");
        var history = (t && t.transitions) || [];
        var last = history.length ? history[history.length - 1] : null;
        if (last && last.from_state === "execution" && last.to_state === "planning") {
          rollback.classList.add("active");
        }
      }
    }

    function renderAllowed(t, allowed, options) {
      var list = $("d15-allowed-list");
      list.innerHTML = "";
      if (!t) {
        $("d15-allowed-title").textContent = "Allowed transitions: —";
        return;
      }
      $("d15-allowed-title").textContent =
        "Allowed transitions from " + STAGE_LABEL[t.state] + ":";
      var items = options.slice();
      if (allowed && allowed.length && !items.length) {
        items = allowed.map(function (name) {
          return { state: name, allowed: true, locked_reason: "" };
        });
      }
      items.forEach(function (option) {
        var row = element("li", "d15-allowed-item " + (option.allowed ? "ok" : "locked"));
        row.appendChild(element(
          "span", "d15-allowed-mark", option.allowed ? "✓" : "🔒"
        ));
        row.appendChild(element(
          "span", "d15-allowed-name", STAGE_LABEL[option.state] || option.state.toUpperCase()
        ));
        if (!option.allowed && option.locked_reason) {
          row.appendChild(element(
            "span", "d15-allowed-reason", "Locked: " + option.locked_reason
          ));
        } else if (option.requires) {
          row.appendChild(element(
            "span", "d15-allowed-reason", "requires " + option.requires
          ));
        }
        list.appendChild(row);
      });
      var pauseRow = element("li", "d15-allowed-item " + (t.paused ? "locked" : "ok"));
      pauseRow.appendChild(element("span", "d15-allowed-mark", t.paused ? "🔒" : "✓"));
      pauseRow.appendChild(element("span", "d15-allowed-name", "PAUSE"));
      list.appendChild(pauseRow);
    }

    function renderFlags(t) {
      if (!t) {
        $("d15-plan-approved").textContent = "Plan approved: —";
        $("d15-validation-passed").textContent = "Validation passed: —";
        $("d15-paused").textContent = "Paused: —";
        return;
      }
      $("d15-plan-approved").textContent = "Plan approved: " + yesNo(t.plan_approved);
      $("d15-validation-passed").textContent =
        "Validation passed: " + yesNo(t.validation_passed);
      $("d15-paused").textContent = "Paused: " + yesNo(t.paused);
    }

    function renderLog(history) {
      var box = $("d15-log");
      box.innerHTML = "";
      if (!history || !history.length) {
        box.appendChild(element("div", "d10-msg-empty", "(история переходов пуста)"));
        return;
      }
      history.forEach(function (record) {
        var row = element("div", "d15-log-row " + record.status);
        row.appendChild(element("span", "d15-log-time", timeOf(record.timestamp)));
        row.appendChild(element(
          "span", "d15-log-path",
          record.from_state + " → " + record.to_state
        ));
        row.appendChild(element(
          "span", "d15-log-status", record.status.toUpperCase()
        ));
        if (record.from_state === "execution" && record.to_state === "planning") {
          row.appendChild(element("span", "d15-rollback-badge", "ROLLBACK"));
        }
        var detail = record.reason || "";
        if (record.intent) {
          detail = "intent: " + record.intent +
            (record.trigger ? " [" + record.trigger + "]" : "") +
            (record.user_message ? " · «" + record.user_message + "»" : "") +
            (detail ? " — " + detail : "");
        }
        if (detail) {
          row.appendChild(element("span", "d15-log-reason", detail));
        }
        box.appendChild(row);
      });
      box.scrollTop = box.scrollHeight;
    }

    function showDecision(allowed, fromState, toState, reason, action) {
      var banner = $("d15-decision");
      banner.classList.remove("hidden");
      $("d15-decision-status").className =
        "d15-decision-badge " + (allowed ? "allowed" : "blocked");
      $("d15-decision-status").textContent =
        allowed ? "TRANSITION ALLOWED" : "TRANSITION BLOCKED";
      if (!allowed && action && action !== "transition") {
        $("d15-decision-status").textContent = "ACTION BLOCKED";
      }
      $("d15-decision-text").textContent = fromState + " → " + toState;
      $("d15-decision-reason").textContent = reason || "";
    }

    function clearDecision() {
      $("d15-decision").classList.add("hidden");
      $("d15-decision-reason").textContent = "";
    }

    function renderAll(t, allowed, options, history) {
      state.task = t || null;
      state.hasTask = !!t;
      renderLifecycle(t);
      renderFlags(t);
      if (!t) {
        $("d15-current-state").textContent = "—";
        $("d15-current-step").textContent = "—";
        $("d15-expected-action").textContent = "—";
        $("d15-raw").textContent = "Задача не создана.";
        renderAllowed(null, [], []);
        renderLog([]);
      } else {
        $("d15-current-state").textContent = STAGE_LABEL[t.state] || t.state;
        $("d15-current-step").textContent = t.plan && t.plan.length
          ? t.current_step + " / " + t.plan.length
          : String(t.current_step);
        $("d15-expected-action").textContent = t.expected_action || "—";
        $("d15-raw").textContent = JSON.stringify(t, null, 2);
        renderAllowed(t, allowed || [], options || []);
        renderLog(history || t.transitions || []);
      }
      renderChat(t ? t.chat_messages : []);
      updateButtons();
    }

    function renderStateResponse(data) {
      renderAll(
        data.state || null,
        data.allowed_transitions || [],
        data.transition_options || [],
        data.transition_history || (data.state ? data.state.transitions : [])
      );
    }

    function lastPath(history, fallbackFrom) {
      if (history && history.length) {
        var rec = history[history.length - 1];
        return { from: rec.from_state, to: rec.to_state };
      }
      return { from: fallbackFrom || "?", to: "?" };
    }

    function renderTransitionResponse(data) {
      var path = lastPath(data.transition_history, data.state && data.state.state);
      renderAll(
        data.state,
        data.allowed_transitions || [],
        data.transition_options || [],
        data.transition_history || []
      );
      showDecision(data.allowed, path.from, path.to, data.reason, data.action);
    }

    /* Chat journal: human bubbles + visually separate DEBUG blocks. */
    function chatBubble(role, content, kind) {
      var bubble = element(
        "div",
        "chat-bubble chat-" + role + (kind ? " d15-kind-" + kind : "")
      );
      bubble.appendChild(element("span", "chat-role", role === "user" ? "Вы" : "Ассистент"));
      bubble.appendChild(element("div", "chat-content", content));
      return bubble;
    }

    function debugBlock(message) {
      var box = element(
        "div",
        "d15-debug-block " + (message.status || "")
      );
      box.appendChild(element("span", "d15-debug-title", "DEBUG · state machine"));
      box.appendChild(element("div", "d15-debug-text", message.content || ""));
      return box;
    }

    function renderChat(messages) {
      var box = $("d15-messages");
      box.innerHTML = "";
      if (!messages || !messages.length) {
        box.appendChild(element(
          "div", "d10-msg-empty", "(диалог пуст — создайте задачу)"
        ));
        return;
      }
      messages.forEach(function (message) {
        if (message.kind === "debug") {
          box.appendChild(debugBlock(message));
        } else if (message.role === "user") {
          box.appendChild(chatBubble("user", message.content));
        } else {
          box.appendChild(chatBubble("assistant", message.content, message.kind));
        }
      });
      box.scrollTop = box.scrollHeight;
    }

    /* ---------------- API actions ---------------- */
    function guard(work) {
      if (state.inFlight) return;
      setError(""); setNote(""); clearDecision();
      setLoading(true);
      work().catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    function loadState() {
      return http("/api/day15/state").then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось загрузить состояние."));
        renderStateResponse(res.data);
      }).catch(function (err) { setError(err.message); });
    }

    function post(path, body) {
      return http(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined
      }).then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Ошибка запроса (" + res.status + ")"));
        return res.data;
      });
    }

    function onCreate() {
      var goal = $("d15-goal").value.trim();
      if (!goal) { setError("Введите цель задачи."); return; }
      guard(function () {
        // Create task AND run the PLANNING stage in one step: the agent
        // generates the plan and the machine moves to PLAN_APPROVAL.
        return post("/api/day15/start", { goal: goal }).then(function (data) {
          renderChatResponse(data);
          setNote("Задача создана. Агент автоматически подготовил план "
            + "(PLANNING → PLAN_APPROVAL).");
        });
      });
    }

    function renderChatResponse(data) {
      renderAll(
        data.state,
        data.allowed_transitions || [],
        data.transition_options || [],
        data.transition_history || []
      );
      if (data.decision) {
        showDecision(
          data.decision.allowed,
          data.decision.from_state,
          data.decision.to_state,
          data.decision.reason,
          data.trigger
        );
      } else {
        clearDecision();
      }
      if (data.intent_error) {
        setError("Intent recognition failed: " + data.intent_error);
      }
    }

    function onPrepare() {
      guard(function () {
        return post("/api/day15/plan", { message: "" }).then(function (data) {
          renderTransitionResponse(data);
          setNote(data.allowed
            ? "План подготовлен, переход PLANNING → PLAN_APPROVAL выполнен."
            : "Подготовить план можно только в состоянии PLANNING.");
        });
      });
    }

    function onApprove() {
      guard(function () {
        return post("/api/day15/approve").then(function (data) {
          renderTransitionResponse(data);
          setNote(data.allowed
            ? "План утверждён. EXECUTION разблокирован (guard plan_approved)."
            : "Утвердить план можно только в PLAN_APPROVAL.");
        });
      });
    }

    function onTransition(target, note) {
      guard(function () {
        return post("/api/day15/transition", { to_state: target }).then(function (data) {
          renderTransitionResponse(data);
          setNote(note || (data.allowed
            ? "Переход выполнен."
            : "Переход заблокирован backend'ом."));
        });
      });
    }

    function onValidation(passed) {
      guard(function () {
        return post("/api/day15/validation", {
          passed: passed,
          summary: passed ? "All checks passed." : "Some checks failed."
        }).then(function (data) {
          renderTransitionResponse(data);
          setNote(passed
            ? "Validation passed: guard для DONE выполнен."
            : "Validation failed: DONE остаётся заблокированным.");
        });
      });
    }

    function onPause() {
      guard(function () {
        return post("/api/day15/pause").then(function (data) {
          renderTransitionResponse(data);
          setNote("PAUSED. previous_state = " +
            ((data.state && data.state.previous_state) || "—") +
            "; этап, шаг и контекст сохранены.");
        });
      });
    }

    function onResume() {
      guard(function () {
        return post("/api/day15/resume").then(function (data) {
          renderTransitionResponse(data);
          setNote("Resumed в " + ((data.state && data.state.state) || "—") +
            " — контекст сохранён.");
        });
      });
    }

    function onReset() {
      guard(function () {
        return http("/api/day15/task", { method: "DELETE" }).then(function (res) {
          if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось сбросить."));
          $("d15-messages").innerHTML = "";
          clearDecision();
          return loadState().then(function () {
            setNote("Demo сброшено: задача удалена.");
          });
        });
      });
    }

    function onScenario(name) {
      guard(function () {
        return post("/api/day15/scenario/" + name).then(function (data) {
          $("d15-messages").innerHTML = "";
          renderStateResponse(data);
          clearDecision();
          setNote("Сценарий загружен: " + name + ". Используйте контролы и Try skip…");
        });
      });
    }

    function onSend() {
      var text = $("d15-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      var forced = $("d15-propose").value || null;
      guard(function () {
        var body = { message: text };
        if (forced) body.forced_intent = forced;
        return post("/api/day15/chat", body).then(function (data) {
          renderChatResponse(data);
          $("d15-input").value = "";
        });
      });
    }

    /* ---------------- wiring ---------------- */
    $("d15-create").addEventListener("click", onCreate);
    $("d15-prepare").addEventListener("click", onPrepare);
    $("d15-approve").addEventListener("click", onApprove);
    $("d15-start-execution").addEventListener("click", function () {
      onTransition("execution", "Переход PLAN_APPROVAL → EXECUTION выполнен (guard пройден).");
    });
    $("d15-run-validation").addEventListener("click", function () {
      onTransition("validation", "Переход EXECUTION → VALIDATION выполнен.");
    });
    $("d15-back-execution").addEventListener("click", function () {
      onTransition("execution", "Переход VALIDATION → EXECUTION выполнен: исправляем результат.");
    });
    $("d15-revise").addEventListener("click", function () {
      onTransition("planning", "ROLLBACK: EXECUTION → PLANNING, approval старого плана сброшен.");
    });
    $("d15-pass").addEventListener("click", function () { onValidation(true); });
    $("d15-fail").addEventListener("click", function () { onValidation(false); });
    $("d15-complete").addEventListener("click", function () {
      onTransition("done", "Переход VALIDATION → DONE выполнен (guard пройден).");
    });
    $("d15-pause").addEventListener("click", onPause);
    $("d15-resume").addEventListener("click", onResume);
    $("d15-reset").addEventListener("click", onReset);
    $("d15-skip-approval").addEventListener("click", function () {
      onTransition("plan_approval", "Попытка перейти к PLAN_APPROVAL.");
    });
    $("d15-skip-execution").addEventListener("click", function () {
      onTransition("execution", "Попытка пропустить утверждение плана.");
    });
    $("d15-skip-done").addEventListener("click", function () {
      onTransition("done", "Попытка завершить задачу без валидации.");
    });
    $("d15-send").addEventListener("click", onSend);
    $("d15-scenario-happy").addEventListener("click", function () {
      onScenario("happy_path");
    });
    $("d15-scenario-skip-planning").addEventListener("click", function () {
      onScenario("skip_planning");
    });
    $("d15-scenario-skip-validation").addEventListener("click", function () {
      onScenario("skip_validation");
    });
    $("d15-scenario-failed-validation").addEventListener("click", function () {
      onScenario("failed_validation");
    });
    $("d15-scenario-pause-resume").addEventListener("click", function () {
      onScenario("pause_resume");
    });

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day15") {
        loadState();
      }
    });

    loadState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay15);
  } else {
    initDay15();
  }
})();
