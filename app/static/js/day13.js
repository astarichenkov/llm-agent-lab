/* LLM Agent Lab — Day 13: Task State Machine.
 * Vanilla JS. Talks to /api/day13/*. The backend owns the structured Task
 * State and enforces the allowed FSM transitions; this file only visualises
 * the state, the FSM progress and provides Pause / Resume.
 */
(function () {
  "use strict";

  function initDay13() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day13", "panel-day13",
      "d13-goal", "d13-create", "d13-reset",
      "d13-loading", "d13-error", "d13-note",
      "d13-fsm",
      "d13-stage-planning", "d13-stage-execution",
      "d13-stage-pause",
      "d13-stage-validation", "d13-stage-done",
      "d13-stage", "d13-step", "d13-expected", "d13-status",
      "d13-pause", "d13-resume", "d13-allowed",
      "d13-plan-list", "d13-completed-list", "d13-raw",
      "d13-messages", "d13-input", "d13-send", "d13-context-block"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day13: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var STAGE_ORDER = ["planning", "execution", "validation", "done"];
    var state = { inFlight: false, hasTask: false, task: null };

    /* ---------------- helpers ---------------- */
    function setError(m) {
      $("d13-error").textContent = m || "";
      $("d13-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d13-note").textContent = m || "";
      $("d13-note").classList.toggle("hidden", !m);
    }
    function setLoading(on) {
      state.inFlight = on;
      $("d13-create").disabled = on;
      $("d13-send").disabled = on || !canSend();
      $("d13-pause").disabled = on || !canPause();
      $("d13-resume").disabled = on || !canResume();
      $("d13-loading").classList.toggle("hidden", !on);
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

    /* ---------------- button availability ---------------- */
    function canPause() {
      return state.hasTask && !state.task.paused && state.task.stage !== "done";
    }
    function canResume() {
      return state.hasTask && state.task.paused === true;
    }
    function canSend() {
      return state.hasTask && !state.task.paused && state.task.stage !== "done";
    }

    /* ---------------- rendering ---------------- */
    function renderFsm(stage, paused) {
      var currentIndex = STAGE_ORDER.indexOf(stage);
      STAGE_ORDER.forEach(function (name, index) {
        var node = $("d13-stage-" + name);
        if (!node) return;
        node.classList.remove("active", "passed");
        // While paused, the workflow stage is preserved but NOT highlighted:
        // the temporary Pause node is the only active state.
        if (!paused && state.hasTask) {
          if (index < currentIndex) node.classList.add("passed");
          if (index === currentIndex) node.classList.add("active");
        }
      });
      var pauseNode = $("d13-stage-pause");
      if (pauseNode) {
        pauseNode.classList.remove("active", "passed");
        if (state.hasTask && paused) pauseNode.classList.add("active");
      }
    }

    function renderList(container, items, numbered) {
      container.innerHTML = "";
      if (!items || !items.length) {
        container.appendChild(element("div", "d10-msg-empty", "(пусто)"));
        return;
      }
      items.forEach(function (item, index) {
        var row = element("div", "d10-msg-row");
        var marker = element("span", "d10-msg-role", numbered ? (index + 1) + "." : "•");
        row.appendChild(marker);
        row.appendChild(element("span", "d10-msg-text", item));
        container.appendChild(row);
      });
    }

    function render(data) {
      state.hasTask = !!data.has_task;
      state.task = data.state;
      var allowed = data.allowed_transitions || [];

      if (!state.hasTask) {
        $("d13-stage").textContent = "—";
        $("d13-step").textContent = "—";
        $("d13-expected").textContent = "—";
        $("d13-status").textContent = "NO TASK";
        $("d13-status").className = "d13-status";
        $("d13-raw").textContent = "Задача не создана.";
        $("d13-allowed").textContent = "—";
        $("d13-context-block").textContent = "—";
        renderFsm("", false);
        renderList($("d13-plan-list"), [], true);
        renderList($("d13-completed-list"), [], false);
      } else {
        var task = state.task;
        $("d13-stage").textContent = String(task.stage).toUpperCase();
        $("d13-step").textContent = task.plan && task.plan.length
          ? task.current_step + " / " + task.plan.length
          : String(task.current_step);
        $("d13-expected").textContent = task.expected_action || "—";
        var status;
        if (task.paused) status = "PAUSED";
        else if (task.stage === "done") status = "DONE";
        else status = "ACTIVE";
        $("d13-status").textContent = status;
        $("d13-status").className = "d13-status " +
          (task.paused ? "paused" : (task.stage === "done" ? "done" : "active"));
        $("d13-raw").textContent = JSON.stringify(task, null, 2);
        $("d13-allowed").textContent = allowed.length ? allowed.join(", ") : "нет";
        renderFsm(task.stage, task.paused === true);
        renderList($("d13-plan-list"), task.plan || [], true);
        renderList($("d13-completed-list"), task.completed_steps || [], false);
      }

      $("d13-pause").disabled = state.inFlight || !canPause();
      $("d13-resume").disabled = state.inFlight || !canResume();
      $("d13-send").disabled = state.inFlight || !canSend();
    }

    function addBubble(role, content) {
      var bubble = element("div", "chat-bubble chat-" + role);
      bubble.appendChild(element("span", "chat-role", role === "user" ? "Вы" : "Агент"));
      bubble.appendChild(element("div", "chat-content", content));
      $("d13-messages").appendChild(bubble);
      $("d13-messages").scrollTop = $("d13-messages").scrollHeight;
    }

    function report(event) {
      if (event) setNote("Переход (кодом): " + event);
    }

    /* ---------------- API actions ---------------- */
    function loadState() {
      return http("/api/day13/state").then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось загрузить состояние."));
        render(res.data);
      }).catch(function (err) { setError(err.message); });
    }

    function onCreate() {
      if (state.inFlight) return;
      var goal = $("d13-goal").value.trim();
      if (!goal) { setError("Введите цель задачи."); return; }
      setError(""); setNote("");
      $("d13-messages").innerHTML = "";
      setLoading(true);
      http("/api/day13/task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: goal })
      }).then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось создать задачу."));
        render(res.data);
        setNote("Задача создана. Стадия: planning.");
      }).catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    function onPause() {
      if (state.inFlight || !canPause()) return;
      setError(""); setNote("");
      setLoading(true);
      http("/api/day13/pause", { method: "POST" })
        .then(function (res) {
          if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось поставить на паузу."));
          render(res.data);
          setNote("PAUSED. stage, current_step и expected_action сохранены.");
        }).catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    function onResume() {
      if (state.inFlight || !canResume()) return;
      setError(""); setNote("");
      setLoading(true);
      http("/api/day13/resume", { method: "POST" })
        .then(function (res) {
          if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось продолжить задачу."));
          render(res.data);
          setNote("Resumed — продолжаем с того же шага, без повторного объяснения задачи.");
        }).catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    function onReset() {
      if (state.inFlight) return;
      setError(""); setNote("");
      setLoading(true);
      http("/api/day13/task", { method: "DELETE" })
        .then(function (res) {
          if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось сбросить состояние."));
          $("d13-messages").innerHTML = "";
          $("d13-context-block").textContent = "—";
          return loadState();
        }).then(function () { setNote("Состояние сброшено."); })
        .catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    function onSend() {
      if (state.inFlight || !canSend()) return;
      var text = $("d13-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      setError(""); setNote("");
      addBubble("user", text);
      setLoading(true);
      http("/api/day13/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      }).then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Ошибка запроса (" + res.status + ")"));
        var d = res.data;
        addBubble("assistant", d.answer);
        $("d13-context-block").textContent = d.context_block || "—";
        render({
          has_task: true,
          state: d.state,
          allowed_transitions: d.allowed_transitions || []
        });
        report(d.stage_event);
        $("d13-input").value = "";
      }).catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- wiring ---------------- */
    $("d13-create").addEventListener("click", onCreate);
    $("d13-pause").addEventListener("click", onPause);
    $("d13-resume").addEventListener("click", onResume);
    $("d13-reset").addEventListener("click", onReset);
    $("d13-send").addEventListener("click", onSend);

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day13") {
        loadState();
      }
    });

    loadState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay13);
  } else {
    initDay13();
  }
})();
