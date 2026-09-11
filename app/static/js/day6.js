/* LLM Agent Lab — Day 6: the first agent (stateless).
 * Vanilla JS. One click = one POST /api/day6/agent/chat.
 *
 * The backend creates a FRESH transient Agent for every request, so the
 * previous answer is never part of the next LLM payload. The UI intentionally
 * shows a single "last answer" block instead of a growing timeline.
 */
(function () {
  "use strict";

  function initDay6() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day6", "panel-day6", "d6-agent-name", "d6-agent-provider",
      "d6-agent-model", "d6-agent-max-tokens", "d6-agent-thinking",
      "d6-input", "d6-send", "d6-loading", "d6-error",
      "d6-answer", "d6-meta"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day6: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var inputEl = $("d6-input");
    var sendBtn = $("d6-send");
    var loadingEl = $("d6-loading");
    var errorEl = $("d6-error");
    var answerEl = $("d6-answer");
    var metaEl = $("d6-meta");
    var inFlight = false;

    function setError(message) {
      errorEl.textContent = message || "";
      errorEl.classList.toggle("hidden", !message);
    }

    function setLoading(on) {
      inFlight = on;
      sendBtn.disabled = on;
      loadingEl.classList.toggle("hidden", !on);
    }

    function reasoningLabel(thinking) {
      return thinking ? "ON" : "OFF";
    }

    function applyConfig(agent) {
      if (!agent) return;
      if (agent.name) $("d6-agent-name").textContent = agent.name;
      $("d6-agent-provider").textContent = agent.provider || "—";
      $("d6-agent-model").textContent = agent.model || "—";
      $("d6-agent-max-tokens").textContent =
        (agent.max_tokens == null ? "—" : agent.max_tokens + " tokens");
      $("d6-agent-thinking").textContent = reasoningLabel(agent.thinking);
    }

    function showAnswer(data) {
      $("d6-agent-provider").textContent = data.provider || "—";
      $("d6-agent-model").textContent = data.model || "—";
      metaEl.textContent =
        "provider: " + (data.provider || "—") +
        " · model: " + (data.model || "—") +
        " · finish_reason: " + (data.finish_reason || "—");
      answerEl.textContent = data.answer || "";
    }

    // Load the default agent metadata from the backend on init, so the
    // Agent/Provider/Model/limit/reasoning cards are populated BEFORE the
    // first request. Values come from the same backend config resolution as
    // the chat call — nothing is hardcoded here.
    function loadMetadata() {
      fetch("/api/day6/agent")
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
        .then(function (res) {
          if (!res.ok) return;
          applyConfig(res.data);
        })
        .catch(function () { /* keep the — placeholders if metadata fails */ });
    }

    function sendMessage() {
      if (inFlight) return;
      var text = inputEl.value.trim();
      if (!text) { setError("Введите запрос."); return; }

      setError("");
      setLoading(true);
      // Stateless: every submit replaces the previous answer.
      answerEl.textContent = "";
      metaEl.textContent = "Запрос выполняется…";

      fetch("/api/day6/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; }); })
        .then(function (res) {
          if (!res.ok) {
            throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
          }
          showAnswer(res.data);
        })
        .catch(function (err) {
          metaEl.textContent = "—";
          setError(err.message || "Что-то пошло не так. Попробуйте ещё раз.");
        })
        .finally(function () { setLoading(false); });
    }

    sendBtn.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendMessage();
      }
    });

    loadMetadata();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay6);
  } else {
    initDay6();
  }
})();
