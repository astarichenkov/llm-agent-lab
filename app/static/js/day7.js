/* LLM Agent Lab — Day 7: persistent agent context.
 * Vanilla JS. Loads the saved dialog and talks to POST /api/chat, which is
 * backed by the persistent "default" Agent (SQLite context).
 */
(function () {
  "use strict";

  function initDay7() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day7", "panel-day7", "d7-agent-name", "d7-agent-provider",
      "d7-agent-model", "d7-agent-max-tokens", "d7-agent-thinking",
      "d7-context-count", "d7-messages", "d7-input",
      "d7-send", "d7-clear", "d7-loading", "d7-error"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day7: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var messagesEl = $("d7-messages");
    var inputEl = $("d7-input");
    var sendBtn = $("d7-send");
    var clearBtn = $("d7-clear");
    var loadingEl = $("d7-loading");
    var errorEl = $("d7-error");
    var inFlight = false;
    var loadedOnce = false;

    /* ---------------- helpers ---------------- */
    function setError(message) {
      errorEl.textContent = message || "";
      errorEl.classList.toggle("hidden", !message);
    }

    function setLoading(on) {
      inFlight = on;
      sendBtn.disabled = on;
      clearBtn.disabled = on;
      loadingEl.classList.toggle("hidden", !on);
    }

    function updateInfo(agent, count) {
      if (agent) {
        $("d7-agent-name").textContent = agent.name || agent.agent_id || "—";
        $("d7-agent-provider").textContent = agent.provider || "—";
        $("d7-agent-model").textContent = agent.model || "—";
        $("d7-agent-max-tokens").textContent =
          (agent.max_tokens == null ? "—" : agent.max_tokens + " tokens");
        $("d7-agent-thinking").textContent = agent.thinking ? "ON" : "OFF";
      }
      $("d7-context-count").textContent =
        count + " " + pluralMessages(count);
    }

    function pluralMessages(n) {
      var mod10 = n % 10;
      var mod100 = n % 100;
      if (mod10 === 1 && mod100 !== 11) return "сообщение";
      if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return "сообщения";
      return "сообщений";
    }

    function addBubble(role, content) {
      var bubble = document.createElement("div");
      bubble.className = "chat-bubble chat-" + role;
      var label = document.createElement("span");
      label.className = "chat-role";
      label.textContent = role === "user" ? "Вы" : "Агент";
      var body = document.createElement("div");
      body.className = "chat-content";
      body.textContent = content;
      bubble.appendChild(label);
      bubble.appendChild(body);
      messagesEl.appendChild(bubble);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderEmpty() {
      messagesEl.innerHTML = "";
      var hint = document.createElement("p");
      hint.className = "field-hint";
      hint.textContent = "История пуста. Начните диалог — он сохранится автоматически.";
      messagesEl.appendChild(hint);
    }

    function renderHistory(body) {
      messagesEl.innerHTML = "";
      var messages = (body && body.messages) || [];
      updateInfo(body && body.agent, messages.length);
      if (!messages.length) {
        renderEmpty();
        return;
      }
      messages.forEach(function (m) { addBubble(m.role, m.content); });
    }

    /* ---------------- HTTP ---------------- */
    function loadHistory() {
      if (inFlight) return Promise.resolve();
      return fetch("/api/chat/history")
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
        .then(function (res) {
          if (!res.ok) throw new Error(res.data.detail || "Не удалось загрузить историю.");
          renderHistory(res.data);
          loadedOnce = true;
        })
        .catch(function (err) { setError(err.message || "Ошибка загрузки истории."); });
    }

    function sendMessage() {
      if (inFlight) return;
      var text = inputEl.value.trim();
      if (!text) { setError("Введите сообщение."); return; }

      setError("");
      setLoading(true);
      if (!loadedOnce || messagesEl.querySelector(".chat-bubble") === null) {
        messagesEl.innerHTML = "";
      }
      addBubble("user", text);

      fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
        .then(function (res) {
          if (!res.ok) throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
          addBubble("assistant", res.data.answer);
          inputEl.value = "";
          loadedOnce = true;
          return loadHistory();
        })
        .catch(function (err) { setError(err.message || "Что-то пошло не так."); })
        .finally(function () { setLoading(false); });
    }

    function clearHistory() {
      if (inFlight) return;
      setError("");
      setLoading(true);
      fetch("/api/chat/history", { method: "DELETE" })
        .then(function (r) { if (!r.ok) throw new Error("Не удалось очистить контекст."); return loadHistory(); })
        .catch(function (err) { setError(err.message || "Ошибка очистки."); })
        .finally(function () { setLoading(false); });
    }

    /* ---------------- wiring ---------------- */
    sendBtn.addEventListener("click", sendMessage);
    clearBtn.addEventListener("click", clearHistory);
    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day7") loadHistory();
    });
    inputEl.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendMessage();
      }
    });

    loadHistory();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay7);
  } else {
    initDay7();
  }
})();
