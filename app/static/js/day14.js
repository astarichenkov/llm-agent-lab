/* LLM Agent Lab — Day 14: invariants and state constraints.
 * Vanilla JS. Talks to /api/day14/*. The backend owns the invariants (stored
 * separately from the dialog) and runs the deterministic conflict check; this
 * file renders the active invariants, the conversation and the conflict result.
 */
(function () {
  "use strict";

  function initDay14() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day14", "panel-day14",
      "d14-invariants",
      "d14-result", "d14-result-status", "d14-result-text", "d14-conflicts",
      "d14-messages", "d14-input", "d14-send", "d14-clear",
      "d14-loading", "d14-error", "d14-note", "d14-context-block", "d14-raw"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day14: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = { inFlight: false, invariants: [], messages: [] };

    /* ---------------- helpers ---------------- */
    function setError(m) {
      $("d14-error").textContent = m || "";
      $("d14-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d14-note").textContent = m || "";
      $("d14-note").classList.toggle("hidden", !m);
    }
    function setLoading(on) {
      state.inFlight = on;
      $("d14-send").disabled = on;
      $("d14-clear").disabled = on;
      $("d14-loading").classList.toggle("hidden", !on);
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

    /* ---------------- rendering ---------------- */
    function renderInvariants(items) {
      state.invariants = items || [];
      var box = $("d14-invariants");
      box.innerHTML = "";
      if (!state.invariants.length) {
        box.appendChild(element("div", "d10-msg-empty", "(нет активных инвариантов)"));
        return;
      }
      state.invariants.forEach(function (inv) {
        var card = element("div", "d14-invariant");
        var head = element("div", "d14-invariant-head");
        head.appendChild(element("span", "d14-invariant-cat", inv.category_label || inv.category));
        head.appendChild(element("code", "d14-invariant-id", inv.id));
        card.appendChild(head);
        card.appendChild(element("div", "d14-invariant-rule", inv.rule));
        box.appendChild(card);
      });
    }

    function renderMessages(messages) {
      state.messages = messages || [];
      var box = $("d14-messages");
      box.innerHTML = "";
      state.messages.forEach(function (msg) {
        var bubble = element("div", "chat-bubble chat-" + msg.role);
        bubble.appendChild(element(
          "span", "chat-role", msg.role === "user" ? "Вы" : "Ассистент"
        ));
        bubble.appendChild(element("div", "chat-content", msg.content));
        box.appendChild(bubble);
      });
      box.scrollTop = box.scrollHeight;
      $("d14-raw").textContent = state.messages.length
        ? JSON.stringify(state.messages, null, 2)
        : "История пуста.";
    }

    function renderConflicts(conflicts) {
      var box = $("d14-conflicts");
      box.innerHTML = "";
      if (!conflicts || !conflicts.length) return;
      conflicts.forEach(function (c) {
        var item = element("div", "d14-conflict-item");
        item.appendChild(element(
          "div", "d14-conflict-cat", (c.category_label || c.category) + " — " + c.invariant_id
        ));
        item.appendChild(element("div", "d14-conflict-rule", c.rule));
        if (c.matched && c.matched.length) {
          item.appendChild(element(
            "div", "d14-conflict-match", "Совпадение: " + c.matched.join(", ")
          ));
        }
        box.appendChild(item);
      });
    }

    function renderResult(data) {
      var status = $("d14-result-status");
      status.className = "d14-badge " + (data.allowed ? "ok" : "conflict");
      status.textContent = data.allowed ? "РАЗРЕШЕНО" : "КОНФЛИКТ";
      if (data.allowed) {
        $("d14-result-text").textContent =
          "Запрос не нарушает активные инварианты — он обработан нормально.";
        renderConflicts([]);
      } else {
        $("d14-result-text").textContent =
          "Запрос нарушает " + data.conflicts.length + " инвариант(а/ов). " +
          "Ассистент отказал и указал конфликтующие инварианты.";
        renderConflicts(data.conflicts);
      }
      $("d14-context-block").textContent = data.context_block || "—";
    }

    function resetResult() {
      var status = $("d14-result-status");
      status.className = "d14-badge";
      status.textContent = "—";
      $("d14-result-text").textContent =
        "Отправьте запрос, чтобы проверить его на конфликт с инвариантами.";
      renderConflicts([]);
    }

    /* ---------------- API actions ---------------- */
    function loadState() {
      return http("/api/day14/state").then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось загрузить инварианты."));
        renderInvariants(res.data.invariants);
        renderMessages(res.data.messages);
        resetResult();
      }).catch(function (err) { setError(err.message); });
    }

    function onSend() {
      if (state.inFlight) return;
      var text = $("d14-input").value.trim();
      if (!text) { setError("Введите запрос."); return; }
      setError(""); setNote("");
      setLoading(true);
      http("/api/day14/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      }).then(function (res) {
        if (!res.ok) throw new Error(detailMessage(res.data, "Ошибка запроса (" + res.status + ")"));
        var d = res.data;
        renderResult(d);
        renderInvariants(d.invariants);
        // Reload the conversation so user + assistant messages are shown.
        return http("/api/day14/state").then(function (stateRes) {
          if (stateRes.ok) renderMessages(stateRes.data.messages);
        }).then(function () {
          if (d.allowed) {
            setNote("Запрос разрешён: конфликтов с инвариантами нет.");
          } else {
            setNote("Запрос отклонён: нарушенные инварианты показаны выше.");
          }
          $("d14-input").value = "";
        });
      }).catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    function onClear() {
      if (state.inFlight) return;
      setError(""); setNote("");
      setLoading(true);
      http("/api/day14/history", { method: "DELETE" })
        .then(function (res) {
          if (!res.ok) throw new Error(detailMessage(res.data, "Не удалось очистить историю."));
          return loadState();
        }).then(function () { setNote("История диалога очищена. Инварианты сохранены."); })
        .catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    /* ---------------- wiring ---------------- */
    $("d14-send").addEventListener("click", onSend);
    $("d14-clear").addEventListener("click", onClear);

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day14") {
        loadState();
      }
    });

    loadState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay14);
  } else {
    initDay14();
  }
})();
