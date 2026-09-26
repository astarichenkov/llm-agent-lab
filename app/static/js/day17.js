/* LLM Agent Lab — Day 17: VictoriaLogs MCP tool.
 *
 * Vanilla JS. Talks to POST /api/week4/day17/chat. The backend owns the whole
 * MCP flow: it discovers the real tools, the LLM selects `search_logs`, the
 * backend performs MCP `tools/call` over stdio and feeds the result back to
 * the model. This file only renders the returned answer, the REAL tool call
 * and the REAL execution trace — it never hardcodes any tool or log data.
 */
(function () {
  "use strict";

  function initDay17() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day17",
      "d17-message", "d17-send", "d17-loading", "d17-error",
      "d17-answer-section", "d17-answer",
      "d17-toolcall-section", "d17-toolcall-server", "d17-toolcall-tool",
      "d17-toolcount", "d17-toolcall-args", "d17-samples",
      "d17-trace-section", "d17-trace",
      "d17-window-label", "d17-start", "d17-end",
      "d17-back", "d17-now", "d17-forward",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day17: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var inFlight = false;

    /* ---------------- Time interval ---------------- */
    // datetime-local expects a LOCAL "YYYY-MM-DDTHH:mm" value.
    function toInputValue(date) {
      function pad(n) { return (n < 10 ? "0" : "") + n; }
      return (
        date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate()) +
        "T" + pad(date.getHours()) + ":" + pad(date.getMinutes())
      );
    }
    function fromInputValue(value) {
      var parsed = new Date(value);
      return isNaN(parsed.getTime()) ? null : parsed;
    }
    function updateWindowLabel() {
      var start = fromInputValue($("d17-start").value);
      var end = fromInputValue($("d17-end").value);
      if (!start || !end) { $("d17-window-label").textContent = "—"; return; }
      var minutes = Math.round((end - start) / 60000);
      $("d17-window-label").textContent =
        toInputValue(start).replace("T", " ") + " → " +
        toInputValue(end).replace("T", " ") + "  (" + minutes + " мин)";
    }
    function setWindow(minutes) {
      var now = new Date();
      now.setSeconds(0, 0);
      $("d17-start").value = toInputValue(new Date(now.getTime() - minutes * 60000));
      $("d17-end").value = toInputValue(now);
      updateWindowLabel();
    }
    function shiftWindow(hours) {
      var start = fromInputValue($("d17-start").value);
      var end = fromInputValue($("d17-end").value);
      if (!start || !end) { setWindow(30); return; }
      var delta = hours * 3600000;
      $("d17-start").value = toInputValue(new Date(start.getTime() + delta));
      $("d17-end").value = toInputValue(new Date(end.getTime() + delta));
      updateWindowLabel();
    }
    function setError(message) {
      var el = $("d17-error");
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }
    function setLoading(on) {
      inFlight = on;
      $("d17-loading").classList.toggle("hidden", !on);
      $("d17-send").disabled = on;
    }
    function clear(el) {
      el.innerHTML = "";
      el.textContent = "";
    }
    function show(el, on) {
      el.classList.toggle("hidden", !on);
    }

    function renderTrace(steps) {
      var list = $("d17-trace");
      clear(list);
      (steps || []).forEach(function (step) {
        var li = document.createElement("li");
        li.className = "d16-trace-step d16-trace-" + (step.status || "ok");

        var badge = document.createElement("span");
        badge.className = "d16-trace-badge";
        badge.textContent = step.status === "error" ? "ERROR" : "OK";
        li.appendChild(badge);

        var message = document.createElement("span");
        message.className = "d16-trace-message";
        message.textContent = step.message;
        li.appendChild(message);

        list.appendChild(li);
      });
    }

    function renderToolCalls(toolCalls) {
      toolCalls = toolCalls || [];
      if (toolCalls.length === 0) {
        show($("d17-toolcall-section"), false);
        return;
      }
      show($("d17-toolcall-section"), true);
      var first = toolCalls[0];
      $("d17-toolcall-server").textContent = first.server || "victorialogs";
      $("d17-toolcall-tool").textContent = first.tool || "search_logs";
      $("d17-toolcall-args").textContent = JSON.stringify(
        first.arguments || {}, null, 2
      );
      var summary = first.result_summary || {};
      $("d17-toolcount").textContent =
        summary.count != null ? summary.count : 0;
      var samples = $("d17-samples");
      clear(samples);
      var note = document.createElement("p");
      note.className = "field-hint";
      note.textContent = first.result_summary && first.result_summary.error
        ? "VictoriaLogs error: " + first.result_summary.error
        : "Полные строки логов передаются в LLM (и санитизируются); в браузер "
          + "выводится только счётчик, чтобы не публиковать stage-данные.";
      samples.appendChild(note);
    }

    function render(data) {
      data = data || {};
      show($("d17-answer-section"), true);
      $("d17-answer").textContent = data.answer || "";
      renderToolCalls(data.tool_calls);
      show($("d17-trace-section"), true);
      renderTrace(data.trace);
      if (data.error) {
        setError(data.error);
      } else {
        setError("");
      }
    }

    function send() {
      if (inFlight) return;
      var message = ($("d17-message").value || "").trim();
      if (!message) {
        setError("Введите сообщение.");
        return;
      }
      var start = fromInputValue($("d17-start").value);
      var end = fromInputValue($("d17-end").value);
      if (!start || !end) {
        setError("Выберите корректный интервал времени.");
        return;
      }
      if (end <= start) {
        setError("Конец интервала должен быть позже начала.");
        return;
      }
      if ((end - start) / 60000 > 1440) {
        setError("Интервал не должен превышать 24 часа.");
        return;
      }
      setLoading(true);
      setError("");
      show($("d17-answer-section"), false);
      show($("d17-toolcall-section"), false);
      show($("d17-trace-section"), false);

      fetch("/api/week4/day17/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({
          message: message,
          start: start.toISOString(),
          end: end.toISOString(),
        }),
      })
        .then(function (response) {
          return response.json().then(function (data) {
            return { ok: response.ok, status: response.status, data: data };
          });
        })
        .then(function (res) {
          if (!res.ok) {
            throw new Error(
              (res.data && res.data.detail) ||
              ("Request failed (" + res.status + ")")
            );
          }
          render(res.data);
        })
        .catch(function (err) {
          show($("d17-trace-section"), false);
          setError(err.message || "Day 17 request failed.");
        })
        .finally(function () {
          setLoading(false);
        });
    }

    $("d17-send").addEventListener("click", send);

    var presets = document.querySelectorAll("[data-d17-preset]");
    Array.prototype.forEach.call(presets, function (button) {
      button.addEventListener("click", function () {
        setWindow(parseInt(button.getAttribute("data-d17-preset"), 10) || 30);
        setError("");
      });
    });

    $("d17-back").addEventListener("click", function () { shiftWindow(-1); });
    $("d17-forward").addEventListener("click", function () { shiftWindow(1); });
    $("d17-now").addEventListener("click", function () {
      var start = fromInputValue($("d17-start").value);
      var end = fromInputValue($("d17-end").value);
      var minutes = start && end ? Math.round((end - start) / 60000) : 30;
      setWindow(minutes);
    });
    $("d17-start").addEventListener("change", updateWindowLabel);
    $("d17-end").addEventListener("change", updateWindowLabel);

    var examples = document.querySelectorAll("[data-d17-example]");
    Array.prototype.forEach.call(examples, function (button) {
      button.addEventListener("click", function () {
        $("d17-message").value = button.getAttribute("data-d17-example") || "";
        setError("");
      });
    });

    setWindow(30);
    setError("");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay17);
  } else {
    initDay17();
  }
})();
