/* LLM Agent Lab — comparison frontend (управление ответом модели).
 * Vanilla JavaScript (no framework). Talks to POST /api/compare.
 *
 * One click = exactly ONE /api/compare request; the backend performs
 * exactly TWO DeepSeek provider calls (unrestricted + controlled).
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Global main-tab controller (Days 2–7).
  // Centralized in app.js so every day panel is switched by ONE piece of
  // logic; no day-specific script can make another day unreachable. It also
  // emits a ``llmtabchange`` event so a tab can lazily load its data.
  // ------------------------------------------------------------------
  var MAIN_TABS = ["day2", "day3", "day4", "day5", "day6", "day7"];

  function switchMainTab(name) {
    if (MAIN_TABS.indexOf(name) === -1) return;
    MAIN_TABS.forEach(function (tab) {
      var panel = document.getElementById("panel-" + tab);
      if (panel) panel.style.display = (tab === name) ? "block" : "none";
      var btn = document.getElementById("tab-" + tab);
      if (btn) {
        var on = tab === name;
        btn.classList.toggle("active", on);
        btn.setAttribute("aria-selected", on ? "true" : "false");
      }
    });
    try {
      document.dispatchEvent(
        new CustomEvent("llmtabchange", { detail: { tab: name } })
      );
    } catch (err) {
      /* very old browsers: tab switching still works without the event */
    }
  }

  function initTabs() {
    MAIN_TABS.forEach(function (tab) {
      var btn = document.getElementById("tab-" + tab);
      if (!btn) return;
      btn.addEventListener("click", function () { switchMainTab(tab); });
    });
    switchMainTab("day2");
  }

  function initApp() {
    var requiredIds = [
      "compare-form", "message", "json-structure", "response-format",
      "max-tokens", "stop-sequence", "reset-json", "compare-button",
      "loading", "error", "api-preview", "structure-instruction",
      "results", "answer-unrestricted", "finish-unrestricted",
      "answer-controlled", "finish-controlled", "applied-settings",
      "applied-heading", "structure-used", "controlled-error",
      "summary-section", "summary"
    ];
    var missing = requiredIds.filter(function (id) {
      return !document.getElementById(id);
    });
    if (missing.length > 0) {
      console.error(
        "LLM Agent Lab: missing DOM elements, frontend disabled:",
        missing.join(", ")
      );
      return;
    }

    var form = document.getElementById("compare-form");
    var messageEl = document.getElementById("message");
    var structureEl = document.getElementById("json-structure");
    var maxTokensEl = document.getElementById("max-tokens");
    var stopEl = document.getElementById("stop-sequence");
    var resetJsonBtn = document.getElementById("reset-json");
    var button = document.getElementById("compare-button");
    var loading = document.getElementById("loading");
    var errorBox = document.getElementById("error");
    var previewEl = document.getElementById("api-preview");
    var structurePreviewEl = document.getElementById("structure-instruction");

    var resultsSection = document.getElementById("results");
    var answerUnrestricted = document.getElementById("answer-unrestricted");
    var finishUnrestricted = document.getElementById("finish-unrestricted");
    var answerControlled = document.getElementById("answer-controlled");
    var finishControlled = document.getElementById("finish-controlled");
    var appliedHeading = document.getElementById("applied-heading");
    var appliedSettings = document.getElementById("applied-settings");
    var structureUsed = document.getElementById("structure-used");
    var controlledError = document.getElementById("controlled-error");
    var summarySection = document.getElementById("summary-section");
    var summaryList = document.getElementById("summary");

    var DEFAULT_STRUCTURE = '{\n  "products": [\n    {\n      "name": "Название продукта",\n      "count": "Количество",\n      "unit": "Единица измерения"\n    }\n  ]\n}';
    var inFlight = false;

    /* ---------------- Tooltips: hover, focus, tap ---------------- */
    document.querySelectorAll(".info-btn").forEach(function (btn) {
      var tip = document.getElementById(btn.getAttribute("aria-controls"));
      if (!tip) return;

      function show() {
        tip.classList.add("open");
        btn.setAttribute("aria-expanded", "true");
      }
      function hide() {
        tip.classList.remove("open");
        btn.setAttribute("aria-expanded", "false");
      }

      btn.addEventListener("mouseenter", show);
      btn.addEventListener("mouseleave", hide);
      btn.addEventListener("focus", show);
      btn.addEventListener("blur", hide);
      btn.addEventListener("click", function (event) {
        event.preventDefault();
        if (tip.classList.contains("open")) hide(); else show();
      });
      btn.addEventListener("keydown", function (event) {
        if (event.key === "Escape") hide();
      });
    });

    /* ---------------- Reset structure JSON ---------------- */
    resetJsonBtn.addEventListener("click", function () {
      structureEl.value = DEFAULT_STRUCTURE;
      structureEl.classList.remove("invalid");
      updatePreview();
    });

    /* ---------------- Live preview ---------------- */
    function parseStructure() {
      // Throws Error with readable message when JSON is malformed / not object.
      var raw = structureEl.value.trim();
      if (!raw) throw new Error("структура пуста");
      var parsed = JSON.parse(raw); // may throw SyntaxError
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("корень должен быть JSON-объектом, а не массивом/значением");
      }
      return parsed;
    }

    function updatePreview() {
      var stop = stopEl.value.trim();
      var params = {
        response_format: { type: "json_object" },
        max_tokens: (function () {
          var mt = parseInt(maxTokensEl.value, 10);
          return Number.isInteger(mt) ? mt : null;
        })()
      };
      // stop is OPTIONAL: it is only shown (and only sent) when non-empty.
      if (stop) {
        params.stop = [stop];
      }
      previewEl.textContent = JSON.stringify(params, null, 2);

      try {
        var structure = parseStructure();
        structureEl.classList.remove("invalid");
        structurePreviewEl.textContent = JSON.stringify(structure, null, 2);
        structurePreviewEl.classList.remove("preview-invalid");
      } catch (err) {
        structureEl.classList.add("invalid");
        structurePreviewEl.textContent = "Некорректный JSON: " + err.message;
        structurePreviewEl.classList.add("preview-invalid");
      }
    }

    messageEl.addEventListener("input", updatePreview);
    structureEl.addEventListener("input", updatePreview);
    maxTokensEl.addEventListener("input", updatePreview);
    stopEl.addEventListener("input", updatePreview);
    updatePreview();

    /* ---------------- Submission ---------------- */
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submitComparison();
    });

    function submitComparison() {
      if (inFlight) return;

      var message = messageEl.value.trim();
      if (!message) {
        showError("Введите запрос — он будет отправлен дважды.");
        return;
      }
      var structure;
      try {
        structure = parseStructure();
      } catch (err) {
        structureEl.classList.add("invalid");
        showError("Некорректная структура JSON: " + err.message);
        structureEl.focus();
        return; // do not call backend / do not spend API balance
      }
      var maxTokens = parseInt(maxTokensEl.value, 10);
      if (!Number.isInteger(maxTokens) || maxTokens < 16 || maxTokens > 2000) {
        showError("Максимальная длина ответа: целое число от 16 до 2000 токенов.");
        return;
      }

      var payload = {
        message: message,
        json_structure: structure,
        max_tokens: maxTokens
      };
      var stopRaw = stopEl.value.trim();
      if (stopRaw && stopRaw.length < 4) {
        showError(
          "Стоп-последовательность должна быть различимым маркером (минимум 4 символа, включая букву или цифру). Короткий маркер может оборвать генерацию внутри JSON."
        );
        stopEl.focus();
        return;
      }
      if (stopRaw) {
        payload.stop_sequence = stopRaw;
      }

      setLoading(true);
      hideError();
      hideResults();

      fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
        .then(function (response) {
          return response.json().then(function (data) {
            return { ok: response.ok, status: response.status, data: data };
          });
        })
        .then(function (res) {
          if (!res.ok) {
            throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
          }
          render(res.data);
        })
        .catch(function (err) {
          showError(err.message || "Что-то пошло не так. Попробуйте ещё раз.");
        })
        .finally(function () {
          setLoading(false);
        });
    }

    /* ---------------- Rendering ---------------- */
    function render(data) {
      answerUnrestricted.textContent = data.unrestricted.answer || "";
      finishUnrestricted.textContent = data.unrestricted.finish_reason || "—";

      var requested = data.settings || {};

      if (data.controlled) {
        var controlled = data.controlled;
        answerControlled.textContent = prettyJson(controlled.answer || "");
        finishControlled.textContent = controlled.finish_reason || "—";
        appliedHeading.textContent = "Применённые API-параметры";
        appliedSettings.textContent = JSON.stringify(
          {
            response_format: requested.response_format,
            max_tokens: requested.max_tokens,
            stop: requested.stop
          },
          null,
          2
        );
        structureUsed.textContent = JSON.stringify(requested.json_structure, null, 2);
        controlledError.classList.add("hidden");
        controlledError.textContent = "";
      } else {
        // Controlled provider call failed: requested controls still shown.
        answerControlled.textContent = "";
        // If the real finish_reason is known (e.g. max_tokens output limit
        // gave finish_reason "length"), show it; otherwise mark unavailable.
        finishControlled.textContent = data.controlled_finish_reason ||
          "недоступна — запрос завершился ошибкой";
        appliedHeading.textContent = "Запрошенные API-параметры";
        appliedSettings.textContent = JSON.stringify(
          {
            response_format: requested.response_format,
            max_tokens: requested.max_tokens,
            stop: requested.stop
          },
          null,
          2
        );
        structureUsed.textContent = JSON.stringify(requested.json_structure, null, 2);
        var prefix = data.controlled_finish_reason
          ? "Ответ не поместился в max_tokens: "
          : "Контролируемый запрос не выполнен: ";
        controlledError.textContent = prefix +
          (data.controlled_error || "неизвестная ошибка.");
        controlledError.classList.remove("hidden");
      }

      renderSummary(data);
      resultsSection.classList.remove("hidden");
      summarySection.classList.remove("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function prettyJson(raw) {
      if (!raw) return "";
      try {
        return JSON.stringify(JSON.parse(raw), null, 2);
      } catch (err) {
        return raw; // provider did not return valid JSON — show raw content
      }
    }

    function renderSummary(data) {
      var controlled = data.controlled;
      var requested = data.settings || {};
      var topKeys = requested.json_structure
        ? Object.keys(requested.json_structure).join(", ")
        : "—";
      var finishC = controlled
        ? (controlled.finish_reason || "—")
        : (data.controlled_finish_reason || "недоступна (запрос завершился ошибкой)");
      var items = [
        "Запрос отправлен в DeepSeek дважды — один и тот же текст, без изменений.",
        "Формат: response_format = " + JSON.stringify(requested.response_format) +
          " — режим JSON-объекта для контролируемого запроса.",
        "Структура JSON: корневые поля — «" + topKeys + "» (из инструкции в messages).",
        "Длина: max_tokens = " + (requested.max_tokens != null ? requested.max_tokens : "—"),
        "Завершение: stop = " + JSON.stringify(requested.stop),
        "Причина завершения: без ограничений — " + (data.unrestricted.finish_reason || "—") +
          "; с ограничениями — " + finishC + "."
      ];
      if (!controlled) {
        items.push("Контролируемый запрос не выполнен — параметры показаны как запрошенные.");
      }
      summaryList.innerHTML = items
        .map(function (item) { return "<li>" + escapeHtml(item) + "</li>"; })
        .join("");
    }

    /* ---------------- Helpers ---------------- */
    function setLoading(on) {
      inFlight = on;
      button.disabled = on;
      button.textContent = on ? "Выполняется…" : "Сравнить ответы";
      loading.classList.toggle("hidden", !on);
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.classList.remove("hidden");
    }

    function hideError() {
      errorBox.classList.add("hidden");
      errorBox.textContent = "";
    }

    function hideResults() {
      resultsSection.classList.add("hidden");
      summarySection.classList.add("hidden");
    }

    function escapeHtml(text) {
      var div = document.createElement("div");
      div.textContent = String(text);
      return div.innerHTML;
    }
  }

  // The script sits at the end of <body>; run safely even if moved to <head>.
  function boot() {
    initTabs();
    initApp();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
