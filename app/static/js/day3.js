/* LLM Agent Lab — Day 3: reasoning-strategy comparison.
 * Vanilla JavaScript. Talks to POST /api/reasoning (one provider call per
 * request; Method 3 uses two separate requests driven by the user).
 */
(function () {
  "use strict";

  function initDay3() {
    var $ = function (id) { return document.getElementById(id); };

    var required = [
      "tab-day2", "tab-day3", "panel-day2", "panel-day3",
      "d3-task", "d3-max-tokens", "d3-stop", "d3-ref-note",
      "d3-compare-section", "d3-compare-tbody", "d3-conclusion-text", "d3-conclusion",
      "d3-m1-run", "d3-m1-loading", "d3-m1-error", "d3-m1-prompt", "d3-m1-answer",
      "d3-m1-finish", "d3-m1-usage", "d3-m1-status",
      "d3-m2-run", "d3-m2-loading", "d3-m2-error", "d3-m2-prompt", "d3-m2-answer",
      "d3-m2-finish", "d3-m2-usage", "d3-m2-status",
      "d3-m3-gen", "d3-m3-gen-loading", "d3-m3-gen-error", "d3-m3-prompt",
      "d3-m3-use", "d3-m3-use-loading", "d3-m3-use-error", "d3-m3-sent",
      "d3-m3-answer", "d3-m3-finish", "d3-m3-usage", "d3-m3-status",
      "d3-m4-run", "d3-m4-loading", "d3-m4-error", "d3-m4-prompt", "d3-m4-answer",
      "d3-m4-finish", "d3-m4-usage", "d3-m4-status"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day 3: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var DEFAULT_TASK = $("d3-task").value;
    var results = {}; // key -> {label,calls,finish,tokens,status,chars}

    function setText(el, txt) { el.textContent = (txt == null ? "" : String(txt)); }
    function showErr(el, msg) { setText(el, msg); el.classList.remove("hidden"); }
    function hideErr(el) { el.classList.add("hidden"); setText(el, ""); }
    function norm(s) { return s.replace(/\s+/g, " ").trim(); }
    function isDefaultTask() { return norm($("d3-task").value) === norm(DEFAULT_TASK); }
    function usageText(u) { return (u && typeof u.completion_tokens === "number") ? u.completion_tokens : "н/д"; }
    function statusText(s) {
      return { correct: "Совпадает", incorrect: "Не совпадает", indeterminate: "Не удалось определить" }[s] || "";
    }
    function statusClass(s) { return { correct: "status-correct", incorrect: "status-incorrect", indeterminate: "status-indeterminate" }[s] || ""; }

    // Main tab switching is centralized in app.js (initTabs / switchMainTab).

    // ---------- HTTP ----------
    function http(payload, btn, ld, errEl) {
      btn.disabled = true;
      ld.classList.remove("hidden");
      if (errEl) hideErr(errEl);
      return fetch("/api/reasoning", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; }); })
        .then(function (res) {
          if (!res.ok) { throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")")); }
          return res.data;
        })
        .catch(function (err) { if (errEl) showErr(errEl, err.message || "Что-то пошло не так."); throw err; })
        .finally(function () { btn.disabled = false; ld.classList.add("hidden"); });
    }

    function commonBase(errEl) {
      var task = $("d3-task").value.trim();
      if (!task) { showErr(errEl, "Введите задачу — она используется всеми способами."); return null; }
      var mt = parseInt($("d3-max-tokens").value, 10);
      if (!Number.isInteger(mt) || mt < 16 || mt > 2000) {
        showErr(errEl, "Максимальная длина ответа: целое число от 16 до 2000.");
        return null;
      }
      var base = { task: task, max_tokens: mt };
      var stop = $("d3-stop").value.trim();
      if (stop) {
        if (stop.length < 4 || !/[A-Za-zА-Яа-я0-9]/.test(stop)) {
          showErr(errEl, "Стоп-последовательность — различимый маркер (минимум 4 символа).");
          return null;
        }
        base.stop_sequence = stop;
      }
      return base;
    }

    function renderSolution(prefix, data, promptId) {
      setText($(promptId), data.prompt_sent || "");
      setText($("d3-" + prefix + "-answer"), data.solution || "");
      setText($("d3-" + prefix + "-finish"), data.finish_reason || "—");
      setText($("d3-" + prefix + "-usage"), " · токены: " + usageText(data.usage));
      var se = $("d3-" + prefix + "-status");
      var st = statusText(data.status);
      se.textContent = st ? " · " + st : "";
      se.className = statusClass(data.status);
    }

    function record(key, label, calls, data) {
      results[key] = {
        label: label, calls: calls,
        finish: data.finish_reason || "—",
        tokens: usageText(data.usage),
        status: data.status || null,
        chars: (data.solution || "").length
      };
      refreshCompare();
    }

    function refreshCompare() {
      var order = [
        ["direct", "Прямой ответ", 1],
        ["step_by_step", "Пошаговое решение", 1],
        ["use_prompt", "Сгенерированный промпт", 2],
        ["experts", "Группа экспертов", 1]
      ];
      var tbody = $("d3-compare-tbody");
      tbody.innerHTML = "";
      var present = false;
      var anyCorrect = false, sawStatus = false, all = [];
      order.forEach(function (row) {
        var r = results[row[0]];
        if (!r) return;
        present = true;
        all.push(r);
        if (r.status) { sawStatus = true; if (r.status === "correct") anyCorrect = true; }
        var tr = document.createElement("tr");
        var cells = [r.label, String(row[2]), r.finish, r.tokens,
          statusText(r.status) || (r.status === null ? "эталон не задан" : "—"),
          r.status === null ? "своя задача (без эталона)" : (r.status === "indeterminate" ? "не удалось однозначно распознать" : "проверено по эталону")];
        cells.forEach(function (c) { var td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
        tbody.appendChild(tr);
      });
      if (!present) { $("d3-compare-section").classList.add("hidden"); return; }
      $("d3-compare-section").classList.remove("hidden");
      var conclusion = [];
      if (sawStatus && anyCorrect && all.every(function (r) { return r.status === "correct"; })) {
        conclusion.push("Все выполненные методы дали правильный итоговый ответ для демонстрационной задачи.");
      }
      if (all.length > 1) {
        var sorted = all.slice().sort(function (a, b) { return a.chars - b.chars; });
        conclusion.push("Самый короткий ответ — «" + sorted[0].label + "» (" + sorted[0].chars + " симв.), самый подробный — «" +
          sorted[sorted.length - 1].label + "» (" + sorted[sorted.length - 1].chars + " симв.).");
        conclusion.push("Детальность рассуждения не означает автоматически большую точность. Сравнивайте методы по корректности, длине, числу API-вызовов и токенам.");
      }
      setText($("d3-conclusion-text"), conclusion.join(" "));
    }

    // reference note visibility
    function updateRefNote() {
      var note = $("d3-ref-note");
      if (isDefaultTask()) { note.classList.remove("hidden"); } else { note.classList.add("hidden"); }
    }
    $("d3-task").addEventListener("input", updateRefNote);
    updateRefNote();

    // Method 1 — direct
    $("d3-m1-run").addEventListener("click", function () {
      var base = commonBase($("d3-m1-error")); if (!base) return;
      base.method = "direct";
      http(base, $("d3-m1-run"), $("d3-m1-loading"), $("d3-m1-error"))
        .then(function (data) { renderSolution("m1", data, "d3-m1-prompt"); record("direct", "Прямой ответ", 1, data); })
        .catch(function () {});
    });

    // Method 2 — step by step
    $("d3-m2-run").addEventListener("click", function () {
      var base = commonBase($("d3-m2-error")); if (!base) return;
      base.method = "step_by_step";
      http(base, $("d3-m2-run"), $("d3-m2-loading"), $("d3-m2-error"))
        .then(function (data) { renderSolution("m2", data, "d3-m2-prompt"); record("step_by_step", "Пошаговое решение", 1, data); })
        .catch(function () {});
    });

    // Method 3 — generate then use
    function m3SetUseEnabled() { $("d3-m3-use").disabled = !($("d3-m3-prompt").value.trim()); }
    $("d3-m3-prompt").addEventListener("input", m3SetUseEnabled);
    m3SetUseEnabled();
    $("d3-m3-gen").addEventListener("click", function () {
      var base = commonBase($("d3-m3-gen-error")); if (!base) return;
      base.method = "generate_prompt";
      http(base, $("d3-m3-gen"), $("d3-m3-gen-loading"), $("d3-m3-gen-error"))
        .then(function (data) {
          $("d3-m3-prompt").value = data.generated_prompt || "";
          m3SetUseEnabled();
        })
        .catch(function () {});
    });
    $("d3-m3-use").addEventListener("click", function () {
      var gp = $("d3-m3-prompt").value.trim();
      if (!gp) { showErr($("d3-m3-use-error"), "Сначала создайте или введите промпт."); return; }
      var base = commonBase($("d3-m3-use-error")); if (!base) return;
      base.method = "use_prompt";
      base.generated_prompt = gp;
      http(base, $("d3-m3-use"), $("d3-m3-use-loading"), $("d3-m3-use-error"))
        .then(function (data) { renderSolution("m3", data, "d3-m3-sent"); record("use_prompt", "Сгенерированный промпт", 2, data); })
        .catch(function () {});
    });

    // Method 4 — experts
    $("d3-m4-run").addEventListener("click", function () {
      var base = commonBase($("d3-m4-error")); if (!base) return;
      base.method = "experts";
      http(base, $("d3-m4-run"), $("d3-m4-loading"), $("d3-m4-error"))
        .then(function (data) { renderSolution("m4", data, "d3-m4-prompt"); record("experts", "Группа экспертов", 1, data); })
        .catch(function () {});
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay3);
  } else {
    initDay3();
  }
})();
