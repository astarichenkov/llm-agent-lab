/* LLM Agent Lab — Day 4: temperature experiment.
 * Vanilla JS. One Send = one POST /api/temperature (one provider call).
 */
(function () {
  "use strict";

  function initDay4() {
    var $ = function (id) { return document.getElementById(id); };

    var required = [
      "tab-day2", "tab-day3", "tab-day4", "panel-day2", "panel-day3", "panel-day4",
      "d4-task", "d4-compare-tbody", "d4-fair-indicator", "d4-answers-block",
      "d4-conclusion"
    ];
    var items = [
      { key: "0", label: "0", panel: "t0", def: 0 },
      { key: "07", label: "0.7", panel: "t07", def: 0.7 },
      { key: "12", label: "1.2", panel: "t12", def: 1.2 }
    ];
    items.forEach(function (it) {
      required.push("d4-" + it.key + "-temp", "d4-" + it.key + "-max", "d4-" + it.key + "-stop",
        "d4-" + it.key + "-send", "d4-" + it.key + "-loading", "d4-" + it.key + "-error",
        "d4-" + it.key + "-params", "d4-" + it.key + "-answer", "d4-" + it.key + "-finish",
        "d4-" + it.key + "-usage");
      required.push("d4-r-" + it.key + "-acc", "d4-r-" + it.key + "-crea", "d4-r-" + it.key + "-dive", "d4-r-" + it.key + "-comment");
    });
    ["d4-sub-0", "d4-sub-07", "d4-sub-12", "d4-sub-out",
     "d4-panel-t0", "d4-panel-t07", "d4-panel-t12", "d4-panel-out",
     "d4-out-t0use", "d4-out-t07use", "d4-out-t12use"].forEach(function (id) { required.push(id); });

    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) { console.error("Day4: missing DOM:", missing.join(", ")); return; }

    var results = {}; // key -> snapshot

    function setText(el, t) { el.textContent = (t == null ? "" : String(t)); }
    function showErr(el, m) { setText(el, m); el.classList.remove("hidden"); }
    function hideErr(el) { el.classList.add("hidden"); setText(el, ""); }
    function norm(s) { return String(s || "").replace(/\s+/g, " ").trim(); }

    // Main tab switching is centralized in app.js (initTabs / switchMainTab).

    // ---------- Day4 sub-tabs ----------
    function switchSub(name) {
      var panels = { t0: "d4-panel-t0", t07: "d4-panel-t07", t12: "d4-panel-t12", out: "d4-panel-out" };
      Object.keys(panels).forEach(function (k) { $(panels[k]).classList.toggle("hidden", k !== name); });
      var btns = { t0: "d4-sub-0", t07: "d4-sub-07", t12: "d4-sub-12", out: "d4-sub-out" };
      Object.keys(btns).forEach(function (k) { $(btns[k]).classList.toggle("active", k === name); });
      if (name === "out") refreshMyOutput();
    }
    $("d4-sub-0").addEventListener("click", function () { switchSub("t0"); });
    $("d4-sub-07").addEventListener("click", function () { switchSub("t07"); });
    $("d4-sub-12").addEventListener("click", function () { switchSub("t12"); });
    $("d4-sub-out").addEventListener("click", function () { switchSub("out"); });

    // ---------- HTTP ----------
    function http(payload, btn, ld, errEl) {
      btn.disabled = true; ld.classList.remove("hidden"); if (errEl) hideErr(errEl);
      return fetch("/api/temperature", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; }); })
        .then(function (res) { if (!res.ok) { throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")")); } return res.data; })
        .catch(function (err) { if (errEl) showErr(errEl, err.message || "Что-то пошло не так."); throw err; })
        .finally(function () { btn.disabled = false; ld.classList.add("hidden"); });
    }

    function usageLine(u) {
      if (!u) return "промпт: н/д · вывод: н/д · всего: н/д";
      return "промпт: " + (u.prompt_tokens != null ? u.prompt_tokens : "н/д") +
        " · вывод: " + (u.completion_tokens != null ? u.completion_tokens : "н/д") +
        " · всего: " + (u.total_tokens != null ? u.total_tokens : "н/д");
    }

    function send(key) {
      var errEl = $("d4-" + key + "-error");
      var msg = $("d4-task").value.trim();
      if (!msg) { showErr(errEl, "Введите общий запрос."); return; }
      var temp = parseFloat($("d4-" + key + "-temp").value);
      if (!isFinite(temp) || temp < 0 || temp > 2) { showErr(errEl, "temperature: число от 0 до 2."); return; }
      var mt = parseInt($("d4-" + key + "-max").value, 10);
      if (!Number.isInteger(mt) || mt < 16 || mt > 2000) { showErr(errEl, "max_tokens: целое от 16 до 2000."); return; }
      var stop = $("d4-" + key + "-stop").value.trim();
      var payload = { message: msg, temperature: temp, max_tokens: mt };
      if (stop) {
        if (stop.length < 4 || !/[A-Za-zА-Яа-я0-9]/.test(stop)) { showErr(errEl, "stop — различимый маркер (от 4 символов)."); return; }
        payload.stop_sequence = stop;
      }
      http(payload, $("d4-" + key + "-send"), $("d4-" + key + "-loading"), errEl)
        .then(function (data) {
          var ap = data.applied_parameters || {};
          var stopShown = (ap.stop && ap.stop.length) ? JSON.stringify(ap.stop) : "не задан";
          setText($("d4-" + key + "-params"),
            "Режим: Non-Thinking (thinking: disabled)\nmodel: " + ap.model + "\ntemperature: " + ap.temperature + "\nmax_tokens: " + ap.max_tokens + "\nstop: " + stopShown);
          setText($("d4-" + key + "-answer"), data.answer || "");
          setText($("d4-" + key + "-finish"), data.finish_reason || "—");
          if (data.finish_reason === "length") {
            showErr($("d4-" + key + "-error"), "Ответ достиг ограничения max_tokens.");
          } else { hideErr(errEl); }
          setText($("d4-" + key + "-usage"), usageLine(data.usage));
          results[key] = {
            prompt: msg, temperature: ap.temperature, max_tokens: ap.max_tokens,
            stop: ap.stop ? ap.stop.slice() : null,
            answer: data.answer || "", finish: data.finish_reason || "",
            outTokens: (data.usage && data.usage.completion_tokens != null) ? data.usage.completion_tokens : null
          };
        })
        .catch(function () {});
    }

    items.forEach(function (it) { $("d4-" + it.key + "-send").addEventListener("click", function () { send(it.key); }); });

    // ---------- My output ----------
    function refreshMyOutput() {
      var tbody = $("d4-compare-tbody");
      tbody.innerHTML = "";
      var labels = { "0": "0", "07": "0.7", "12": "1.2" };
      var done = [];
      items.forEach(function (it) {
        var r = results[it.key];
        var tr = document.createElement("tr");
        var doneText = r ? "Да" : "Нет";
        [labels[it.key], r ? r.finish : "—", r ? (r.outTokens != null ? r.outTokens : "н/д") : "н/д", doneText]
          .forEach(function (c) { var td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
        tbody.appendChild(tr);
        if (r) done.push(r);
      });

      // answers collapsible
      var block = $("d4-answers-block");
      block.innerHTML = "";
      items.forEach(function (it) {
        var r = results[it.key];
        if (!r) return;
        var det = document.createElement("details");
        var sum = document.createElement("summary");
        sum.textContent = "Ответ при temperature = " + labels[it.key];
        var pre = document.createElement("pre"); pre.className = "api-preview"; pre.textContent = r.answer;
        det.appendChild(sum); det.appendChild(pre); block.appendChild(det);
      });

      // fair-comparison indicator
      if (done.length === 0) { setText($("d4-fair-indicator"), "Выполните хотя бы один эксперимент."); return; }
      var lines = [];
      var prompts = done.map(function (r) { return norm(r.prompt); });
      var mts = done.map(function (r) { return r.max_tokens; });
      var stops = done.map(function (r) { return norm(JSON.stringify(r.stop)); });
      var allEq = function (arr) { return arr.every(function (v) { return v === arr[0]; }); };
      var integrity = [];
      if (allEq(prompts)) { lines.push("✓ Один и тот же запрос"); }
      else { lines.push("✗ Разные запросы"); integrity.push("Некоторые результаты получены с разными текстами запроса."); }
      if (allEq(mts)) { lines.push("✓ Одинаковый max_tokens"); }
      else { lines.push("✗ Разный max_tokens"); integrity.push("Различается max_tokens."); }
      if (allEq(stops)) { lines.push("✓ Одинаковый stop"); }
      else { lines.push("✗ Разный stop"); integrity.push("Различается stop."); }
      lines.push("Различается temperature (это и есть эксперимент)");
      if (integrity.length) {
        lines.unshift("ВНИМАНИЕ: для корректного сравнения необходимо использовать один и тот же запрос и одинаковые max_tokens/stop. " + integrity.join(" "));
      }
      setText($("d4-fair-indicator"), lines.join("\n"));
    }

    switchSub("t0");
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", initDay4); }
  else { initDay4(); }
})();
