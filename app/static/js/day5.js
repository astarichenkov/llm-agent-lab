/* DeepSeek Study App — Day 5: OpenRouter model comparison.
 * Vanilla JS. One Send = one POST /api/openrouter/run (one paid call).
 */
(function () {
  "use strict";

  function initDay5() {
    var $ = function (id) { return document.getElementById(id); };
    var slots = ["weak", "medium", "strong"];
    var labels = { weak: "Слабая модель", medium: "Средняя модель", strong: "Сильная модель" };
    var required = [
      "tab-day2", "tab-day3", "tab-day4", "tab-day5",
      "panel-day2", "panel-day3", "panel-day4", "panel-day5",
      "d5-task", "d5-max", "d5-temp", "d5-stop", "d5-fair",
      "d5-compare-tbody", "d5-answers", "d5-links", "d5-run-all",
      "d5-runall-loading", "d5-conclusion",
      "d5-sub-weak", "d5-sub-medium", "d5-sub-strong", "d5-sub-out",
      "d5-panel-weak", "d5-panel-medium", "d5-panel-strong", "d5-panel-out"
    ];
    slots.forEach(function (k) {
      required.push("d5-" + k + "-model", "d5-" + k + "-meta", "d5-" + k + "-send",
        "d5-" + k + "-loading", "d5-" + k + "-error", "d5-" + k + "-metrics", "d5-" + k + "-answer");
      required.push("d5-r-" + k + "-qual", "d5-r-" + k + "-acc", "d5-r-" + k + "-util", "d5-r-" + k + "-comment");
    });
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) { console.error("Day5 missing DOM:", missing.join(", ")); return; }

    var catalog = {};  // id -> meta
    var defaults = {};
    var results = {};  // slot -> snapshot
    var running = {};

    function setText(el, t) { el.textContent = (t == null ? "" : String(t)); }
    function showErr(el, m) { setText(el, m); el.classList.remove("hidden"); }
    function hideErr(el) { el.classList.add("hidden"); setText(el, ""); }
    function norm(s) { return String(s || "").replace(/\s+/g, " ").trim(); }

    // main tabs (all)
    function switchMain(name) {
      var panels = { day2: "panel-day2", day3: "panel-day3", day4: "panel-day4", day5: "panel-day5" };
      Object.keys(panels).forEach(function (k) { $(panels[k]).style.display = (k === name) ? "block" : "none"; });
      var btns = { day2: "tab-day2", day3: "tab-day3", day4: "tab-day4", day5: "tab-day5" };
      Object.keys(btns).forEach(function (k) {
        var on = k === name;
        $(btns[k]).classList.toggle("active", on);
        $(btns[k]).setAttribute("aria-selected", on ? "true" : "false");
      });
    }
    ["day2", "day3", "day4", "day5"].forEach(function (n) {
      $("tab-" + n).addEventListener("click", function () { switchMain(n); });
    });

    // sub tabs
    function switchSub(name) {
      slots.concat(["out"]).forEach(function (k) {
        $(k === "out" ? "d5-panel-out" : "d5-panel-" + k).classList.toggle("hidden", k !== name);
      });
      var b = { weak: "d5-sub-weak", medium: "d5-sub-medium", strong: "d5-sub-strong", out: "d5-sub-out" };
      Object.keys(b).forEach(function (k) { $(b[k]).classList.toggle("active", k === name); });
      if (name === "out") refreshOut();
    }
    $("d5-sub-weak").addEventListener("click", function () { switchSub("weak"); });
    $("d5-sub-medium").addEventListener("click", function () { switchSub("medium"); });
    $("d5-sub-strong").addEventListener("click", function () { switchSub("strong"); });
    $("d5-sub-out").addEventListener("click", function () { switchSub("out"); });

    // catalog
    function priceText(m) {
      if (!m) return "н/д";
      var ip = (m.input_per_million != null) ? "$" + m.input_per_million + "/1M in" : "цена н/д";
      var op = (m.output_per_million != null) ? "$" + m.output_per_million + "/1M out" : "цена н/д";
      var ctx = m.context_length ? ("ctx " + m.context_length) : "ctx н/д";
      return ip + " · " + op + " · " + ctx;
    }
    function populate() {
      slots.forEach(function (k) {
        var sel = $("d5-" + k + "-model");
        sel.innerHTML = "";
        Object.keys(catalog).forEach(function (id) {
          var m = catalog[id];
          var o = document.createElement("option");
          o.value = id; o.textContent = m.name + " — " + id;
          sel.appendChild(o);
        });
        sel.value = defaults[k] || (Object.keys(catalog)[0] || "");
        $("d5-" + k + "-meta").textContent = "Цена: " + priceText(catalog[sel.value]);
        sel.addEventListener("change", function () {
          $("d5-" + k + "-meta").textContent = "Цена: " + priceText(catalog[sel.value]);
        });
      });
    }
    fetch("/api/openrouter/models").then(function (r) { return r.json(); }).then(function (d) {
      (d.models || []).forEach(function (m) { catalog[m.id] = m; });
      defaults = d.defaults || {};
      populate();
    }).catch(function () {
      slots.forEach(function (k) { $("d5-" + k + "-meta").textContent = "не удалось получить каталог"; });
    });

    function http(payload, btn, ld, errEl) {
      btn.disabled = true; ld.classList.remove("hidden"); if (errEl) hideErr(errEl);
      return fetch("/api/openrouter/run", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; }); })
        .then(function (res) { if (!res.ok) { throw new Error(res.data.detail || ("Ошибка (" + res.status + ")")); } return res.data; })
        .catch(function (err) { if (errEl) showErr(errEl, err.message || "Ошибка."); throw err; })
        .finally(function () { btn.disabled = false; ld.classList.add("hidden"); });
    }

    function common(payload, errEl) {
      var msg = $("d5-task").value.trim();
      if (!msg) { showErr(errEl, "Введите запрос."); return null; }
      var mt = parseInt($("d5-max").value, 10);
      var temp = parseFloat($("d5-temp").value);
      if (!Number.isInteger(mt) || mt < 16 || mt > 2000) { showErr(errEl, "max_tokens 16..2000."); return null; }
      if (!isFinite(temp) || temp < 0 || temp > 2) { showErr(errEl, "temperature 0..2."); return null; }
      payload.message = msg; payload.max_tokens = mt; payload.temperature = temp;
      var stop = $("d5-stop").value.trim();
      if (stop) { payload.stop_sequence = stop; }
      return payload;
    }

    function costText(c, est) {
      if (c == null) return "н/д";
      return "$" + c.toFixed(6) + (est ? " (оценочная)" : "");
    }

    function runSlot(k) {
      var model = $("d5-" + k + "-model").value;
      if (!model) { showErr($("d5-" + k + "-error"), "Выберите модель."); return; }
      var payload = common({ model: model }, $("d5-" + k + "-error"));
      if (!payload) return;
      running[k] = true;
      http(payload, $("d5-" + k + "-send"), $("d5-" + k + "-loading"), $("d5-" + k + "-error"))
        .then(function (d) { render(k, d); })
        .catch(function () {})
        .finally(function () { running[k] = false; });
    }
    slots.forEach(function (k) { $("d5-" + k + "-send").addEventListener("click", function () { runSlot(k); }); });

    function render(k, d) {
      var u = d.usage || {};
      var mismatch = d.actual_model !== d.requested_model;
      var lines = [];
      lines.push("Запрошенная модель: " + d.requested_model + " · Фактическая: " + d.actual_model + (mismatch ? " (ОТЛИЧАЕТСЯ)" : ""));
      lines.push("Время: " + (d.elapsed_ms != null ? Math.round(d.elapsed_ms) + " мс" : "н/д"));
      lines.push("Prompt: " + (u.prompt_tokens != null ? u.prompt_tokens : "н/д") +
        " · Completion: " + (u.completion_tokens != null ? u.completion_tokens : "н/д") +
        " · Total: " + (u.total_tokens != null ? u.total_tokens : "н/д"));
      lines.push("Стоимость запроса: " + costText(d.cost, d.cost_estimated));
      lines.push("finish_reason: " + (d.finish_reason || "—"));
      if (d.model_url) lines.push("Страница модели: " + d.model_url);
      setText($("d5-" + k + "-metrics"), lines.join("\n"));
      setText($("d5-" + k + "-answer"), d.answer || "");
      results[k] = {
        level: labels[k], model: d.requested_model, actual: d.actual_model,
        prompt: payloadPromptSnapshot(), max: parseInt($("d5-max").value, 10),
        temp: parseFloat($("d5-temp").value), stop: $("d5-stop").value.trim(),
        answer: d.answer || "", finish: d.finish_reason || "",
        elapsed: d.elapsed_ms, u: u, cost: d.cost, costEst: d.cost_estimated,
        url: d.model_url
      };
      refreshOut();
    }
    function payloadPromptSnapshot() { return norm($("d5-task").value); }

    function refreshOut() {
      var tbody = $("d5-compare-tbody");
      tbody.innerHTML = "";
      var done = [];
      slots.forEach(function (k) {
        var r = results[k];
        var tr = document.createElement("tr");
        var cells = [
          r ? r.level : labels[k],
          r ? (r.model + (r.model !== r.actual ? " → " + r.actual : "")) : "—",
          r ? (r.elapsed != null ? Math.round(r.elapsed) + " мс" : "н/д") : "н/д",
          r ? ((r.u && r.u.prompt_tokens) != null ? r.u.prompt_tokens : "н/д") : "н/д",
          r ? ((r.u && r.u.completion_tokens) != null ? r.u.completion_tokens : "н/д") : "н/д",
          r ? ((r.u && r.u.total_tokens) != null ? r.u.total_tokens : "н/д") : "н/д",
          r ? costText(r.cost, r.costEst) : "н/д",
          r ? r.finish : "—"
        ];
        cells.forEach(function (c) { var td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
        tbody.appendChild(tr);
        if (r) done.push(r);
      });

      // answers
      var ab = $("d5-answers"); ab.innerHTML = "";
      slots.forEach(function (k) {
        var r = results[k]; if (!r) return;
        var det = document.createElement("details");
        var sum = document.createElement("summary"); sum.textContent = labels[k] + " (" + r.model + ")";
        var pre = document.createElement("pre"); pre.className = "api-preview"; pre.textContent = r.answer;
        det.appendChild(sum); det.appendChild(pre); ab.appendChild(det);
      });

      // fair check
      if (!done.length) { setText($("d5-fair"), "Выполните хотя бы один запрос."); return; }
      var lines = [];
      var prompts = done.map(function (r) { return norm(r.prompt); });
      var mts = done.map(function (r) { return r.max; });
      var temps = done.map(function (r) { return r.temp; });
      var stops = done.map(function (r) { return norm(r.stop); });
      var eq = function (a) { return a.every(function (v) { return v === a[0]; }); };
      if (eq(prompts)) lines.push("✓ Один и тот же запрос"); else lines.push("✗ Разные запросы");
      if (eq(mts)) lines.push("✓ Одинаковый max_tokens"); else lines.push("✗ Разный max_tokens");
      if (eq(temps)) lines.push("✓ Одинаковая temperature"); else lines.push("✗ Разная temperature");
      if (eq(stops)) lines.push("✓ Одинаковый stop"); else lines.push("✗ Разный stop");
      var distinct = new Set(done.map(function (r) { return r.model; }));
      if (distinct.size === done.length) lines.push("✓ Использованы разные модели"); else lines.push("✗ Одинаковые/повторяющиеся модели");
      setText($("d5-fair"), lines.join("\n"));

      // links
      var lk = $("d5-links"); lk.innerHTML = "";
      var seen = {};
      done.forEach(function (r) {
        if (!r.url || seen[r.url]) return; seen[r.url] = 1;
        var a = document.createElement("a"); a.href = r.url; a.target = "_blank"; a.rel = "noopener";
        a.textContent = r.model + " — страница модели"; lk.appendChild(a); lk.appendChild(document.createTextNode(" · "));
      });
      var cat = document.createElement("a"); cat.href = "https://openrouter.ai/models"; cat.target = "_blank"; cat.rel = "noopener";
      cat.textContent = "Каталог OpenRouter";
      lk.appendChild(cat);
    }

    // run all (explicit)
    $("d5-run-all").addEventListener("click", function () {
      if (Object.keys(running).some(function (k) { return running[k]; })) return;
      var btn = $("d5-run-all"), ld = $("d5-runall-loading");
      btn.disabled = true; ld.classList.remove("hidden");
      var chain = Promise.resolve();
      slots.forEach(function (k) {
        chain = chain.then(function () {
          var model = $("d5-" + k + "-model").value;
          var payload = common({ model: model }, $("d5-" + k + "-error"));
          if (!payload) return;
          return http(payload, $("d5-" + k + "-send"), $("d5-" + k + "-loading"), $("d5-" + k + "-error"))
            .then(function (d) { render(k, d); }).catch(function () {});
        });
      });
      chain.finally(function () { btn.disabled = false; ld.classList.add("hidden"); });
    });

    switchMain("day2");
    switchSub("weak");
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", initDay5); }
  else { initDay5(); }
})();
