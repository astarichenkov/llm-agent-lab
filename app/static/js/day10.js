/* LLM Agent Lab — Day 10: context-management strategies.
 * Vanilla JS. Talks to /api/day10/*. All strategy logic (window, facts
 * extraction, branch isolation) lives on the backend; this file only drives
 * the UI and the demo scenario.
 */
(function () {
  "use strict";

  var STRATEGIES = ["sliding_window", "sticky_facts", "branching"];

  function initDay10() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day10", "panel-day10",
      "d10-s-sliding", "d10-s-sticky", "d10-s-branching",
      "d10-strategy-name", "d10-model", "d10-history-count", "d10-active-branch",
      "d10-window-input", "d10-recent-input", "d10-update-facts",
      "d10-seed-branches", "d10-demo-run", "d10-compare-run", "d10-clear",
      "d10-loading", "d10-progress", "d10-note",
      "d10-branch-panel", "d10-checkpoint", "d10-branch-list",
      "d10-branch-name", "d10-branch-create",
      "d10-messages", "d10-input", "d10-send", "d10-error",
      "d10-sys", "d10-facts-tokens", "d10-recent-tokens", "d10-cur",
      "d10-est-input", "d10-api-input", "d10-api-output", "d10-api-total",
      "d10-facts-extract", "d10-cost",
      "d10-cum-requests", "d10-cum-extractions", "d10-cum-input",
      "d10-cum-output", "d10-cum-total", "d10-cum-extract",
      "d10-sliding-panel", "d10-sw-summary", "d10-full-count",
      "d10-sent-count", "d10-dropped-count", "d10-dropped-list",
      "d10-facts-panel", "d10-facts-count", "d10-facts-recent",
      "d10-facts-block", "d10-facts-json", "d10-facts-error",
      "d10-compare-rows"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day10: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      inFlight: false,
      cumulative: {},
      results: {},
      seeded: false
    };
    STRATEGIES.forEach(function (s) {
      state.cumulative[s] = {
        requests: 0, extractions: 0, input: 0, output: 0, total: 0, extract: 0
      };
      state.results[s] = {
        remembered: "—", input: 0, extra: 0, stability: "—", ux: "—"
      };
    });

    /* ---------------- helpers ---------------- */
    function currentStrategy() {
      if ($("d10-s-sticky").checked) return "sticky_facts";
      if ($("d10-s-branching").checked) return "branching";
      return "sliding_window";
    }
    function fmt(n) {
      if (n === null || n === undefined) return "—";
      return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
    }
    function setError(m) {
      $("d10-error").textContent = m || "";
      $("d10-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d10-note").textContent = m || "";
      $("d10-note").classList.toggle("hidden", !m);
    }
    function setProgress(m) { $("d10-progress").textContent = m || "—"; }
    function setLoading(on) {
      state.inFlight = on;
      $("d10-send").disabled = on;
      $("d10-demo-run").disabled = on;
      $("d10-compare-run").disabled = on;
      $("d10-loading").classList.toggle("hidden", !on);
    }
    function http(path, opts) {
      return fetch(path, opts).then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, status: r.status, data: d };
        });
      });
    }
    function buildBody(strategy, text) {
      var body = { strategy: strategy, message: text };
      body.update_facts = $("d10-update-facts").checked;
      var w = parseInt($("d10-window-input").value, 10);
      var rec = parseInt($("d10-recent-input").value, 10);
      if (Number.isInteger(w)) body.window_size = w;
      if (Number.isInteger(rec)) body.recent_messages_limit = rec;
      return body;
    }

    /* ---------------- rendering ---------------- */
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
      $("d10-messages").appendChild(bubble);
      $("d10-messages").scrollTop = $("d10-messages").scrollHeight;
    }
    function renderMessages(messages) {
      $("d10-messages").innerHTML = "";
      (messages || []).forEach(function (m) { addBubble(m.role, m.content); });
    }
    function messageList(el, messages) {
      el.innerHTML = "";
      if (!messages || !messages.length) {
        var empty = document.createElement("div");
        empty.className = "d10-msg-empty";
        empty.textContent = "(нет)";
        el.appendChild(empty);
        return;
      }
      messages.forEach(function (m) {
        var row = document.createElement("div");
        row.className = "d10-msg-row";
        var role = document.createElement("span");
        role.className = "d10-msg-role";
        role.textContent = m.role === "user" ? "Вы" : "Агент";
        var text = document.createElement("span");
        text.className = "d10-msg-text";
        text.textContent = m.content;
        row.appendChild(role);
        row.appendChild(text);
        el.appendChild(row);
      });
    }
    function setPanelVisibility(strategy) {
      $("d10-sliding-panel").classList.toggle("hidden", strategy !== "sliding_window");
      $("d10-facts-panel").classList.toggle("hidden", strategy !== "sticky_facts");
      $("d10-branch-panel").classList.toggle("hidden", strategy !== "branching");
    }
    function renderContext(ctx) {
      if (!ctx) return;
      setPanelVisibility(ctx.strategy);
      $("d10-strategy-name").textContent = ctx.strategy;
      $("d10-history-count").textContent = fmt(ctx.total_history_messages) + " сообщений";
      $("d10-active-branch").textContent =
        ctx.active_branch_name || "—";

      if (ctx.strategy === "sliding_window") {
        $("d10-sw-summary").textContent =
          "Conversation history: " + fmt(ctx.total_history_messages) + " messages · " +
          "Sent to LLM: " + fmt(ctx.sent_history_messages) + " messages · " +
          "Dropped: " + fmt(ctx.dropped_messages) + " messages";
        $("d10-full-count").textContent = fmt(ctx.total_history_messages) + " messages";
        $("d10-sent-count").textContent = fmt(ctx.sent_history_messages) + " messages";
        $("d10-dropped-count").textContent = fmt(ctx.dropped_messages) + " messages";
        messageList($("d10-dropped-list"), ctx.dropped_preview);
      }

      if (ctx.strategy === "sticky_facts") {
        var facts = ctx.facts || {};
        var count =
          (facts.goal ? 1 : 0) +
          ((facts.constraints || []).length) +
          ((facts.preferences || []).length) +
          ((facts.decisions || []).length) +
          ((facts.agreements || []).length);
        $("d10-facts-count").textContent = fmt(count);
        $("d10-facts-recent").textContent =
          fmt(ctx.sent_history_messages) + " messages";
        $("d10-facts-block").textContent = ctx.facts_block || "(facts ещё не собраны)";
        $("d10-facts-json").textContent = JSON.stringify(facts, null, 2);
      }

      if (ctx.strategy === "branching") {
        $("d10-checkpoint").textContent = ctx.checkpoint_message_id
          ? "#" + ctx.checkpoint_message_id
          : "—";
        renderBranches(ctx);
      }
    }
    function renderBranches(ctx) {
      var box = $("d10-branch-list");
      box.innerHTML = "";
      (ctx.branches || []).forEach(function (b) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "d10-branch-btn" + (b.is_active ? " active" : "");
        btn.textContent = b.name + " (" + b.message_count + ")";
        btn.addEventListener("click", function () { activateBranch(b.id); });
        box.appendChild(btn);
      });
    }
    function renderUsage(usage) {
      $("d10-sys").textContent = "≈ " + fmt(usage.system_prompt_tokens_estimated);
      $("d10-facts-tokens").textContent = "≈ " + fmt(usage.facts_tokens_estimated);
      $("d10-recent-tokens").textContent = "≈ " + fmt(usage.recent_tokens_estimated);
      $("d10-cur").textContent = "≈ " + fmt(usage.current_user_tokens_estimated);
      $("d10-est-input").textContent = "≈ " + fmt(usage.estimated_input_tokens);
      $("d10-api-input").textContent = fmt(usage.prompt_tokens);
      $("d10-api-output").textContent = fmt(usage.completion_tokens);
      $("d10-api-total").textContent = fmt(usage.total_tokens);
      var ext = usage.facts_extraction_total_tokens;
      $("d10-facts-extract").textContent = ext === null || ext === undefined
        ? "не выполнялся"
        : fmt(ext) + " tokens";
    }
    function addCumulative(strategy, usage, extractionPerformed) {
      var c = state.cumulative[strategy];
      c.requests += 1;
      if (extractionPerformed) c.extractions += 1;
      c.input += usage.prompt_tokens || 0;
      c.output += usage.completion_tokens || 0;
      c.total += usage.total_tokens || 0;
      c.extract += usage.facts_extraction_total_tokens || 0;
      if (strategy === currentStrategy()) {
        $("d10-cum-requests").textContent = fmt(c.requests);
        $("d10-cum-extractions").textContent = fmt(c.extractions);
        $("d10-cum-input").textContent = fmt(c.input);
        $("d10-cum-output").textContent = fmt(c.output);
        $("d10-cum-total").textContent = fmt(c.total);
        $("d10-cum-extract").textContent = fmt(c.extract);
      }
      state.results[strategy].input = c.input;
      state.results[strategy].extra = c.extractions;
      renderCompare();
    }
    function renderCompare() {
      var tbody = $("d10-compare-rows");
      tbody.innerHTML = "";
      STRATEGIES.forEach(function (s) {
        var r = state.results[s];
        var tr = document.createElement("tr");
        [s, r.remembered, fmt(r.input), fmt(r.extra), r.stability, r.ux]
          .forEach(function (v) {
            var td = document.createElement("td");
            td.textContent = v;
            tr.appendChild(td);
          });
        tbody.appendChild(tr);
      });
    }

    /* ---------------- state loading ---------------- */
    function loadState() {
      var strategy = currentStrategy();
      var q = "?strategy=" + encodeURIComponent(strategy);
      var w = parseInt($("d10-window-input").value, 10);
      var rec = parseInt($("d10-recent-input").value, 10);
      if (Number.isInteger(w)) q += "&window_size=" + w;
      if (Number.isInteger(rec)) q += "&recent_messages_limit=" + rec;
      return http("/api/day10/state" + q).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Ошибка состояния.");
        $("d10-model").textContent = res.data.model;
        renderContext(res.data.context);
        renderMessages(res.data.context.sent_preview);
        renderCumulative(strategy);
      }).catch(function (err) { setError(err.message); });
    }
    function renderCumulative(strategy) {
      var c = state.cumulative[strategy];
      $("d10-cum-requests").textContent = fmt(c.requests);
      $("d10-cum-extractions").textContent = fmt(c.extractions);
      $("d10-cum-input").textContent = fmt(c.input);
      $("d10-cum-output").textContent = fmt(c.output);
      $("d10-cum-total").textContent = fmt(c.total);
      $("d10-cum-extract").textContent = fmt(c.extract);
    }

    /* ---------------- chat ---------------- */
    function sendMessage(strategy, text, opts) {
      opts = opts || {};
      var body = buildBody(strategy, text);
      return http("/api/day10/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
        var d = res.data;
        addCumulative(strategy, d.usage, d.facts_extraction_performed);
        if (!opts.silent) {
          addBubble("assistant", d.answer);
          renderUsage(d.usage);
          renderContext(d.context);
        }
        if (d.facts_error) {
          $("d10-facts-error").textContent = d.facts_error;
          $("d10-facts-error").classList.toggle("hidden", false);
          setNote(d.facts_error);
        } else if (strategy === "sticky_facts") {
          $("d10-facts-error").classList.add("hidden");
        }
        if (strategy === "branching") {
          state.results.branching.stability = "ветки изолированы";
          renderCompare();
        }
        return d;
      });
    }
    function onSend() {
      if (state.inFlight) return;
      var text = $("d10-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      var strategy = currentStrategy();
      setError(""); setNote("");
      addBubble("user", text);
      setLoading(true);
      sendMessage(strategy, text).then(function () {
        $("d10-input").value = "";
      }).catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- branch management ---------------- */
    function createBranch() {
      if (state.inFlight) return;
      setError("");
      var name = $("d10-branch-name").value.trim();
      var body = { strategy: "branching" };
      if (name) body.name = name;
      http("/api/day10/branches", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Не удалось создать ветку.");
        $("d10-branch-name").value = "";
        renderContext(res.data.context);
        loadState();
        setNote("Создана ветка «" + res.data.branch.name + "» от checkpoint.");
      }).catch(function (err) { setError(err.message); });
    }
    function activateBranch(branchId) {
      if (state.inFlight) return;
      http("/api/day10/branches/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: "branching", branch_id: branchId })
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Не удалось переключить ветку.");
        renderContext(res.data.context);
        renderMessages(res.data.context.sent_preview);
      }).catch(function (err) { setError(err.message); });
    }
    function seedBranches() {
      if (state.inFlight) return;
      setLoading(true);
      setError("");
      http("/api/day10/branches/seed", { method: "POST" }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Не удалось создать ветки.");
        state.seeded = true;
        $("d10-s-branching").checked = true;
        $("d10-model").textContent = res.data.model;
        renderContext(res.data.context);
        renderMessages(res.data.context.sent_preview);
        setNote("Созданы ветки PostgreSQL и MongoDB от общего checkpoint.");
      }).catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    /* ---------------- demo scenario ---------------- */
    function delay(ms) {
      return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }
    function runScenario(strategy, label) {
      var scenarioCache = null;
      return http("/api/day10/scenario").then(function (res) {
        if (!res.ok) throw new Error("Не удалось получить сценарий.");
        scenarioCache = res.data;
        return http("/api/day10/state?strategy=" + strategy, { method: "DELETE" });
      }).then(function () {
        state.cumulative[strategy] = {
          requests: 0, extractions: 0, input: 0, output: 0, total: 0, extract: 0
        };
        state.results[strategy].remembered = "—";
        var chain = Promise.resolve();
        var lastAnswer = "";
        scenarioCache.messages.forEach(function (text, idx) {
          chain = chain.then(function () {
            setProgress(label + ": " + (idx + 1) + "/" + scenarioCache.messages.length +
              " — " + text.slice(0, 60));
            var silent = strategy !== currentStrategy();
            if (!silent) {
              addBubble("user", text);
            }
            return sendMessage(strategy, text, { silent: silent }).then(function (d) {
              lastAnswer = d.answer;
              if (!silent) { addBubble("assistant", d.answer); }
            });
          });
        });
        return chain.then(function () {
          return http("/api/day10/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ answer: lastAnswer, expected: scenarioCache.expected })
          }).then(function (ev) {
            if (ev.ok) {
              state.results[strategy].remembered =
                ev.data.score + " / " + ev.data.total;
              state.results[strategy].stability = ev.data.score === ev.data.total
                ? "ключевые требования сохранены"
                : "часть ранних требований потеряна";
              renderCompare();
            }
          });
        });
      });
    }
    function onDemoRun() {
      if (state.inFlight) return;
      var strategy = currentStrategy();
      setLoading(true); setError(""); setNote("");
      setProgress("Запуск сценария…");
      runScenario(strategy, "Demo " + strategy)
        .then(function () { setNote("Сценарий завершён для стратегии " + strategy + "."); })
        .catch(function (err) { setError(err.message || "Сценарий не удалось выполнить."); })
        .finally(function () { setLoading(false); setProgress("—"); });
    }
    function onCompareRun() {
      if (state.inFlight) return;
      setLoading(true); setError(""); setNote("");
      setProgress("Сравнение: Sliding Window…");
      runScenario("sliding_window", "Sliding Window")
        .then(function () {
          setProgress("Сравнение: Sticky Facts…");
          return runScenario("sticky_facts", "Sticky Facts");
        })
        .then(function () {
          var a = state.results.sliding_window;
          var b = state.results.sticky_facts;
          setNote("Sliding: " + a.remembered + " · Sticky: " + b.remembered +
            ". Разница отражает потерю ранних требований скользящим окном.");
        })
        .catch(function (err) { setError(err.message || "Сравнение не удалось."); })
        .finally(function () { setLoading(false); setProgress("—"); });
    }

    /* ---------------- wiring ---------------- */
    $("d10-send").addEventListener("click", onSend);
    $("d10-demo-run").addEventListener("click", onDemoRun);
    $("d10-compare-run").addEventListener("click", onCompareRun);
    $("d10-seed-branches").addEventListener("click", seedBranches);
    $("d10-branch-create").addEventListener("click", createBranch);

    $("d10-clear").addEventListener("click", function () {
      if (state.inFlight) return;
      var strategy = currentStrategy();
      http("/api/day10/state?strategy=" + strategy, { method: "DELETE" }).then(function () {
        state.cumulative[strategy] = {
          requests: 0, extractions: 0, input: 0, output: 0, total: 0, extract: 0
        };
        state.results[strategy].remembered = "—";
        state.results[strategy].input = 0;
        state.results[strategy].extra = 0;
        state.results[strategy].stability = "—";
        renderCompare();
        renderMessages([]);
        setNote("");
        setError("");
        return loadState();
      });
    });

    ["d10-s-sliding", "d10-s-sticky", "d10-s-branching"].forEach(function (id) {
      $(id).addEventListener("change", function () {
        setError(""); setNote("");
        loadState();
      });
    });
    ["d10-window-input", "d10-recent-input"].forEach(function (id) {
      $(id).addEventListener("change", function () { loadState(); });
    });

    $("d10-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        $("d10-send").click();
      }
    });

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day10") {
        loadState();
      }
    });

    renderCompare();
    loadState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay10);
  } else {
    initDay10();
  }
})();
