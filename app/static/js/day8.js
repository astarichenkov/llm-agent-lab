/* LLM Agent Lab — Day 8: token accounting.
 * Vanilla JS. Talks to /api/day8/chat (real, accounted), /api/day8/estimate
 * (local only, no API spend) and /api/day8/history.
 *
 * Exact token counts come from the DeepSeek API `usage` block; per-part
 * numbers are local estimates and are shown with a "≈" marker.
 */
(function () {
  "use strict";

  var SHORT_DIALOG = [
    "Что такое токен? Ответь одним предложением.",
    "А чем токен отличается от слова?",
    "Спасибо, коротко подытожь."
  ];

  // Long scenario: 12 sizeable messages so the input grows visibly.
  var LONG_TOPIC = [
    "Подробно объясни, как работает HTTP-протокол, уровень за уровнем.",
    "Теперь опиши, как браузер устанавливает TLS-соединение с сервером.",
    "Разбери, что происходит при DNS-разрешении имени, по шагам.",
    "Объясни разницу между TCP и UDP с примерами использования.",
    "Опиши жизненный цикл HTTP-запроса в современном CDN.",
    "Расскажи про балансировку нагрузки и алгоритмы их выбора.",
    "Объясни, как работают таймауты и повторные попытки.",
    "Что такое идемпотентность и почему она важна для API?",
    "Опиши схемы аутентификации OAuth2 и JWT.",
    "Как устроено кэширование на уровне HTTP-заголовков?",
    "Объясни концепцию rate limiting и его стратегии.",
    "Сделай краткое резюме всего сказанного."
  ];

  function initDay8() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day8", "panel-day8", "d8-model", "d8-context-window",
      "d8-max-output", "d8-scenario-short", "d8-scenario-long",
      "d8-fill-context", "d8-clear", "d8-scenario-loading",
      "d8-messages", "d8-input", "d8-send", "d8-loading", "d8-error",
      "d8-sys", "d8-hist", "d8-cur", "d8-est-input", "d8-api-input",
      "d8-api-output", "d8-api-total", "d8-api-cache", "d8-cost",
      "d8-cum-requests", "d8-cum-messages", "d8-cum-history",
      "d8-cum-input", "d8-cum-output", "d8-cum-total", "d8-cum-cost",
      "d8-rows", "d8-totals", "d8-chart",
      "d8-ov-estimate", "d8-ov-limit", "d8-ov-excess", "d8-ov-status",
      "d8-overflow-send", "d8-ov-provider"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day8: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      rows: [],           // per-exchange stats
      cumInput: 0,
      cumOutput: 0,
      cumTotal: 0,
      cumCost: 0,
      historyTokens: 0,
      historyCount: 0,
      limits: null,
      overflowEstimate: null,
      inFlight: false
    };

    function setError(message) {
      $("d8-error").textContent = message || "";
      $("d8-error").classList.toggle("hidden", !message);
    }

    function setLoading(on) {
      state.inFlight = on;
      $("d8-send").disabled = on;
      $("d8-loading").classList.toggle("hidden", !on);
    }

    function fmt(n) {
      if (n === null || n === undefined) return "—";
      return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
    }

    function fmtCost(v) {
      if (v === null || v === undefined) return "—";
      return "$" + Number(v).toFixed(8);
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
      $("d8-messages").appendChild(bubble);
      $("d8-messages").scrollTop = $("d8-messages").scrollHeight;
    }

    function renderMessages(messages) {
      var el = $("d8-messages");
      el.innerHTML = "";
      (messages || []).forEach(function (m) { addBubble(m.role, m.content); });
    }

    function applyLimits(limits) {
      if (!limits) return;
      state.limits = limits;
      $("d8-context-window").textContent = fmt(limits.context_window) + " tokens";
      $("d8-max-output").textContent = fmt(limits.max_output_tokens) + " tokens";
      $("d8-ov-limit").textContent = fmt(limits.context_window);
    }

    /* ---------------- tables ---------------- */
    function renderBreakdown(data) {
      var u = data.usage || {};
      $("d8-sys").textContent = "≈ " + fmt(u.system_prompt_tokens_estimated);
      $("d8-hist").textContent = "≈ " + fmt(u.history_tokens_estimated);
      $("d8-cur").textContent = "≈ " + fmt(u.current_user_tokens_estimated);
      $("d8-est-input").textContent = "≈ " + fmt(u.estimated_input_tokens);
      $("d8-api-input").textContent = fmt(u.prompt_tokens);
      $("d8-api-output").textContent = fmt(u.completion_tokens);
      $("d8-api-total").textContent = fmt(u.total_tokens);
      var hit = u.prompt_cache_hit_tokens;
      var miss = u.prompt_cache_miss_tokens;
      $("d8-api-cache").textContent =
        (hit === null || hit === undefined) ? "—" : (fmt(hit) + " / " + fmt(miss));
      var c = data.cost || {};
      $("d8-cost").textContent =
        c.estimated ? (fmtCost(c.total) + " (оценка)") : "—";
    }

    function renderCumulative() {
      $("d8-cum-requests").textContent = String(state.rows.length);
      $("d8-cum-messages").textContent = String(state.historyCount);
      $("d8-cum-history").textContent = "≈ " + fmt(state.historyTokens);
      $("d8-cum-input").textContent = fmt(state.cumInput);
      $("d8-cum-output").textContent = fmt(state.cumOutput);
      $("d8-cum-total").textContent = fmt(state.cumTotal);
      $("d8-cum-cost").textContent = fmtCost(state.cumCost);
    }

    function renderTable() {
      var tbody = $("d8-rows");
      tbody.innerHTML = "";
      state.rows.forEach(function (r, i) {
        var tr = document.createElement("tr");
        [
          String(i + 1), "≈ " + fmt(r.userTokens), "≈ " + fmt(r.historyTokens),
          fmt(r.promptTokens), fmt(r.completionTokens), fmt(r.totalTokens),
          fmtCost(r.cost)
        ].forEach(function (val) {
          var td = document.createElement("td");
          td.textContent = val;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      var tfoot = $("d8-totals");
      tfoot.innerHTML = "";
      if (!state.rows.length) return;
      var tr = document.createElement("tr");
      [
        "Σ", "≈ " + fmt(state.cumInput), "", fmt(state.cumInput),
        fmt(state.cumOutput), fmt(state.cumTotal), fmtCost(state.cumCost)
      ].forEach(function (val) {
        var td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
      });
      tfoot.appendChild(tr);
    }

    /* ---------------- simple vanilla chart ---------------- */
    function renderChart() {
      var chart = $("d8-chart");
      chart.innerHTML = "";
      if (!state.rows.length) return;
      var W = 520, H = 140, PAD = 24;
      var maxInput = 1, maxTotal = 1;
      state.rows.forEach(function (r) {
        maxInput = Math.max(maxInput, r.promptTokens || 0);
        maxTotal = Math.max(maxTotal, r.totalTokens || 0);
      });
      var n = state.rows.length;
      function x(i) { return PAD + (n === 1 ? 0 : (W - 2 * PAD) * i / (n - 1)); }
      function y(val, max) { return H - PAD - (H - 2 * PAD) * (val || 0) / max; }
      var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", String(H));
      function polyline(getVal, max, color) {
        var points = state.rows.map(function (r, i) {
          return x(i) + "," + y(getVal(r), max);
        }).join(" ");
        var line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        line.setAttribute("points", points);
        line.setAttribute("fill", "none");
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", "2");
        svg.appendChild(line);
      }
      polyline(function (r) { return r.promptTokens; }, maxInput, "#2563eb");
      polyline(function (r) { return r.totalTokens; }, maxTotal, "#16a34a");
      chart.appendChild(svg);
    }

    /* ---------------- HTTP ---------------- */
    function recordExchange(data, userText) {
      var u = data.usage || {};
      var c = data.cost || {};
      state.rows.push({
        userTokens: u.current_user_tokens_estimated,
        historyTokens: u.history_tokens_estimated,
        promptTokens: u.prompt_tokens,
        completionTokens: u.completion_tokens,
        totalTokens: u.total_tokens,
        cost: c.total
      });
      state.cumInput += (u.prompt_tokens || 0);
      state.cumOutput += (u.completion_tokens || 0);
      state.cumTotal += (u.total_tokens || 0);
      state.cumCost += (c.total || 0);
      state.historyTokens = u.history_tokens_estimated;
      state.historyCount = (state.historyCount || 0) + 2;
      applyLimits(data.limits);
      renderBreakdown(data);
      renderCumulative();
      renderTable();
      renderChart();
    }

    function doChat(payloadText, opts) {
      opts = opts || {};
      var body = { message: payloadText };
      if (opts.history) body.history = opts.history;
      return fetch("/api/day8/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, status: r.status, data: d };
        });
      }).then(function (res) {
        if (!res.ok) {
          var err = new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
          err.status = res.status;
          throw err;
        }
        return res.data;
      });
    }

    function sendMessage() {
      if (state.inFlight) return;
      var text = $("d8-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      setError("");
      setLoading(true);
      addBubble("user", text);
      doChat(text).then(function (data) {
        addBubble("assistant", data.answer);
        recordExchange(data, text);
        $("d8-input").value = "";
      }).catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- scenarios ---------------- */
    function runSequence(messages, index) {
      if (index >= messages.length) {
        $("d8-scenario-loading").classList.add("hidden");
        return;
      }
      addBubble("user", messages[index]);
      doChat(messages[index]).then(function (data) {
        addBubble("assistant", data.answer);
        recordExchange(data, messages[index]);
        runSequence(messages, index + 1);
      }).catch(function (err) {
        setError(err.message || "Сценарий прерван.");
        $("d8-scenario-loading").classList.add("hidden");
      });
    }

    function resetUi() {
      state.rows = [];
      state.cumInput = state.cumOutput = state.cumTotal = state.cumCost = 0;
      state.historyTokens = 0;
      state.historyCount = 0;
      state.overflowEstimate = null;
      renderMessages([]);
      renderCumulative();
      renderTable();
      renderChart();
      $("d8-ov-provider").classList.add("hidden");
      $("d8-overflow-send").disabled = true;
      $("d8-ov-estimate").textContent = "—";
      $("d8-ov-excess").textContent = "—";
      $("d8-ov-status").textContent = "—";
    }

    function scenario(short) {
      if (state.inFlight) return;
      setError("");
      resetUi();
      fetch("/api/day8/history", { method: "DELETE" }).catch(function () {});
      $("d8-scenario-loading").classList.remove("hidden");
      runSequence(short ? SHORT_DIALOG : LONG_TOPIC, 0);
    }

    /* ---------------- context-overflow experiment ---------------- */
    // The oversized synthetic context is generated SERVER-SIDE: the browser
    // must not upload megabytes of filler (nginx caps the request body). No
    // provider call happens here — only a local estimate.
    function fillContext() {
      if (state.inFlight) return;
      setError("");
      if (!state.limits) {
        setError("Лимиты модели ещё не загружены.");
        return;
      }
      $("d8-scenario-loading").classList.remove("hidden");
      fetch("/api/day8/overflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ overshoot_factor: 1.15 })
      }).then(function (r) { return r.json(); }).then(function (data) {
        state.overflowEstimate = { request: data.request, estimate: data };
        applyLimits(data.limits);
        $("d8-ov-estimate").textContent =
          "≈ " + fmt(data.usage.estimated_input_tokens) +
          " (+ " + fmt(data.limits.max_output_tokens) + " output budget = " +
          fmt(data.effective_input_tokens) + ")";
        $("d8-ov-limit").textContent = fmt(data.limits.context_window);
        $("d8-ov-excess").textContent = "≈ +" + fmt(data.overflow_tokens);
        $("d8-ov-status").textContent =
          data.exceeds_context ? "LIMIT EXCEEDED" : "within limit";
        $("d8-overflow-send").disabled = !data.exceeds_context;
        $("d8-scenario-loading").classList.add("hidden");
      }).catch(function (err) {
        setError(err.message || "Не удалось оценить контекст.");
        $("d8-scenario-loading").classList.add("hidden");
      });
    }

    function runOverflowRequest() {
      if (state.inFlight || !state.overflowEstimate) return;
      if (!window.confirm("Выполнить реальный запрос с превышением контекста? API вернёт ошибку.")) {
        return;
      }
      setError("");
      setLoading(true);
      $("d8-ov-provider").classList.add("hidden");
      fetch("/api/day8/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.overflowEstimate.request)
      }).then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; });
      }).then(function (res) {
        var est = state.overflowEstimate.estimate;
        var lines = [
          "HTTP " + res.status,
          "",
          res.ok ? "Неожиданно: запрос принят." : "DeepSeek API rejected the request.",
          "",
          "Context limit: " + fmt(est.limits.context_window),
          "Estimated request: " + fmt(est.effective_input_tokens) +
            " (messages ≈" + fmt(est.usage.estimated_input_tokens) +
            " + output budget " + fmt(est.limits.max_output_tokens) + ")",
          "Exceeded by: ≈" + fmt(est.overflow_tokens) + " tokens",
          "",
          "Provider message:",
          String(res.data.detail || "")
        ];
        $("d8-ov-provider").textContent = lines.join("\n");
        $("d8-ov-provider").classList.remove("hidden");
        $("d8-ov-status").textContent = res.ok ? "ACCEPTED" : "REJECTED (HTTP " + res.status + ")";
      }).catch(function (err) {
        setError(err.message || "Ошибка запроса переполнения.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- wiring ---------------- */
    $("d8-send").addEventListener("click", sendMessage);
    $("d8-scenario-short").addEventListener("click", function () { scenario(true); });
    $("d8-scenario-long").addEventListener("click", function () { scenario(false); });
    $("d8-fill-context").addEventListener("click", fillContext);
    $("d8-overflow-send").addEventListener("click", runOverflowRequest);
    $("d8-clear").addEventListener("click", function () {
      if (state.inFlight) return;
      fetch("/api/day8/history", { method: "DELETE" }).catch(function () {});
      resetUi();
      setError("");
    });
    $("d8-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendMessage();
      }
    });

    // Load the model limits from the backend (never hardcoded in JS).
    fetch("/api/day8/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "" })
    }).then(function (r) { return r.json(); }).then(function (data) {
      $("d8-model").textContent = data.model || "—";
      applyLimits(data.limits);
    }).catch(function () { /* keep placeholders */ });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay8);
  } else {
    initDay8();
  }
})();
