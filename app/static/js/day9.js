/* LLM Agent Lab — Day 9: history compression.
 * Vanilla JS. Talks to /api/day9/chat, /api/day9/seed, /api/day9/diagnostics
 * and DELETE /api/day9/history. All compression happens on the backend.
 *
 * Transparency: the dialog shows exactly what the agent REMEMBERS. A memory
 * card at the top renders the current summary, every bubble is coloured by
 * zone (already summarized / outside the window / recent window), and each
 * compression cycle adds a system event message to the dialog.
 */
(function () {
  "use strict";

  var CONTROL_QUESTION =
    "Какие ключевые архитектурные решения и ограничения мы выбрали в начале разговора?";

  var ZONE_LABELS = {
    summarized: "Свёрнуто в summary — модель видит только память агента",
    pending: "Старше recent-window и ещё не сжато — в текущий запрос НЕ попадает",
    recent: "Recent window — передаётся в модель дословно"
  };

  function initDay9() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day9", "panel-day9", "d9-model", "d9-recent", "d9-batch", "d9-cycles",
      "d9-mode-full", "d9-mode-compressed", "d9-recent-input", "d9-batch-input",
      "d9-seed", "d9-ask-control", "d9-clear", "d9-loading",
      "d9-messages", "d9-input", "d9-send", "d9-error", "d9-compression-note",
      "d9-full-count", "d9-summarized-count", "d9-recent-count", "d9-pending-count",
      "d9-full-tokens", "d9-summary-tokens", "d9-recent-tokens",
      "d9-compressed-tokens", "d9-saved", "d9-saved-percent", "d9-cycles-diag",
      "d9-summary-details", "d9-summary-text",
      "d9-rows", "d9-chart",
      "d9-answer-full", "d9-answer-compressed"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day9: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      rows: [],
      inFlight: false,
      // Transparency state: the dialog + what the backend remembers about it.
      messages: [],
      summarizedCount: 0,
      recentCount: 0,
      summary: null,
      events: []
    };

    function setError(m) {
      $("d9-error").textContent = m || "";
      $("d9-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d9-compression-note").textContent = m || "";
      $("d9-compression-note").classList.toggle("hidden", !m);
    }
    function setLoading(on) {
      state.inFlight = on;
      $("d9-send").disabled = on;
      $("d9-loading").classList.toggle("hidden", !on);
    }
    function fmt(n) {
      if (n === null || n === undefined) return "—";
      return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
    }
    function mode() {
      return $("d9-mode-compressed").checked ? "compressed" : "full";
    }
    function reqOptions() {
      var body = { mode: mode() };
      var r = parseInt($("d9-recent-input").value, 10);
      var b = parseInt($("d9-batch-input").value, 10);
      if (r) body.recent_messages_limit = r;
      if (b) body.compression_batch_size = b;
      return body;
    }

    /* ---------------- dialog rendering (transparent) ---------------- */
    function makeBubble(role, content) {
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
      return bubble;
    }

    function makeZoneLabel(zone) {
      var el = document.createElement("div");
      el.className = "d9-zone-label d9-zone-label-" + zone;
      el.textContent = "▼ " + ZONE_LABELS[zone];
      return el;
    }

    function makeSystemEvent(text) {
      var el = document.createElement("div");
      el.className = "d9-system-bubble";
      el.textContent = text;
      return el;
    }

    function zoneOf(index) {
      var summarized = Math.min(state.summarizedCount, state.messages.length);
      var recentStart = Math.max(state.messages.length - state.recentCount, summarized);
      if (index < summarized) return "summarized";
      if (index < recentStart) return "pending";
      return "recent";
    }

    function renderChat() {
      var el = $("d9-messages");
      if (!el) return;
      el.innerHTML = "";

      // 1) Persistent "what the agent remembers" card, right in the dialog.
      var card = document.createElement("div");
      card.className = "d9-memory-card";
      var title = document.createElement("div");
      title.className = "d9-memory-title";
      title.textContent = "🧠 Память агента (summary)";
      var body = document.createElement("div");
      body.className = "d9-memory-body";
      body.textContent = state.summary
        ? state.summary
        : "Пока пусто: summary ещё не построен. Он появится, когда старые " +
          "сообщения уйдут за recent-window и сработает сжатие.";
      card.appendChild(title);
      card.appendChild(body);
      el.appendChild(card);

      // 2) The dialog itself, grouped by zone.
      var total = state.messages.length;
      if (total === 0) {
        var empty = document.createElement("div");
        empty.className = "d9-empty";
        empty.textContent = "Диалог пуст. Нажмите «Заполнить тестовый длинный диалог».";
        el.appendChild(empty);
      }
      var lastZone = null;
      state.messages.forEach(function (m, i) {
        var zone = zoneOf(i);
        if (zone !== lastZone) {
          el.appendChild(makeZoneLabel(zone));
          lastZone = zone;
        }
        var bubble = makeBubble(m.role, m.content);
        bubble.classList.add("d9-zone-" + zone);
        el.appendChild(bubble);
      });

      // 3) System events (compression cycles) at the end.
      state.events.forEach(function (ev) {
        el.appendChild(makeSystemEvent(ev.text));
      });

      el.scrollTop = el.scrollHeight;
    }

    function renderMessages(messages) {
      state.messages = (messages || []).map(function (m) {
        return { role: m.role, content: m.content };
      });
      renderChat();
    }

    /* ---------------- diagnostics ---------------- */
    function renderDiagnostics(d) {
      if (!d) return;
      state.summarizedCount = d.summarized_messages || 0;
      state.recentCount = d.recent_messages_sent || 0;
      state.summary = d.summary || null;
      $("d9-full-count").textContent = fmt(d.full_history_messages);
      $("d9-summarized-count").textContent = fmt(d.summarized_messages);
      $("d9-recent-count").textContent = fmt(d.recent_messages_sent);
      $("d9-pending-count").textContent = fmt(d.pending_messages);
      $("d9-full-tokens").textContent = "≈ " + fmt(d.full_history_tokens_estimated);
      $("d9-summary-tokens").textContent = "≈ " + fmt(d.summary_tokens_estimated);
      $("d9-recent-tokens").textContent = "≈ " + fmt(d.recent_tokens_estimated);
      $("d9-compressed-tokens").textContent =
        "≈ " + fmt(d.compressed_context_tokens_estimated);
      $("d9-saved").textContent = "≈ " + fmt(d.tokens_saved);
      $("d9-saved-percent").textContent = (d.tokens_saved_percent || 0) + "%";
      $("d9-cycles").textContent = fmt(d.compression_cycles);
      $("d9-cycles-diag").textContent = fmt(d.compression_cycles);
      $("d9-recent").textContent = fmt(d.recent_messages_limit) + " сообщений";
      $("d9-batch").textContent = fmt(d.compression_batch_size) + " сообщений";
      $("d9-summary-text").textContent =
        d.summary || "(summary ещё не построен)";
    }

    function renderTable() {
      var tbody = $("d9-rows");
      tbody.innerHTML = "";
      state.rows.forEach(function (r) {
        var tr = document.createElement("tr");
        [
          String(r.step), r.mode, fmt(r.full), fmt(r.compressed),
          fmt(r.saved), (r.savedPct || 0) + "%"
        ].forEach(function (v) {
          var td = document.createElement("td");
          td.textContent = v;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    }

    function renderChart() {
      var chart = $("d9-chart");
      chart.innerHTML = "";
      if (!state.rows.length) return;
      var W = 520, H = 150, PAD = 26;
      var maxV = 1;
      state.rows.forEach(function (r) {
        maxV = Math.max(maxV, r.full || 0, r.compressed || 0);
      });
      var n = state.rows.length;
      function x(i) { return PAD + (n === 1 ? 0 : (W - 2 * PAD) * i / (n - 1)); }
      function y(v) { return H - PAD - (H - 2 * PAD) * (v || 0) / maxV; }
      var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", String(H));
      function line(key, color) {
        var pts = state.rows.map(function (r, i) {
          return x(i) + "," + y(r[key]);
        }).join(" ");
        var p = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        p.setAttribute("points", pts);
        p.setAttribute("fill", "none");
        p.setAttribute("stroke", color);
        p.setAttribute("stroke-width", "2");
        svg.appendChild(p);
      }
      line("full", "#dc2626");
      line("compressed", "#16a34a");
      chart.appendChild(svg);
    }

    function http(path, opts) {
      return fetch(path, opts).then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; });
      });
    }

    function send(text) {
      setError("");
      state.messages.push({ role: "user", content: text });
      renderChat();
      setLoading(true);
      var body = reqOptions();
      body.message = text;
      return http("/api/day9/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
        var d = res.data;
        state.messages.push({ role: "assistant", content: d.answer });
        renderDiagnostics(d.diagnostics);

        if (d.compression_performed) {
          var msg = "🧠 Сжатие #" + d.diagnostics.compression_cycles +
            ": старые сообщения свёрнуты в память агента. " +
            "В summary теперь " + fmt(d.diagnostics.summarized_messages) +
            " сообщений. Экономия ≈" + fmt(d.usage.tokens_saved) +
            " токенов (" + (d.usage.tokens_saved_percent || 0) + "%).";
          state.events.push({ text: msg });
          setNote(msg);
        } else if (d.compression_error) {
          setNote("Сжатие не удалось: " + d.compression_error + ". История не изменена.");
        } else {
          var pending = d.diagnostics.pending_messages;
          setNote(pending > 0
            ? "Сжатие пока не запускалось: " + fmt(pending) +
              " старых сообщений ждут накопления batch. Модель их пока не видит."
            : "Сжатие не требовалось.");
        }
        renderChat();

        state.rows.push({
          step: state.rows.length + 1,
          mode: d.mode === "full" ? "без сжатия" : "со сжатием",
          full: d.usage.full_context_tokens_estimated,
          compressed: d.usage.compressed_context_tokens_estimated,
          saved: d.usage.tokens_saved,
          savedPct: d.usage.tokens_saved_percent
        });
        renderTable();
        renderChart();
        if (d.mode === "full") $("d9-answer-full").textContent = d.answer;
        else $("d9-answer-compressed").textContent = d.answer;
        $("d9-model").textContent = d.model;
        return d;
      }).catch(function (err) {
        // Roll back the optimistic user bubble so the dialog stays truthful.
        state.messages.pop();
        renderChat();
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    function loadDiagnostics() {
      return http("/api/day9/diagnostics").then(function (res) {
        if (res.ok) {
          if (res.data.model) $("d9-model").textContent = res.data.model;
          renderDiagnostics(res.data.diagnostics);
          renderChat();
        }
      }).catch(function () {});
    }

    /* ---------------- wiring ---------------- */
    $("d9-send").addEventListener("click", function () {
      if (state.inFlight) return;
      var text = $("d9-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      send(text).then(function () { $("d9-input").value = ""; });
    });

    $("d9-seed").addEventListener("click", function () {
      if (state.inFlight) return;
      setError("");
      setNote("Создан тестовый диалог: факты заявлены в начале, затем идут " +
        "сообщения, вытесняющие их за recent-window.");
      setLoading(true);
      var body = {};
      var r = parseInt($("d9-recent-input").value, 10);
      var b = parseInt($("d9-batch-input").value, 10);
      if (r) body.recent_messages_limit = r;
      if (b) body.compression_batch_size = b;
      http("/api/day9/seed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Ошибка сценария.");
        state.events = [];
        renderMessages(res.data.messages);
        renderDiagnostics(res.data.diagnostics);
        renderChat();
        state.rows = [];
        renderTable();
        renderChart();
        $("d9-answer-full").textContent = "—";
        $("d9-answer-compressed").textContent = "—";
      }).catch(function (err) {
        setError(err.message || "Не удалось заполнить диалог.");
      }).finally(function () { setLoading(false); });
    });

    $("d9-ask-control").addEventListener("click", function () {
      if (state.inFlight) return;
      send(CONTROL_QUESTION);
    });

    $("d9-clear").addEventListener("click", function () {
      if (state.inFlight) return;
      http("/api/day9/history", { method: "DELETE" }).then(function () {
        state.messages = [];
        state.events = [];
        state.summarizedCount = 0;
        state.recentCount = 0;
        state.summary = null;
        renderChat();
        state.rows = [];
        renderTable();
        renderChart();
        $("d9-answer-full").textContent = "—";
        $("d9-answer-compressed").textContent = "—";
        setNote("");
        setError("");
        return loadDiagnostics();
      });
    });

    ["d9-mode-full", "d9-mode-compressed"].forEach(function (id) {
      $(id).addEventListener("change", function () { return loadDiagnostics(); });
    });
    ["d9-recent-input", "d9-batch-input"].forEach(function (id) {
      $(id).addEventListener("change", function () { return loadDiagnostics(); });
    });

    $("d9-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        $("d9-send").click();
      }
    });

    loadDiagnostics();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay9);
  } else {
    initDay9();
  }
})();
