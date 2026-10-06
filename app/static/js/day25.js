/* LLM Agent Lab — Day 25: Mini Chat with RAG + Task Memory.
 *
 * Vanilla JavaScript (no framework). Talks to:
 *   GET    /api/week5/day25/status
 *   GET    /api/week5/day25/sessions
 *   POST   /api/week5/day25/sessions
 *   GET    /api/week5/day25/sessions/{id}
 *   DELETE /api/week5/day25/sessions/{id}
 *   POST   /api/week5/day25/sessions/{id}/messages
 *   GET    /api/week5/day25/sessions/{id}/state
 *   GET    /api/week5/day25/evaluation/scenarios
 *   POST   /api/week5/day25/evaluation/run/{scenario_id}
 *   GET    /api/week5/day25/evaluation/results
 *
 * The backend owns sessions, task memory, retrieval and grounding. This file
 * only renders the structured responses and NEVER invents a source.
 */
(function () {
  "use strict";

  var API = "/api/week5/day25";

  function initDay25() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day25",
      "d25-tab-chat", "d25-tab-eval", "d25-view-chat", "d25-view-eval",
      "d25-status", "d25-new-chat", "d25-chat-list", "d25-messages",
      "d25-form", "d25-input", "d25-send", "d25-send-loading", "d25-error",
      "d25-task-state", "d25-tech", "d25-tech-body",
      "d25-eval-refresh", "d25-eval-results", "d25-eval-loading", "d25-eval-error",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day25: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      status: null,
      sessions: [],
      current: null,
      busy: false,
      evalLoaded: false,
      firstLoad: true,
      booted: false,
    };

    /* ------------------------------------------------------------------ */
    /* helpers                                                             */
    /* ------------------------------------------------------------------ */
    function show(el, on) { if (el) el.classList.toggle("hidden", !on); }
    function clear(el) { while (el && el.firstChild) el.removeChild(el.firstChild); }
    function setError(id, message) {
      var el = $(id);
      if (!el) return;
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }
    function make(tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    }
    async function api(method, path, body) {
      var options = { method: method, headers: {} };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      var response = await fetch(API + path, options);
      if (response.status === 204) return null;
      var data = null;
      try { data = await response.json(); } catch (err) { data = null; }
      if (!response.ok) {
        var detail = data && data.detail ? data.detail : ("HTTP " + response.status);
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    }

    /* ------------------------------------------------------------------ */
    /* section tabs                                                        */
    /* ------------------------------------------------------------------ */
    function switchSection(name) {
      var map = {
        chat: ["d25-tab-chat", "d25-view-chat"],
        eval: ["d25-tab-eval", "d25-view-eval"],
      };
      Object.keys(map).forEach(function (key) {
        var on = key === name;
        var btn = $(map[key][0]);
        var view = $(map[key][1]);
        if (btn) { btn.classList.toggle("active", on); btn.setAttribute("aria-selected", on ? "true" : "false"); }
        if (view) view.classList.toggle("hidden", !on);
      });
      if (name === "eval") loadEvaluation();
    }
    $("d25-tab-chat").addEventListener("click", function () { switchSection("chat"); });
    $("d25-tab-eval").addEventListener("click", function () { switchSection("eval"); });

    /* ------------------------------------------------------------------ */
    /* status                                                              */
    /* ------------------------------------------------------------------ */
    function renderStatus() {
      var box = $("d25-status");
      clear(box);
      var s = state.status;
      if (!s) { box.textContent = "—"; return; }
      [
        ["Generation", s.generation_provider + " / " + s.generation_model],
        ["Embedding", s.embedding_model],
        ["Index", s.index_path + " (" + s.chunks + " chunks)"],
        ["Recent messages", s.recent_messages],
        ["Chat DB", s.chat_db_path],
      ].forEach(function (row) {
        var line = make("div", "d24-status-row");
        line.appendChild(make("span", "d24-status-label", row[0]));
        line.appendChild(make("code", null, row[1]));
        box.appendChild(line);
      });
    }

    async function loadStatus() {
      try { state.status = await api("GET", "/status"); }
      catch (err) { state.status = null; }
      renderStatus();
    }

    /* ------------------------------------------------------------------ */
    /* chat list                                                           */
    /* ------------------------------------------------------------------ */
    function renderChatList() {
      var list = $("d25-chat-list");
      clear(list);
      if (!state.sessions.length) {
        list.appendChild(make("p", "field-hint", "Нет чатов. Создайте новый."));
        return;
      }
      state.sessions.forEach(function (session) {
        var item = make("div", "d25-chat-item");
        if (state.current && state.current.id === session.id) item.classList.add("active");
        var open = make("button", "d25-chat-open", session.title);
        open.type = "button";
        open.addEventListener("click", function () { selectSession(session.id); });
        item.appendChild(open);
        var del = make("button", "d25-chat-del", "×");
        del.type = "button";
        del.title = "Удалить чат";
        del.addEventListener("click", function (event) {
          event.stopPropagation();
          deleteSession(session.id);
        });
        item.appendChild(del);
        list.appendChild(item);
      });
    }

    async function loadSessions() {
      try {
        var data = await api("GET", "/sessions");
        state.sessions = (data && data.sessions) || [];
      } catch (err) {
        state.sessions = [];
      }
      renderChatList();
    }

    async function newChat() {
      try {
        var session = await api("POST", "/sessions", {});
        state.sessions.unshift(session);
        renderChatList();
        if (state.status && state.status.demo_opening_question) {
          $("d25-input").value = state.status.demo_opening_question;
        }
        await selectSession(session.id);
        $("d25-input").focus();
      } catch (err) {
        setError("d25-error", err.message);
      }
    }

    async function deleteSession(id) {
      try {
        await api("DELETE", "/sessions/" + id);
        state.sessions = state.sessions.filter(function (s) { return s.id !== id; });
        if (state.current && state.current.id === id) {
          state.current = null;
          clear($("d25-messages"));
          renderTaskState(null);
          renderTrace(null);
        }
        renderChatList();
      } catch (err) {
        setError("d25-error", err.message);
      }
    }

    /* ------------------------------------------------------------------ */
    /* session detail / messages                                           */
    /* ------------------------------------------------------------------ */
    async function selectSession(id) {
      setError("d25-error", "");
      try {
        var detail = await api("GET", "/sessions/" + id);
        state.current = detail.session;
        renderChatList();
        renderMessages(detail.messages || []);
        renderTaskState(detail.state);
        var lastAssistant = (detail.messages || []).filter(function (m) {
          return m.role === "assistant";
        }).pop();
        renderTrace(lastAssistant && lastAssistant.trace ? lastAssistant.trace : null);
      } catch (err) {
        setError("d25-error", err.message);
      }
    }

    function sourceTypeLabel(sourceType) {
      if (sourceType === "manual") return "MANUAL / TECHNICAL SOURCE";
      if (sourceType === "telegram") return "COMMUNITY / TELEGRAM";
      return (sourceType || "SOURCE").toUpperCase();
    }

    function renderEvidence(evidence) {
      var details = make("details", "d25-sources");
      details.appendChild(make(
        "summary", null, "Sources (" + evidence.length + ")"
      ));
      evidence.forEach(function (entry, index) {
        var card = make("article", "d24-evidence-item");
        if (entry.source_type === "manual") card.classList.add("d24-evidence-manual");
        if (entry.source_type === "telegram") card.classList.add("d24-evidence-telegram");

        var head = make("div", "d24-evidence-head");
        head.appendChild(make("span", "d24-evidence-index", "[" + (index + 1) + "]"));
        head.appendChild(make(
          "span",
          "d24-source-type d24-source-" + (entry.source_type || "other"),
          sourceTypeLabel(entry.source_type)
        ));
        card.appendChild(head);

        var meta = make("div", "d24-evidence-meta");
        function metaRow(label, value) {
          var line = make("div", "d24-meta-row");
          line.appendChild(make("span", "d24-meta-label", label));
          line.appendChild(make("span", "d24-meta-value", value));
          meta.appendChild(line);
        }
        if (entry.source) metaRow("Source", entry.source);
        if (entry.section) metaRow("Section", entry.section);
        if (entry.page !== null && entry.page !== undefined) metaRow("Page", entry.page);
        if (entry.message_ids && entry.message_ids.length) {
          metaRow("Messages", entry.message_ids.join(", "));
        }
        if (entry.date_from) {
          metaRow("Date", entry.date_to && entry.date_to !== entry.date_from
            ? entry.date_from + " — " + entry.date_to : entry.date_from);
        }
        metaRow("Chunk ID", entry.chunk_id);
        card.appendChild(meta);

        card.appendChild(make("blockquote", "d24-quote", entry.quote || "(пустая цитата)"));
        card.appendChild(make(
          "div",
          "d24-validation " + (entry.quote_valid ? "ok" : "err"),
          entry.quote_valid ? "quote verified: YES" : "quote verified: NO"
        ));

        var full = make("details", "d24-full-chunk");
        full.appendChild(make("summary", null, "Show full chunk"));
        full.appendChild(make("pre", "d24-full-chunk-text",
          entry.chunk_text || "(chunk text unavailable)"));
        card.appendChild(full);
        details.appendChild(card);
      });
      return details;
    }

    function renderMessage(message) {
      var article = make("article", "d25-message d25-" + message.role);
      article.appendChild(make("div", "d25-message-role",
        message.role === "user" ? "User" : "Assistant"));
      var body = make("div", "d25-message-body", message.content);
      article.appendChild(body);
      if (message.status && message.role === "assistant") {
        article.appendChild(make("div", "d25-message-status", "Status: " + message.status));
      }
      if (message.evidence && message.evidence.length) {
        article.appendChild(renderEvidence(message.evidence));
      }
      return article;
    }

    function renderMessages(messages) {
      var box = $("d25-messages");
      clear(box);
      if (!messages.length) {
        box.appendChild(make("p", "field-hint",
          "История пуста. Напишите первое сообщение."));
        return;
      }
      messages.forEach(function (message) { box.appendChild(renderMessage(message)); });
      box.scrollTop = box.scrollHeight;
    }

    /* ------------------------------------------------------------------ */
    /* task state panel                                                    */
    /* ------------------------------------------------------------------ */
    function renderTaskState(state) {
      var box = $("d25-task-state");
      clear(box);
      if (!state) {
        box.appendChild(make("p", "field-hint", "Task state появится после первого сообщения."));
        return;
      }
      function block(title, values, cls) {
        if (!values || !values.length) return;
        var section = make("div", "d25-state-block" + (cls ? " " + cls : ""));
        section.appendChild(make("h3", null, title));
        var ul = make("ul", "d25-state-list");
        values.forEach(function (value) { ul.appendChild(make("li", null, value)); });
        section.appendChild(ul);
        box.appendChild(section);
      }
      var goal = make("div", "d25-state-block");
      goal.appendChild(make("h3", null, "GOAL"));
      goal.appendChild(make("p", "d25-state-goal", state.goal || "(не определена)"));
      box.appendChild(goal);

      var v = state.vehicle || {};
      var vehicleLines = [v.model || "Mitsubishi Xpander"];
      if (v.year) vehicleLines.push("Year: " + v.year);
      if (v.engine) vehicleLines.push("Engine: " + v.engine);
      if (v.transmission) vehicleLines.push("Transmission: " + v.transmission);
      if (v.mileage_km !== null && v.mileage_km !== undefined) {
        vehicleLines.push("Mileage: " + v.mileage_km + " km");
      }
      block("VEHICLE", vehicleLines);
      block("KNOWN FACTS", (state.known_facts || []).map(function (f) { return f.text; }));
      block("CONSTRAINTS", state.constraints || []);
      block("TERMS", state.terms || []);
      block("CHECKS PERFORMED", (state.checks_performed || []).map(function (c) { return c.text; }));
      block("RESULTS", (state.results || []).map(function (r) { return r.text; }));
      block("HYPOTHESES (unconfirmed)", (state.hypotheses || []).map(function (h) { return h.text; }), "d25-hypotheses");
      block("OPEN QUESTIONS", state.open_questions || []);
    }

    /* ------------------------------------------------------------------ */
    /* technical details                                                   */
    /* ------------------------------------------------------------------ */
    function renderTrace(trace) {
      var body = $("d25-tech-body");
      clear(body);
      if (!trace) {
        body.appendChild(make("p", "field-hint", "Technical details появятся после ответа."));
        return;
      }
      function row(label, value) {
        var line = make("div", "d24-meta-row");
        line.appendChild(make("span", "d24-meta-label", label));
        line.appendChild(make("span", "d24-meta-value", value === null || value === undefined ? "—" : String(value)));
        body.appendChild(line);
      }
      row("Turn type", trace.turn_type);
      row("Current message", trace.current_message);
      row("Contextual query", trace.contextual_query);
      row("Rewritten query", trace.rewritten_query);
      row("Retrieval candidates", trace.retrieved_count);
      row("Passed filter", trace.accepted_count);
      row("Used", trace.used_count);
      row("Grounding", trace.grounding);
      row("Grounding status", trace.grounding_status);
      if (trace.task_state_error) row("Task state update", "FAILED: " + trace.task_state_error);
      if (trace.pipeline && trace.pipeline.length) {
        body.appendChild(make("h4", null, "Pipeline"));
        var pre = make("pre", "d22-pipeline", trace.pipeline.join("\n"));
        body.appendChild(pre);
      }
    }

    /* ------------------------------------------------------------------ */
    /* send message                                                        */
    /* ------------------------------------------------------------------ */
    async function sendMessage(event) {
      if (event) event.preventDefault();
      if (state.busy) return;
      var content = ($("d25-input").value || "").trim();
      if (!content) return;
      if (!state.current) {
        setError("d25-error", "Сначала создайте или выберите чат.");
        return;
      }
      state.busy = true;
      show($("d25-send-loading"), true);
      setError("d25-error", "");
      try {
        var response = await api(
          "POST", "/sessions/" + state.current.id + "/messages", { content: content }
        );
        $("d25-input").value = "";
        state.current = response.session;
        var box = $("d25-messages");
        if (box.querySelector(".field-hint")) clear(box);
        box.appendChild(renderMessage(response.user_message));
        box.appendChild(renderMessage(response.assistant_message));
        box.scrollTop = box.scrollHeight;
        renderTaskState(response.state);
        renderTrace(response.assistant_message.trace);
        await loadSessions();
      } catch (err) {
        setError("d25-error", err.message);
      } finally {
        state.busy = false;
        show($("d25-send-loading"), false);
      }
    }

    /* ------------------------------------------------------------------ */
    /* evaluation                                                          */
    /* ------------------------------------------------------------------ */
    function metricRow(label, value) {
      var line = make("div", "d24-meta-row");
      line.appendChild(make("span", "d24-meta-label", label));
      line.appendChild(make("span", "d24-meta-value", String(value)));
      return line;
    }

    function renderEvalResults(payload) {
      var box = $("d25-eval-results");
      clear(box);
      var runs = (payload && payload.runs) || {};
      var ids = Object.keys(runs);
      if (!ids.length) {
        box.appendChild(make("p", "field-hint",
          "Результатов пока нет. Запустите сценарий."));
      }
      ids.forEach(function (id) {
        var run = runs[id];
        var m = run.metrics || {};
        var card = make("section", "preview-panel");
        card.appendChild(make("h3", null, (run.scenario && run.scenario.title) || id));
        card.appendChild(metricRow("Turns", m.turns_completed + " / " + m.turns_total));
        card.appendChild(metricRow("Goal retained", m.goal_retained ? "PASS" : "FAIL"));
        card.appendChild(metricRow("Facts retained", m.memory_retention));
        card.appendChild(metricRow("Corrections applied",
          m.corrections_applied + " / " + m.corrections_expected));
        card.appendChild(metricRow("Contextual follow-ups",
          m.contextual_followups_resolved + " / " + m.contextual_followups));
        card.appendChild(metricRow("Grounded answers", m.grounded_answers));
        card.appendChild(metricRow("Answers with sources",
          m.answers_with_sources + " / " + m.grounded_answers));
        card.appendChild(metricRow("Quotes valid",
          m.quotes_valid + " / " + m.grounded_answers));
        card.appendChild(metricRow("Insufficient context", m.insufficient_context_turns));
        card.appendChild(metricRow("Correct refusals", m.correct_refusals));
        var runBtn = make("button", "primary-btn", "Run scenario");
        runBtn.type = "button";
        runBtn.addEventListener("click", function () { runScenario(id, runBtn); });
        card.appendChild(runBtn);
        box.appendChild(card);
      });
    }

    async function loadEvaluation() {
      try {
        var scenarios = await api("GET", "/evaluation/scenarios");
        var payload = await api("GET", "/evaluation/results");
        var runs = (payload && payload.runs) || {};
        // Show every scenario (even without a run yet) via a synthetic card.
        var combined = Object.assign({}, runs);
        (scenarios || []).forEach(function (scenario) {
          if (!combined[scenario.id]) {
            combined[scenario.id] = {
              scenario: scenario,
              metrics: { turns_total: scenario.turns.length },
            };
          }
        });
        renderEvalResults({ runs: combined });
        state.evalLoaded = true;
      } catch (err) {
        setError("d25-eval-error", err.message);
      }
    }

    async function runScenario(id, button) {
      if (button) button.disabled = true;
      show($("d25-eval-loading"), true);
      setError("d25-eval-error", "");
      try {
        await api("POST", "/evaluation/run/" + id);
        await loadEvaluation();
      } catch (err) {
        setError("d25-eval-error", err.message);
      } finally {
        if (button) button.disabled = false;
        show($("d25-eval-loading"), false);
      }
    }

    /* ------------------------------------------------------------------ */
    /* wiring                                                              */
    /* ------------------------------------------------------------------ */
    $("d25-new-chat").addEventListener("click", newChat);
    $("d25-form").addEventListener("submit", sendMessage);
    $("d25-eval-refresh").addEventListener("click", loadEvaluation);

    async function bootDay25() {
      if (state.booted) return;
      state.booted = true;
      await loadStatus();
      await loadSessions();
      if (state.sessions.length) {
        await selectSession(state.sessions[0].id);
      }
      state.firstLoad = false;
    }

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day25") bootDay25();
    });
    // If Day 25 is already the active tab at load time.
    var panel = $("panel-day25");
    if (panel && panel.style.display !== "none") bootDay25();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay25);
  } else {
    initDay25();
  }
})();
