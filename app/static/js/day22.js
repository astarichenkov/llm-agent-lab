/* LLM Agent Lab — Day 22: first RAG request (No-RAG vs RAG).
 *
 * Vanilla JavaScript (no framework), matching the rest of the project.
 * Talks to:
 *   GET  /api/week5/day22/status
 *   POST /api/week5/day22/ask
 *   POST /api/week5/day22/compare
 *   GET  /api/week5/day22/evaluation/questions
 *   POST /api/week5/day22/evaluation/run/{id}
 *   POST /api/week5/day22/evaluation/run-all
 *   GET  /api/week5/day22/evaluation/results
 *   PUT  /api/week5/day22/evaluation/result/{id}
 *
 * The backend owns retrieval, embeddings and generation. This file only
 * renders what it returned; it never touches SQLite or computes embeddings.
 */
(function () {
  "use strict";

  var API = "/api/week5/day22";

  function initDay22() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day22",
      "d22-tab-ask", "d22-tab-eval", "d22-view-ask", "d22-view-eval",
      "d22-status", "d22-question", "d22-mode-no-rag", "d22-mode-rag",
      "d22-top-k", "d22-ask", "d22-compare", "d22-ask-loading", "d22-ask-error",
      "d22-pipeline-panel", "d22-pipeline", "d22-results",
      "d22-eval-summary", "d22-eval-run-all", "d22-eval-refresh",
      "d22-eval-loading", "d22-eval-error", "d22-eval-table-body",
      "d22-eval-detail",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day22: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      statusLoaded: false,
      questionsLoaded: false,
      questions: [],
      results: {},
      summary: null,
      currentId: null,
      busy: false,
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
    function formatSim(value) {
      var n = Number(value);
      return isNaN(n) ? String(value) : n.toFixed(4);
    }
    function asList(value) {
      if (value === undefined || value === null) return [];
      return Array.isArray(value) ? value : [value];
    }
    function previewText(text, limit) {
      var t = (text || "").replace(/\s+/g, " ").trim();
      var max = limit || 220;
      return t.length > max ? t.slice(0, max) + "…" : t;
    }
    async function api(method, path, body) {
      var options = { method: method, headers: {} };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      var response = await fetch(API + path, options);
      var data = null;
      try { data = await response.json(); } catch (err) { data = null; }
      if (!response.ok) {
        var detail = data && data.detail ? data.detail : ("HTTP " + response.status);
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    }

    /* ------------------------------------------------------------------ */
    /* status                                                              */
    /* ------------------------------------------------------------------ */
    function renderStatus(status) {
      var box = $("d22-status");
      clear(box);
      box.appendChild(make("span", null, "Embedding model: " + (status.embedding_model || "—")));
      box.appendChild(make("span", null, "Generation model: " + (status.generation_model || "—")));
      box.appendChild(make("span", null, "Provider: " + (status.generation_provider || "—")));
      box.appendChild(make("span", null, "Temperature: " + status.temperature));
      box.appendChild(make("span", null, "Index: " + (status.chunking || "—") +
        " (" + status.chunks + " chunks)"));
      box.appendChild(make("span", null, "Default Top-K: " + status.default_top_k));
    }

    async function loadStatus() {
      if (state.statusLoaded) return;
      try {
        renderStatus(await api("GET", "/status"));
        state.statusLoaded = true;
      } catch (err) {
        $("d22-status").textContent = "Не удалось получить статус: " + err.message;
      }
    }

    /* ------------------------------------------------------------------ */
    /* answer rendering                                                    */
    /* ------------------------------------------------------------------ */
    function answerMeta(answer) {
      var meta = make("div", "d22-answer-meta");
      meta.appendChild(make("span", null, "Model: " + answer.model));
      meta.appendChild(make("span", null, "Mode: " + (answer.mode === "rag" ? "RAG" : "No RAG")));
      meta.appendChild(make("span", null, "RAG context: " +
        (answer.retrieval_performed ? "enabled" : "disabled")));
      meta.appendChild(make("span", null, "Top-K: " + answer.top_k));
      return meta;
    }

    function renderAnswerCard(title, answer, kind) {
      var card = make("article", "d22-answer-card d22-answer-" + kind);
      card.appendChild(make("h3", null, title));
      card.appendChild(answerMeta(answer));
      var body = make("div", "d17-answer");
      body.textContent = answer.answer || "(пустой ответ)";
      card.appendChild(body);
      return card;
    }

    function renderChunkCard(chunk) {
      var card = make("article", "d22-chunk");
      var head = make("div", "d22-chunk-head");
      head.appendChild(make("span", "d22-rank", "#" + chunk.rank));
      head.appendChild(make("span", "d22-similarity", "Similarity: " + formatSim(chunk.similarity)));
      head.appendChild(make("span", "d22-source-type d22-source-" + chunk.source_type,
        "Source type: " + chunk.source_type));
      card.appendChild(head);

      var meta = make("div", "d22-chunk-meta");
      if (chunk.source) meta.appendChild(make("span", null, "Source: " + chunk.source));
      var m = chunk.metadata || {};
      if (chunk.source_type === "manual") {
        if (m.page !== undefined && m.page !== null) {
          meta.appendChild(make("span", null, "Page: " + m.page));
        } else if (m.page_from !== undefined && m.page_from !== null) {
          meta.appendChild(make("span", null, "Pages: " + m.page_from +
            (m.page_to && m.page_to !== m.page_from ? "–" + m.page_to : "")));
        }
        if (m.section) meta.appendChild(make("span", null, "Section: " + m.section));
      } else if (chunk.source_type === "telegram") {
        if (m.chat_name) meta.appendChild(make("span", null, "Chat: " + m.chat_name));
        if (m.date_from) meta.appendChild(make("span", null, "Date: " + m.date_from));
        if (m.message_ids && m.message_ids.length) {
          meta.appendChild(make("span", null, "Message IDs: " + asList(m.message_ids).join(", ")));
        }
        if (m.authors && m.authors.length) {
          meta.appendChild(make("span", null, "Authors: " + asList(m.authors).join(", ")));
        }
      }
      meta.appendChild(make("span", null, "Chunk ID: " + chunk.chunk_id));
      card.appendChild(meta);

      card.appendChild(make("p", "d22-chunk-text", previewText(chunk.text)));

      var full = make("p", "d22-chunk-text d22-chunk-full hidden");
      full.textContent = chunk.text || "";
      card.appendChild(full);

      var toggle = make("button", "link-btn", "Show full chunk");
      toggle.type = "button";
      toggle.addEventListener("click", function () {
        var hidden = full.classList.toggle("hidden");
        toggle.textContent = hidden ? "Show full chunk" : "Hide full chunk";
      });
      card.appendChild(toggle);
      return card;
    }

    function renderChunks(chunks, heading) {
      var section = make("article", "d22-context");
      section.appendChild(make("h3", null, heading || "Retrieved Context"));
      if (!chunks || !chunks.length) {
        section.appendChild(make("p", "field-hint", "Релевантные фрагменты не найдены."));
        return section;
      }
      chunks.forEach(function (chunk) { section.appendChild(renderChunkCard(chunk)); });
      return section;
    }

    function renderPipeline(stages) {
      $("d22-pipeline").textContent = (stages || []).join("\n   ↓\n");
      show($("d22-pipeline-panel"), true);
    }

    function renderSingle(answer) {
      var box = $("d22-results");
      clear(box);
      var title = answer.mode === "rag" ? "RAG ANSWER" : "NO RAG ANSWER";
      box.appendChild(renderAnswerCard(title, answer, answer.mode));
      if (answer.mode === "rag") {
        box.appendChild(renderChunks(answer.chunks, "Retrieved Context (Top-" + answer.top_k + ")"));
      }
      renderPipeline(answer.pipeline);
    }

    function renderComparison(comparison) {
      var box = $("d22-results");
      clear(box);
      var q = make("h2", null, "QUESTION");
      box.appendChild(q);
      box.appendChild(make("p", "d22-question-echo", comparison.question));

      var grid = make("div", "d22-answer-grid");
      grid.appendChild(renderAnswerCard("WITHOUT RAG", comparison.no_rag, "no_rag"));
      grid.appendChild(renderAnswerCard("WITH RAG", comparison.rag, "rag"));
      box.appendChild(grid);

      box.appendChild(renderChunks(comparison.rag.chunks, "RETRIEVED CONTEXT"));
      renderPipeline(comparison.rag.pipeline);
    }

    /* ------------------------------------------------------------------ */
    /* ask / compare                                                       */
    /* ------------------------------------------------------------------ */
    function selectedMode() {
      return $("d22-mode-rag").checked ? "rag" : "no_rag";
    }
    function selectedTopK() {
      var value = parseInt($("d22-top-k").value, 10);
      return isNaN(value) ? 5 : value;
    }
    function setBusy(on, label) {
      state.busy = on;
      $("d22-ask").disabled = on;
      $("d22-compare").disabled = on;
      var loading = $("d22-ask-loading");
      if (on) loading.textContent = label || "Searching knowledge base…";
      show(loading, on);
    }

    async function ask() {
      if (state.busy) return;
      var question = $("d22-question").value.trim();
      if (!question) { setError("d22-ask-error", "Введите вопрос."); return; }
      setError("d22-ask-error", "");
      setBusy(true, selectedMode() === "rag"
        ? "Searching knowledge base…" : "Generating answer…");
      try {
        var answer = await api("POST", "/ask", {
          question: question,
          mode: selectedMode(),
          top_k: selectedTopK(),
        });
        renderSingle(answer);
      } catch (err) {
        setError("d22-ask-error", err.message);
      } finally {
        setBusy(false);
      }
    }

    async function compare() {
      if (state.busy) return;
      var question = $("d22-question").value.trim();
      if (!question) { setError("d22-ask-error", "Введите вопрос."); return; }
      setError("d22-ask-error", "");
      setBusy(true, "Searching knowledge base…");
      try {
        var comparison = await api("POST", "/compare", {
          question: question,
          top_k: selectedTopK(),
        });
        renderComparison(comparison);
      } catch (err) {
        setError("d22-ask-error", err.message);
      } finally {
        setBusy(false);
      }
    }

    /* ------------------------------------------------------------------ */
    /* evaluation                                                          */
    /* ------------------------------------------------------------------ */
    function expectedSourceLabel(source) {
      var parts = [source.source_type];
      if (source.source) parts.push(source.source);
      if (source.page !== undefined && source.page !== null) parts.push("p." + source.page);
      if (source.message_ids && source.message_ids.length) {
        parts.push("ids " + source.message_ids.slice(0, 4).join(",") + "…");
      }
      return parts.join(" · ");
    }

    function renderQuestionsTable() {
      var body = $("d22-eval-table-body");
      clear(body);
      state.questions.forEach(function (question) {
        var tr = document.createElement("tr");
        tr.appendChild(make("td", "d22-eval-id", question.id));

        var questionCell = make("td", "d22-eval-question");
        questionCell.textContent = question.question;
        tr.appendChild(questionCell);

        tr.appendChild(make("td", null, question.category));

        var sources = make("td", "d22-eval-sources");
        (question.expected_sources || []).forEach(function (source) {
          sources.appendChild(make("span", "d22-expected-source", expectedSourceLabel(source)));
        });
        tr.appendChild(sources);

        var actions = make("td", "d22-eval-actions");
        var run = make("button", "link-btn", "Run");
        run.type = "button";
        run.addEventListener("click", function () { runQuestion(question.id); });
        actions.appendChild(run);
        var runCompare = make("button", "link-btn", "Run Compare");
        runCompare.type = "button";
        runCompare.addEventListener("click", function () { runQuestion(question.id); });
        actions.appendChild(runCompare);
        tr.appendChild(actions);

        body.appendChild(tr);
      });
    }

    function renderSummary() {
      var box = $("d22-eval-summary");
      clear(box);
      var summary = state.summary;
      if (!summary) { box.textContent = "No data"; return; }

      function block(title, counts) {
        var wrap = make("div", "d22-summary-block");
        wrap.appendChild(make("b", null, title));
        ["pass", "partial", "fail"].forEach(function (grade) {
          var n = (counts && counts[grade]) || 0;
          wrap.appendChild(make("span", "d22-summary-item",
            grade.toUpperCase() + ": " + n + " / " + summary.total_questions));
        });
        return wrap;
      }
      box.appendChild(block("NO RAG", summary.no_rag));
      box.appendChild(block("RAG", summary.rag));

      var sourceWrap = make("div", "d22-summary-block");
      sourceWrap.appendChild(make("b", null, "EXPECTED SOURCE RETRIEVED"));
      sourceWrap.appendChild(make("span", "d22-summary-item",
        (summary.expected_source_retrieved || 0) + " / " + summary.total_questions));
      box.appendChild(sourceWrap);
      box.appendChild(make("div", "field-hint",
        "Оценено вопросов: " + summary.evaluated + " / " + summary.total_questions));
    }

    function recordFor(id) {
      var record = state.results[id] || {};
      return {
        no_rag: record.no_rag || null,
        rag: record.rag || null,
        expected_source_retrieved:
          record.expected_source_retrieved === undefined ? null : record.expected_source_retrieved,
      };
    }

    async function saveGrade(id, patch) {
      var record = recordFor(id);
      Object.keys(patch).forEach(function (key) { record[key] = patch[key]; });
      var response = await api("PUT", "/evaluation/result/" + encodeURIComponent(id), record);
      state.results = response.results || {};
      state.summary = response.summary;
      renderSummary();
      renderDetail(state.currentId);
    }

    function gradeButtons(label, value, gradeKey) {
      var wrap = make("div", "d22-grade");
      wrap.appendChild(make("span", "d22-grade-label", label + ":"));
      ["pass", "partial", "fail"].forEach(function (grade) {
        var btn = make("button", "link-btn" + (value === grade ? " active" : ""),
          grade.toUpperCase());
        btn.type = "button";
        btn.addEventListener("click", function () {
          var patch = {};
          patch[gradeKey] = grade;
          saveGrade(state.currentId, patch).catch(function (err) {
            setError("d22-eval-error", err.message);
          });
        });
        wrap.appendChild(btn);
      });
      return wrap;
    }

    function renderDetail(id) {
      var box = $("d22-eval-detail");
      clear(box);
      if (!id) return;
      var question = state.questions.filter(function (q) { return q.id === id; })[0];
      var comparison = state.comparisons ? state.comparisons[id] : null;
      if (!question) return;

      var panel = make("section", "preview-panel");
      panel.appendChild(make("h2", null, "Control question " + question.id));
      panel.appendChild(make("p", "d22-question-echo", question.question));
      panel.appendChild(make("p", "field-hint", "Category: " + question.category));

      panel.appendChild(make("h3", null, "EXPECTED FACTS"));
      var facts = make("ul", "d22-facts");
      (question.expected_facts || []).forEach(function (fact) {
        facts.appendChild(make("li", null, fact));
      });
      panel.appendChild(facts);

      panel.appendChild(make("h3", null, "EXPECTED SOURCES"));
      var sources = make("ul", "d22-facts");
      (question.expected_sources || []).forEach(function (source) {
        sources.appendChild(make("li", null, expectedSourceLabel(source)));
      });
      panel.appendChild(sources);

      var record = recordFor(id);
      var grades = make("div", "d22-grades");
      grades.appendChild(gradeButtons("No RAG", record.no_rag, "no_rag"));
      grades.appendChild(gradeButtons("RAG", record.rag, "rag"));

      var sourceRow = make("div", "d22-grade");
      sourceRow.appendChild(make("span", "d22-grade-label", "Expected source retrieved:"));
      [["YES", true], ["NO", false]].forEach(function (pair) {
        var btn = make("button", "link-btn" +
          (record.expected_source_retrieved === pair[1] ? " active" : ""), pair[0]);
        btn.type = "button";
        btn.addEventListener("click", function () {
          saveGrade(id, { expected_source_retrieved: pair[1] }).catch(function (err) {
            setError("d22-eval-error", err.message);
          });
        });
        sourceRow.appendChild(btn);
      });
      grades.appendChild(sourceRow);
      panel.appendChild(grades);

      if (!comparison) {
        panel.appendChild(make("p", "field-hint", "Нажмите Run, чтобы получить ответы."));
        box.appendChild(panel);
        return;
      }

      var grid = make("div", "d22-answer-grid");
      grid.appendChild(renderAnswerCard("WITHOUT RAG", comparison.no_rag, "no_rag"));
      grid.appendChild(renderAnswerCard("WITH RAG", comparison.rag, "rag"));
      panel.appendChild(grid);
      panel.appendChild(renderChunks(comparison.rag.chunks, "RETRIEVED CONTEXT"));
      box.appendChild(panel);
    }

    async function loadQuestions(force) {
      if (state.questionsLoaded && !force) return;
      try {
        var questions = await api("GET", "/evaluation/questions");
        state.questions = questions || [];
        state.questionsLoaded = true;
        renderQuestionsTable();
        await loadResults();
      } catch (err) {
        setError("d22-eval-error", err.message);
      }
    }

    async function loadResults() {
      var response = await api("GET", "/evaluation/results");
      state.results = response.results || {};
      state.summary = response.summary;
      renderSummary();
    }

    async function runQuestion(id) {
      if (state.busy) return;
      state.busy = true;
      state.currentId = id;
      show($("d22-eval-loading"), true);
      $("d22-eval-loading").textContent = "Running " + id + "…";
      setError("d22-eval-error", "");
      try {
        var run = await api("POST", "/evaluation/run/" + encodeURIComponent(id),
          { top_k: selectedTopK() });
        state.comparisons = state.comparisons || {};
        state.comparisons[id] = run.comparison;
        renderDetail(id);
      } catch (err) {
        setError("d22-eval-error", err.message);
      } finally {
        state.busy = false;
        show($("d22-eval-loading"), false);
      }
    }

    async function runAll() {
      if (state.busy) return;
      state.busy = true;
      show($("d22-eval-loading"), true);
      $("d22-eval-loading").textContent = "Running all questions…";
      setError("d22-eval-error", "");
      try {
        var response = await api("POST", "/evaluation/run-all", { top_k: selectedTopK() });
        state.comparisons = state.comparisons || {};
        (response.runs || []).forEach(function (run) {
          state.comparisons[run.question.id] = run.comparison;
        });
        if (response.runs && response.runs.length) {
          state.currentId = response.runs[0].question.id;
          renderDetail(state.currentId);
        }
      } catch (err) {
        setError("d22-eval-error", err.message);
      } finally {
        state.busy = false;
        show($("d22-eval-loading"), false);
      }
    }

    /* ------------------------------------------------------------------ */
    /* sub-tabs                                                            */
    /* ------------------------------------------------------------------ */
    function showView(name) {
      var ask = name === "ask";
      $("d22-view-ask").classList.toggle("hidden", !ask);
      $("d22-view-eval").classList.toggle("hidden", ask);
      $("d22-tab-ask").classList.toggle("active", ask);
      $("d22-tab-eval").classList.toggle("active", !ask);
      $("d22-tab-ask").setAttribute("aria-selected", ask ? "true" : "false");
      $("d22-tab-eval").setAttribute("aria-selected", ask ? "false" : "true");
      if (!ask) loadQuestions();
    }

    /* ------------------------------------------------------------------ */
    /* init                                                                */
    /* ------------------------------------------------------------------ */
    $("d22-tab-ask").addEventListener("click", function () { showView("ask"); });
    $("d22-tab-eval").addEventListener("click", function () { showView("eval"); });
    $("d22-ask").addEventListener("click", ask);
    $("d22-compare").addEventListener("click", compare);
    $("d22-eval-run-all").addEventListener("click", runAll);
    $("d22-eval-refresh").addEventListener("click", function () {
      loadResults().catch(function (err) { setError("d22-eval-error", err.message); });
    });

    function start() {
      loadStatus();
      loadQuestions();
    }

    start();
    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day22") start();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay22);
  } else {
    initDay22();
  }
})();
