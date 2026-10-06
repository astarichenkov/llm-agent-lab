/* LLM Agent Lab — Day 23: query rewrite + relevance filtering.
 *
 * Vanilla JavaScript (no framework). Talks to:
 *   GET  /api/week5/day23/status
 *   POST /api/week5/day23/ask          (improved)
 *   POST /api/week5/day23/compare      (baseline vs improved)
 *   GET  /api/week5/day22/ask          (baseline RAG is reused from Day 22)
 *   GET  /api/week5/day23/evaluation/questions
 *   POST /api/week5/day23/evaluation/run/{id}
 *   POST /api/week5/day23/evaluation/run-all
 *   GET  /api/week5/day23/evaluation/results
 *   PUT  /api/week5/day23/evaluation/result/{id}
 *   GET  /api/week5/day23/evaluation/scores
 *
 * The backend owns retrieval, rewriting, filtering and generation. This file
 * only renders the structured result — including the full accepted/rejected
 * candidate trace — and never computes embeddings.
 */
(function () {
  "use strict";

  var API23 = "/api/week5/day23";
  var API22 = "/api/week5/day22";

  function initDay23() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day23",
      "d23-tab-ask", "d23-tab-eval", "d23-view-ask", "d23-view-eval",
      "d23-status", "d23-question", "d23-mode-baseline", "d23-mode-improved",
      "d23-retrieval-top-k", "d23-similarity-threshold", "d23-final-top-k",
      "d23-ask", "d23-compare", "d23-ask-loading", "d23-ask-error",
      "d23-pipeline-panel", "d23-pipeline", "d23-results",
      "d23-eval-summary", "d23-eval-run-all", "d23-eval-refresh",
      "d23-eval-scores-btn", "d23-eval-scores-panel", "d23-score-report",
      "d23-eval-loading", "d23-eval-error", "d23-eval-table-body",
      "d23-eval-detail",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day23: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      statusLoaded: false,
      status: null,
      questionsLoaded: false,
      questions: [],
      results: {},
      summary: null,
      comparisons: {},
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
    async function api(base, method, path, body) {
      var options = { method: method, headers: {} };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      var response = await fetch(base + path, options);
      var data = null;
      try { data = await response.json(); } catch (err) { data = null; }
      if (!response.ok) {
        var detail = data && data.detail ? data.detail : ("HTTP " + response.status);
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    }

    /* ------------------------------------------------------------------ */
    /* expected-source matching (mirrors backend source_matching.py)       */
    /* ------------------------------------------------------------------ */
    function num(value) {
      var n = Number(value);
      return isNaN(n) ? null : n;
    }
    function chunkCoversExpected(chunk, expected) {
      var m = chunk.metadata || {};
      if (m.source_type !== expected.source_type) return false;
      if (expected.source && m.source !== expected.source) return false;
      if (expected.source_type === "telegram") {
        if (!expected.message_ids || !expected.message_ids.length) return true;
        var mine = asList(m.message_ids).map(num);
        return asList(expected.message_ids).some(function (id) {
          return mine.indexOf(num(id)) !== -1;
        });
      }
      if (expected.page !== undefined && expected.page !== null) {
        if (num(m.page) === num(expected.page)) return true;
        var from = num(m.page_from), to = num(m.page_to);
        if (from !== null) {
          if (to === null) to = from;
          var lo = Math.min(from, to), hi = Math.max(from, to);
          return num(expected.page) >= lo && num(expected.page) <= hi;
        }
        return false;
      }
      return true;
    }
    function allExpectedRetrieved(chunks, expectedSources) {
      if (!expectedSources || !expectedSources.length) return null;
      var list = chunks || [];
      return expectedSources.every(function (expected) {
        return list.some(function (chunk) { return chunkCoversExpected(chunk, expected); });
      });
    }
    function expectedForQuestion(question) {
      var match = (state.questions || []).filter(function (q) {
        return q.question === question;
      })[0];
      return match ? match.expected_sources : null;
    }

    /* ------------------------------------------------------------------ */
    /* status                                                              */
    /* ------------------------------------------------------------------ */
    function renderStatus(status) {
      var box = $("d23-status");
      clear(box);
      box.appendChild(make("span", null, "Embedding model: " + (status.embedding_model || "—")));
      box.appendChild(make("span", null, "Generation model: " + (status.generation_model || "—")));
      box.appendChild(make("span", null, "Rewrite model: " + (status.rewrite_model || "—")));
      box.appendChild(make("span", null, "Index: " + (status.chunking || "—") +
        " (" + status.chunks + " chunks)"));
      box.appendChild(make("span", null, "Retrieval Top-K: " + status.default_retrieval_top_k));
      box.appendChild(make("span", null, "Final Top-K: " + status.default_final_top_k));
      box.appendChild(make("span", null, "Threshold: " + status.default_similarity_threshold));
      if (status.recommended_demo_question) {
        box.appendChild(make("span", "d23-hint",
          "Recommended demo question: " + status.recommended_demo_question));
      }
    }

    async function loadStatus() {
      if (state.statusLoaded) return;
      try {
        var status = await api(API23, "GET", "/status");
        state.status = status;
        renderStatus(status);
        state.statusLoaded = true;
        if (status.recommended_demo_question) {
          $("d23-question").value = status.recommended_demo_question;
        }
      } catch (err) {
        $("d23-status").textContent = "Не удалось получить статус: " + err.message;
      }
    }

    /* ------------------------------------------------------------------ */
    /* generic answer / chunk rendering                                    */
    /* ------------------------------------------------------------------ */
    function answerMeta(answer) {
      var meta = make("div", "d22-answer-meta");
      meta.appendChild(make("span", null, "Model: " + answer.model));
      meta.appendChild(make("span", null, "Top-K: " + (answer.top_k || answer.final_top_k)));
      if (answer.similarity_threshold !== undefined) {
        meta.appendChild(make("span", null, "Threshold: " + answer.similarity_threshold));
      }
      return meta;
    }

    function renderAnswerCard(title, answer, kind) {
      var card = make("article", "d22-answer-card d22-answer-" + kind);
      card.appendChild(make("h3", null, title));
      card.appendChild(answerMeta(answer));
      var body = make("div", "d17-answer");
      body.textContent = answer.answer || "(пустой ответ)";
      card.appendChild(body);
      if (answer.no_relevant_context) {
        card.appendChild(make("p", "field-hint",
          "No relevant chunks passed the configured threshold."));
      }
      return card;
    }

    function renderChunkCard(chunk, opts) {
      opts = opts || {};
      var card = make("article", "d22-chunk" + (opts.statusClass ? " " + opts.statusClass : ""));
      var head = make("div", "d22-chunk-head");
      head.appendChild(make("span", "d22-rank", "#" + chunk.rank));
      head.appendChild(make("span", "d22-similarity", "score: " + formatSim(chunk.similarity)));
      head.appendChild(make("span", "d22-source-type d22-source-" + chunk.source_type,
        "source_type: " + chunk.source_type));
      if (opts.statusLabel) {
        head.appendChild(make("span", "d23-status-pill " + (opts.statusClass || ""),
          opts.statusLabel));
      }
      card.appendChild(head);

      var meta = make("div", "d22-chunk-meta");
      if (chunk.source) meta.appendChild(make("span", null, "source: " + chunk.source));
      var m = chunk.metadata || {};
      if (chunk.source_type === "manual") {
        if (m.page !== undefined && m.page !== null) {
          meta.appendChild(make("span", null, "page: " + m.page));
        } else if (m.page_from !== undefined && m.page_from !== null) {
          meta.appendChild(make("span", null, "pages: " + m.page_from +
            (m.page_to && m.page_to !== m.page_from ? "–" + m.page_to : "")));
        }
        if (m.section) meta.appendChild(make("span", null, "section: " + m.section));
      } else if (chunk.source_type === "telegram") {
        if (m.chat_name) meta.appendChild(make("span", null, "chat: " + m.chat_name));
        if (m.date_from) meta.appendChild(make("span", null, "date: " + m.date_from));
        if (m.message_ids && m.message_ids.length) {
          meta.appendChild(make("span", null, "message_ids: " + asList(m.message_ids).join(", ")));
        }
        if (m.authors && m.authors.length) {
          meta.appendChild(make("span", null, "authors: " + asList(m.authors).join(", ")));
        }
      }
      meta.appendChild(make("span", null, "chunk_id: " + chunk.chunk_id));
      card.appendChild(meta);

      if (opts.reason) {
        card.appendChild(make("p", "field-hint", "reason: " + opts.reason));
      }

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

    function statusFor(chunk) {
      if (chunk.status === "used") return { label: "✓ USED", cls: "d23-used" };
      if (chunk.status === "relevant_not_used") {
        return { label: "✓ RELEVANT (not used: final_top_k limit)", cls: "d23-relevant" };
      }
      return { label: "✕ BELOW THRESHOLD", cls: "d23-rejected" };
    }

    function renderCandidateList(candidates, heading) {
      var section = make("article", "d22-context");
      section.appendChild(make("h3", null, heading));
      if (!candidates || !candidates.length) {
        section.appendChild(make("p", "field-hint", "Нет кандидатов."));
        return section;
      }
      candidates.forEach(function (chunk) {
        var st = statusFor(chunk);
        section.appendChild(renderChunkCard(chunk, {
          statusLabel: st.label, statusClass: st.cls, reason: chunk.reason,
        }));
      });
      return section;
    }

    function renderRejected(result) {
      var rejected = result.rejected_candidates || [];
      var section = make("article", "d22-context");
      var details = document.createElement("details");
      details.className = "d23-rejected-details";
      var summary = make("summary", null, "Rejected chunks (" + rejected.length + ")");
      details.appendChild(summary);
      if (!rejected.length) {
        details.appendChild(make("p", "field-hint", "Ни один кандидат не отклонён."));
      } else {
        rejected.forEach(function (chunk) {
          details.appendChild(renderChunkCard(chunk, {
            statusLabel: "✕ BELOW THRESHOLD", statusClass: "d23-rejected",
            reason: chunk.reason,
          }));
        });
      }
      section.appendChild(details);
      return section;
    }

    function statCard(label, value, cls) {
      var card = make("div", "d23-stat" + (cls ? " " + cls : ""));
      card.appendChild(make("span", "d23-stat-value", value));
      card.appendChild(make("span", "d23-stat-label", label));
      return card;
    }

    function renderPipelineStats(result) {
      var wrap = make("div", "d23-stats");
      wrap.appendChild(statCard("RETRIEVED", (result.retrieved_candidates || []).length, "d23-stat-retrieved"));
      wrap.appendChild(statCard("PASSED FILTER", (result.accepted_candidates || []).length, "d23-stat-passed"));
      wrap.appendChild(statCard("USED BY LLM", (result.context_chunks || []).length, "d23-stat-used"));
      wrap.appendChild(statCard("REJECTED", (result.rejected_candidates || []).length, "d23-stat-rejected"));
      return wrap;
    }

    function renderRewriteBlock(result) {
      var wrap = make("div", "d23-rewrite");
      wrap.appendChild(make("h3", null, "ORIGINAL QUERY"));
      wrap.appendChild(make("p", "d23-query-text", result.original_question));
      wrap.appendChild(make("h3", null, "REWRITTEN QUERY"));
      wrap.appendChild(make("p", "d23-query-text d23-rewritten", result.rewritten_query));
      var info = result.rewrite || {};
      if (!info.applied) {
        wrap.appendChild(make("p", "field-hint",
          "Query rewrite not applied — fallback to the original question." +
          (info.error ? " (" + info.error + ")" : "")));
      }
      return wrap;
    }

    function renderImproved(result) {
      var box = $("d23-results");
      clear(box);
      box.appendChild(renderImprovedSection(result));
    }

    function renderImprovedSection(result) {
      var section = make("section", "d23-improved");
      section.appendChild(renderRewriteBlock(result));
      section.appendChild(renderPipelineStats(result));
      section.appendChild(renderAnswerCard("IMPROVED RAG ANSWER", result, "rag"));
      section.appendChild(renderCandidateList(
        result.retrieved_candidates, "RETRIEVAL CANDIDATES (before filtering)"));
      section.appendChild(renderCandidateList(
        result.context_chunks, "FINAL CONTEXT (used by the LLM)"));
      section.appendChild(renderRejected(result));
      return section;
    }

    function renderBaseline(answer) {
      var box = $("d23-results");
      clear(box);
      box.appendChild(make("h3", null, "ORIGINAL QUERY"));
      box.appendChild(make("p", "d23-query-text", answer.question));
      box.appendChild(make("p", "field-hint",
        "Baseline = Day 22 pipeline: vector search → Top-" + answer.top_k +
        " → context → LLM. No rewrite, no threshold."));
      box.appendChild(renderAnswerCard("BASELINE RAG ANSWER", answer, "no_rag"));
      box.appendChild(renderCandidateList(answer.chunks, "RETRIEVED CONTEXT (Top-" + answer.top_k + ")"));
      renderPipeline(answer.pipeline);
    }

    function renderPipeline(stages) {
      $("d23-pipeline").textContent = (stages || []).join("\n   ↓\n");
      show($("d23-pipeline-panel"), true);
    }

    /* ------------------------------------------------------------------ */
    /* comparison                                                          */
    /* ------------------------------------------------------------------ */
    function compareColumn(title, subtitle, answer, chunks, expectedFlag, result) {
      var col = make("div", "d23-compare-col");
      col.appendChild(make("h3", null, title));
      col.appendChild(make("p", "field-hint", subtitle));
      if (result) {
        col.appendChild(make("h4", null, "Original query"));
        col.appendChild(make("p", "d23-query-text", result.original_question));
        col.appendChild(make("h4", null, "Rewritten query"));
        col.appendChild(make("p", "d23-query-text d23-rewritten", result.rewritten_query));
        col.appendChild(renderPipelineStats(result));
      }
      col.appendChild(renderAnswerCard(result ? "IMPROVED ANSWER" : "BASELINE ANSWER", answer,
        result ? "rag" : "no_rag"));
      var metrics = make("div", "d23-compare-metrics");
      metrics.appendChild(make("span", null, "Retrieved: " + chunks.length));
      metrics.appendChild(make("span", null, "Used: " + chunks.length));
      metrics.appendChild(make("span", null, "Expected sources found: " +
        (expectedFlag === null ? "—" : (expectedFlag ? "YES" : "NO"))));
      col.appendChild(metrics);
      return col;
    }

    function renderComparison(comparison) {
      var box = $("d23-results");
      clear(box);
      box.appendChild(make("h2", null, "QUESTION"));
      box.appendChild(make("p", "d23-query-text", comparison.question));

      var expected = expectedForQuestion(comparison.question);
      var baselineExpected = allExpectedRetrieved(comparison.baseline.chunks, expected);
      var improvedExpected = allExpectedRetrieved(comparison.improved.context_chunks, expected);

      var grid = make("div", "d23-compare-grid");
      grid.appendChild(compareColumn(
        "BASELINE RAG", "Day 22 — no rewrite, no threshold",
        comparison.baseline, comparison.baseline.chunks, baselineExpected, null));
      grid.appendChild(compareColumn(
        "IMPROVED RAG", "Day 23 — rewrite + threshold",
        comparison.improved, comparison.improved.context_chunks, improvedExpected,
        comparison.improved));
      box.appendChild(grid);

      box.appendChild(renderCandidateList(
        comparison.improved.retrieved_candidates, "IMPROVED — RETRIEVAL CANDIDATES"));
      box.appendChild(renderRejected(comparison.improved));
      renderPipeline(comparison.improved.pipeline);
    }

    /* ------------------------------------------------------------------ */
    /* ask / compare                                                       */
    /* ------------------------------------------------------------------ */
    function selectedMode() {
      return $("d23-mode-improved").checked ? "improved" : "baseline";
    }
    function improvedSettings() {
      return {
        retrieval_top_k: parseInt($("d23-retrieval-top-k").value, 10) || 20,
        similarity_threshold: parseFloat($("d23-similarity-threshold").value),
        final_top_k: parseInt($("d23-final-top-k").value, 10) || 5,
      };
    }
    function setBusy(on, label) {
      state.busy = on;
      $("d23-ask").disabled = on;
      $("d23-compare").disabled = on;
      var loading = $("d23-ask-loading");
      if (on) loading.textContent = label || "Searching knowledge base…";
      show(loading, on);
    }

    async function ask() {
      if (state.busy) return;
      var question = $("d23-question").value.trim();
      if (!question) { setError("d23-ask-error", "Введите вопрос."); return; }
      setError("d23-ask-error", "");
      setBusy(true, "Searching knowledge base…");
      try {
        if (selectedMode() === "baseline") {
          var baseline = await api(API22, "POST", "/ask",
            { question: question, mode: "rag", top_k: 5 });
          renderBaseline(baseline);
        } else {
          var settings = improvedSettings();
          var result = await api(API23, "POST", "/ask", {
            question: question,
            retrieval_top_k: settings.retrieval_top_k,
            final_top_k: settings.final_top_k,
            similarity_threshold: settings.similarity_threshold,
          });
          renderImproved(result);
        }
      } catch (err) {
        setError("d23-ask-error", err.message);
      } finally {
        setBusy(false);
      }
    }

    async function compare() {
      if (state.busy) return;
      var question = $("d23-question").value.trim();
      if (!question) { setError("d23-ask-error", "Введите вопрос."); return; }
      setError("d23-ask-error", "");
      setBusy(true, "Running baseline and improved…");
      try {
        var settings = improvedSettings();
        var comparison = await api(API23, "POST", "/compare", {
          question: question,
          retrieval_top_k: settings.retrieval_top_k,
          final_top_k: settings.final_top_k,
          similarity_threshold: settings.similarity_threshold,
        });
        renderComparison(comparison);
      } catch (err) {
        setError("d23-ask-error", err.message);
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
      var body = $("d23-eval-table-body");
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
        var run = make("button", "link-btn", "Run baseline vs improved");
        run.type = "button";
        run.addEventListener("click", function () { runQuestion(question.id); });
        actions.appendChild(run);
        tr.appendChild(actions);

        body.appendChild(tr);
      });
    }

    function gradeBlock(title, counts, total) {
      var wrap = make("div", "d22-summary-block");
      wrap.appendChild(make("b", null, title));
      ["pass", "partial", "fail"].forEach(function (grade) {
        wrap.appendChild(make("span", "d22-summary-item",
          grade.toUpperCase() + ": " + ((counts && counts[grade]) || 0) + " / " + total));
      });
      return wrap;
    }

    function renderSummary() {
      var box = $("d23-eval-summary");
      clear(box);
      var summary = state.summary;
      if (!summary) { box.textContent = "No data"; return; }
      box.appendChild(gradeBlock("BASELINE RAG", summary.baseline, summary.total_questions));
      box.appendChild(gradeBlock("IMPROVED RAG", summary.improved, summary.total_questions));

      var hits = make("div", "d22-summary-block");
      hits.appendChild(make("b", null, "EXPECTED SOURCE HIT RATE"));
      hits.appendChild(make("span", "d22-summary-item",
        "Baseline: " + (summary.baseline_expected_source_retrieved || 0) + " / " + summary.total_questions));
      hits.appendChild(make("span", "d22-summary-item",
        "Improved: " + (summary.improved_expected_source_retrieved || 0) + " / " + summary.total_questions));
      box.appendChild(hits);
      box.appendChild(make("div", "field-hint",
        "Оценено вопросов: " + summary.evaluated + " / " + summary.total_questions));
    }

    function recordFor(id) {
      var record = state.results[id] || {};
      return {
        baseline: record.baseline || null,
        improved: record.improved || null,
        baseline_expected_source_retrieved:
          record.baseline_expected_source_retrieved === undefined
            ? null : record.baseline_expected_source_retrieved,
        improved_expected_source_retrieved:
          record.improved_expected_source_retrieved === undefined
            ? null : record.improved_expected_source_retrieved,
      };
    }

    async function saveGrade(id, patch) {
      var record = recordFor(id);
      Object.keys(patch).forEach(function (key) { record[key] = patch[key]; });
      var response = await api(API23, "PUT",
        "/evaluation/result/" + encodeURIComponent(id), record);
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
            setError("d23-eval-error", err.message);
          });
        });
        wrap.appendChild(btn);
      });
      return wrap;
    }

    function sourceFlagRow(label, value, key) {
      var row = make("div", "d22-grade");
      row.appendChild(make("span", "d22-grade-label", label + ":"));
      [["YES", true], ["NO", false]].forEach(function (pair) {
        var btn = make("button", "link-btn" +
          (value === pair[1] ? " active" : ""), pair[0]);
        btn.type = "button";
        btn.addEventListener("click", function () {
          var patch = {};
          patch[key] = pair[1];
          saveGrade(state.currentId, patch).catch(function (err) {
            setError("d23-eval-error", err.message);
          });
        });
        row.appendChild(btn);
      });
      return row;
    }

    function renderDetail(id) {
      var box = $("d23-eval-detail");
      clear(box);
      if (!id) return;
      var question = state.questions.filter(function (q) { return q.id === id; })[0];
      var run = state.comparisons[id];
      if (!question) return;

      var panel = make("section", "preview-panel");
      panel.appendChild(make("h2", null, "Control question " + question.id));
      panel.appendChild(make("p", "d23-query-text", question.question));
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
      grades.appendChild(gradeButtons("Baseline", record.baseline, "baseline"));
      grades.appendChild(gradeButtons("Improved", record.improved, "improved"));
      grades.appendChild(sourceFlagRow("Baseline expected source retrieved",
        record.baseline_expected_source_retrieved, "baseline_expected_source_retrieved"));
      grades.appendChild(sourceFlagRow("Improved expected source retrieved",
        record.improved_expected_source_retrieved, "improved_expected_source_retrieved"));
      panel.appendChild(grades);

      if (!run) {
        panel.appendChild(make("p", "field-hint", "Нажмите Run, чтобы получить ответы."));
        box.appendChild(panel);
        return;
      }

      var grid = make("div", "d23-compare-grid");
      grid.appendChild(compareColumn("BASELINE RAG", "Day 22",
        run.comparison.baseline, run.comparison.baseline.chunks,
        run.baseline_expected_source_retrieved, null));
      grid.appendChild(compareColumn("IMPROVED RAG", "Day 23",
        run.comparison.improved, run.comparison.improved.context_chunks,
        run.improved_expected_source_retrieved, run.comparison.improved));
      panel.appendChild(grid);
      panel.appendChild(renderCandidateList(run.comparison.improved.retrieved_candidates,
        "IMPROVED — RETRIEVAL CANDIDATES"));
      panel.appendChild(renderRejected(run.comparison.improved));
      panel.appendChild(make("p", "field-hint",
        "Auto expected source check — Baseline: " +
        (run.baseline_expected_source_retrieved ? "YES" : "NO") +
        "; Improved: " + (run.improved_expected_source_retrieved ? "YES" : "NO")));
      box.appendChild(panel);
    }

    function renderScoreReport(report) {
      var box = $("d23-score-report");
      clear(box);
      box.appendChild(make("p", "field-hint",
        "retrieval_top_k=" + report.retrieval_top_k +
        ", threshold=" + report.similarity_threshold +
        " · best expected: min=" + report.best_expected_min +
        ", avg=" + report.best_expected_avg +
        " · expected chunks kept " + report.expected_kept + "/" + report.expected_total +
        " · questions whose best expected passed: " +
        report.question_best_expected_kept + "/" + report.question_count +
        " · noise removed: " + report.noise_removed));
      var table = make("table", "d22-table d23-score-table");
      var thead = document.createElement("thead");
      var hrow = document.createElement("tr");
      ["ID", "Top-1", "Top-5 min", "Top-K min", "Best expected",
        "Rank", "#expected", "#passed", "#noise"].forEach(function (h) {
        hrow.appendChild(make("th", null, h));
      });
      thead.appendChild(hrow);
      table.appendChild(thead);
      var tbody = document.createElement("tbody");
      (report.rows || []).forEach(function (row) {
        var tr = document.createElement("tr");
        [row.id, row.top1, row.top5_min, row.top_retrieval_min,
          row.best_expected_score, row.best_expected_rank,
          row.expected_chunk_count, row.matched_at_threshold, row.noise_count
        ].forEach(function (value) {
          tr.appendChild(make("td", null,
            typeof value === "number" ? value.toFixed(4) : (value === null ? "—" : value)));
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      box.appendChild(table);
      show($("d23-eval-scores-panel"), true);
    }

    async function loadScoreReport() {
      setError("d23-eval-error", "");
      show($("d23-eval-scores-panel"), true);
      $("d23-score-report").textContent = "Calculating…";
      try {
        var settings = improvedSettings();
        var query = "?retrieval_top_k=" + settings.retrieval_top_k +
          "&similarity_threshold=" + settings.similarity_threshold;
        renderScoreReport(await api(API23, "GET", "/evaluation/scores" + query));
      } catch (err) {
        setError("d23-eval-error", err.message);
        $("d23-score-report").textContent = "";
      }
    }

    async function loadQuestions(force) {
      if (state.questionsLoaded && !force) return;
      try {
        state.questions = await api(API23, "GET", "/evaluation/questions");
        state.questionsLoaded = true;
        renderQuestionsTable();
        await loadResults();
      } catch (err) {
        setError("d23-eval-error", err.message);
      }
    }

    async function loadResults() {
      var response = await api(API23, "GET", "/evaluation/results");
      state.results = response.results || {};
      state.summary = response.summary;
      renderSummary();
    }

    async function runQuestion(id) {
      if (state.busy) return;
      state.busy = true;
      state.currentId = id;
      show($("d23-eval-loading"), true);
      $("d23-eval-loading").textContent = "Running " + id + "…";
      setError("d23-eval-error", "");
      try {
        var settings = improvedSettings();
        var run = await api(API23, "POST",
          "/evaluation/run/" + encodeURIComponent(id), {
            retrieval_top_k: settings.retrieval_top_k,
            final_top_k: settings.final_top_k,
            similarity_threshold: settings.similarity_threshold,
          });
        state.comparisons[id] = run;
        renderDetail(id);
      } catch (err) {
        setError("d23-eval-error", err.message);
      } finally {
        state.busy = false;
        show($("d23-eval-loading"), false);
      }
    }

    async function runAll() {
      if (state.busy) return;
      state.busy = true;
      show($("d23-eval-loading"), true);
      $("d23-eval-loading").textContent = "Running all questions…";
      setError("d23-eval-error", "");
      try {
        var settings = improvedSettings();
        var response = await api(API23, "POST", "/evaluation/run-all", {
          retrieval_top_k: settings.retrieval_top_k,
          final_top_k: settings.final_top_k,
          similarity_threshold: settings.similarity_threshold,
        });
        (response.runs || []).forEach(function (run) {
          state.comparisons[run.question.id] = run;
        });
        if (response.runs && response.runs.length) {
          state.currentId = response.runs[0].question.id;
          renderDetail(state.currentId);
        }
      } catch (err) {
        setError("d23-eval-error", err.message);
      } finally {
        state.busy = false;
        show($("d23-eval-loading"), false);
      }
    }

    /* ------------------------------------------------------------------ */
    /* sub-tabs / init                                                     */
    /* ------------------------------------------------------------------ */
    function showView(name) {
      var ask = name === "ask";
      $("d23-view-ask").classList.toggle("hidden", !ask);
      $("d23-view-eval").classList.toggle("hidden", ask);
      $("d23-tab-ask").classList.toggle("active", ask);
      $("d23-tab-eval").classList.toggle("active", !ask);
      $("d23-tab-ask").setAttribute("aria-selected", ask ? "true" : "false");
      $("d23-tab-eval").setAttribute("aria-selected", ask ? "false" : "true");
      if (!ask) loadQuestions();
    }

    $("d23-tab-ask").addEventListener("click", function () { showView("ask"); });
    $("d23-tab-eval").addEventListener("click", function () { showView("eval"); });
    $("d23-ask").addEventListener("click", ask);
    $("d23-compare").addEventListener("click", compare);
    $("d23-eval-run-all").addEventListener("click", runAll);
    $("d23-eval-refresh").addEventListener("click", function () {
      loadResults().catch(function (err) { setError("d23-eval-error", err.message); });
    });
    $("d23-eval-scores-btn").addEventListener("click", loadScoreReport);

    function start() { loadStatus(); }
    start();
    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day23") start();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay23);
  } else {
    initDay23();
  }
})();
