/* LLM Agent Lab — Day 24: grounded RAG, citations and anti-hallucination.
 *
 * Vanilla JavaScript (no framework). Talks to:
 *   GET  /api/week5/day24/status
 *   POST /api/week5/day24/ask
 *   GET  /api/week5/day24/evaluation/questions
 *   POST /api/week5/day24/evaluation/run/{id}
 *   POST /api/week5/day24/evaluation/run-all
 *   GET  /api/week5/day24/evaluation/results
 *   PUT  /api/week5/day24/evaluation/result/{id}
 *   GET  /api/week5/day24/negative/questions
 *   POST /api/week5/day24/negative/run-all
 *   GET  /api/week5/day24/negative/report
 *
 * The backend owns retrieval, the grounding gate and the citation validator.
 * This file only renders the structured result and NEVER invents a source:
 * every field it displays comes from the API response.
 */
(function () {
  "use strict";

  var API = "/api/week5/day24";

  function initDay24() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day24",
      "d24-tab-ask", "d24-tab-eval", "d24-tab-negative",
      "d24-view-ask", "d24-view-eval", "d24-view-negative",
      "d24-status", "d24-question", "d24-retrieval-top-k",
      "d24-similarity-threshold", "d24-final-top-k",
      "d24-ask", "d24-ask-loading", "d24-ask-error",
      "d24-pipeline-panel", "d24-pipeline", "d24-results",
      "d24-eval-summary", "d24-eval-run-all", "d24-eval-refresh",
      "d24-eval-loading", "d24-eval-error", "d24-eval-table-body",
      "d24-eval-detail",
      "d24-negative-summary", "d24-negative-run-all",
      "d24-negative-refresh", "d24-negative-loading",
      "d24-negative-error", "d24-negative-table-body",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day24: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = {
      status: null,
      questions: [],
      results: {},
      summary: null,
      lastResult: null,
      evalLoaded: false,
      negativeLoaded: false,
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
    function asList(value) {
      if (value === undefined || value === null) return [];
      return Array.isArray(value) ? value : [value];
    }
    function previewText(text, limit) {
      var t = (text || "").replace(/\s+/g, " ").trim();
      var max = limit || 220;
      return t.length > max ? t.slice(0, max) + "…" : t;
    }
    function formatScore(value) {
      var n = Number(value);
      return isNaN(n) ? "—" : n.toFixed(4);
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
    /* section tabs                                                        */
    /* ------------------------------------------------------------------ */
    function switchSection(name) {
      var map = {
        ask: ["d24-tab-ask", "d24-view-ask"],
        eval: ["d24-tab-eval", "d24-view-eval"],
        negative: ["d24-tab-negative", "d24-view-negative"],
      };
      Object.keys(map).forEach(function (key) {
        var btn = $(map[key][0]);
        var view = $(map[key][1]);
        var on = key === name;
        if (btn) {
          btn.classList.toggle("active", on);
          btn.setAttribute("aria-selected", on ? "true" : "false");
        }
        if (view) view.classList.toggle("hidden", !on);
      });
      if (name === "eval" && !state.evalLoaded) { loadEvaluation(); }
      if (name === "negative" && !state.negativeLoaded) { loadNegative(); }
    }
    $("d24-tab-ask").addEventListener("click", function () { switchSection("ask"); });
    $("d24-tab-eval").addEventListener("click", function () { switchSection("eval"); });
    $("d24-tab-negative").addEventListener("click", function () { switchSection("negative"); });

    /* ------------------------------------------------------------------ */
    /* status                                                              */
    /* ------------------------------------------------------------------ */
    function renderStatus() {
      var box = $("d24-status");
      clear(box);
      var s = state.status;
      if (!s) { box.textContent = "—"; return; }
      var rows = [
        ["Generation", s.generation_provider + " / " + s.generation_model],
        ["Embedding", s.embedding_model],
        ["Index", s.index_path + " (" + s.chunks + " chunks)"],
        ["Retrieval Top-K", s.default_retrieval_top_k],
        ["Answer threshold", s.default_similarity_threshold],
        ["Rewrite model", s.rewrite_model],
      ];
      rows.forEach(function (row) {
        var line = make("div", "d24-status-row");
        line.appendChild(make("span", "d24-status-label", row[0]));
        line.appendChild(make("code", null, row[1]));
        box.appendChild(line);
      });
    }

    async function loadStatus() {
      try {
        state.status = await api("GET", "/status");
        if (state.status && state.status.default_similarity_threshold !== undefined) {
          $("d24-similarity-threshold").value = state.status.default_similarity_threshold;
        }
      } catch (err) {
        state.status = null;
      }
      renderStatus();
    }

    /* ------------------------------------------------------------------ */
    /* evidence rendering                                                  */
    /* ------------------------------------------------------------------ */
    function sourceTypeLabel(sourceType) {
      if (sourceType === "manual") return "MANUAL / TECHNICAL SOURCE";
      if (sourceType === "telegram") return "COMMUNITY / TELEGRAM";
      return (sourceType || "SOURCE").toUpperCase();
    }

    function sourceIcon(sourceType) {
      if (sourceType === "manual") return "\uD83D\uDCCE";
      if (sourceType === "telegram") return "\uD83D\uDCAC";
      return "\uD83D\uDD17";
    }

    function manualPage(item) {
      var page = item.page;
      if (page === null || page === undefined) {
        if (item.page_from !== null && item.page_from !== undefined) {
          page = item.page_from === item.page_to
            ? item.page_from
            : item.page_from + "\u2013" + item.page_to;
        }
      }
      return page;
    }

    // Human-readable label for a source card / compact list. Generic over
    // GroundedSource and GroundedCitation (same field names).
    function sourceLabel(item) {
      if (item.source_type === "manual") {
        var name = item.title || "Xpander Owner's Manual";
        var page = manualPage(item);
        return page === null || page === undefined ? name : name + " \u2014 page " + page;
      }
      if (item.source_type === "telegram") {
        var chat = item.chat_name || "Mitsubishi Xpander Telegram";
        if (item.cited_message_id !== null && item.cited_message_id !== undefined) {
          return chat + " \u2014 message " + item.cited_message_id;
        }
        if (item.link_kind === "conversation" && item.message_ids && item.message_ids.length) {
          return chat + " \u2014 discussion from message " + item.message_ids[0];
        }
        if (item.message_ids && item.message_ids.length) {
          return chat + " \u2014 message " + item.message_ids.join(", ");
        }
        return chat;
      }
      return item.title || item.source || "Source";
    }

    function externalLink(url, text, note) {
      var link = make("a", "d24-source-link");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      if (note) link.title = note;
      link.appendChild(document.createTextNode(text));
      link.appendChild(make("span", "d24-link-arrow", " \u2197"));
      return link;
    }

    function sourceTitleNode(item, prefix) {
      var label = prefix ? prefix + " " + sourceLabel(item) : sourceLabel(item);
      if (item.url) {
        return externalLink(item.url, label, item.link_note || "");
      }
      return make("span", "d24-source-plain", label);
    }

    // Compact clickable source list shown immediately under the answer.
    function renderCompactSources(sources) {
      var section = make("section", "d24-answer d24-compact");
      section.appendChild(make("h3", null, "SOURCES"));
      var list = make("ul", "d24-compact-list");
      asList(sources).forEach(function (source) {
        var li = make("li", "d24-compact-item");
        li.appendChild(make("span", "d24-compact-icon", sourceIcon(source.source_type)));
        // The source title is itself the clickable link when a URL resolves.
        li.appendChild(sourceTitleNode(source, ""));
        if (source.link_note) {
          li.appendChild(make("span", "d24-link-note", source.link_note));
        }
        list.appendChild(li);
      });
      section.appendChild(list);
      return section;
    }

    // Main cited message + the other messages of a Telegram chunk as links.
    function renderContextLinks(item) {
      var links = asList(item.context_links);
      if (links.length <= 1) return null;
      var wrap = make("div", "d24-context-links");
      wrap.appendChild(make("span", "d24-context-label", "Context messages:"));
      links.forEach(function (link) {
        if (link.is_cited && item.url) {
          wrap.appendChild(externalLink(
            item.url, "message " + link.message_id + " (cited)", item.link_note || ""
          ));
        } else if (link.url) {
          wrap.appendChild(externalLink(link.url, "message " + link.message_id));
        } else {
          wrap.appendChild(make("span", "d24-context-plain", "message " + link.message_id));
        }
      });
      return wrap;
    }

    function sourceMetaLines(item) {
      var lines = [];
      if (item.source_type === "manual") {
        lines.push(["Source", item.source]);
        var page = manualPage(item);
        if (page !== null && page !== undefined) lines.push(["Page", page]);
        if (item.section) lines.push(["Section", item.section]);
      } else if (item.source_type === "telegram") {
        lines.push(["Source", item.chat_name || item.source]);
        if (item.message_ids && item.message_ids.length) {
          lines.push(["Messages", item.message_ids.join(", ")]);
        }
        if (item.date_from) {
          var range = item.date_from;
          if (item.date_to && item.date_to !== item.date_from) {
            range += " \u2014 " + item.date_to;
          }
          lines.push(["Date", range]);
        }
      } else if (item.source) {
        lines.push(["Source", item.source]);
      }
      lines.push(["Chunk", item.chunk_id]);
      return lines;
    }

    function renderEvidence(citations) {
      var wrap = make("div", "d24-evidence");
      asList(citations).forEach(function (citation, index) {
        var card = make("article", "d24-evidence-item");
        if (citation.source_type === "manual") card.classList.add("d24-evidence-manual");
        if (citation.source_type === "telegram") card.classList.add("d24-evidence-telegram");

        var head = make("div", "d24-evidence-head");
        head.appendChild(make("span", "d24-evidence-index", "[" + (index + 1) + "]"));
        head.appendChild(make("span", "d24-evidence-icon", sourceIcon(citation.source_type)));
        head.appendChild(sourceTitleNode(citation, ""));
        card.appendChild(head);
        card.appendChild(make(
          "div",
          "d24-source-kind d24-source-" + (citation.source_type || "other"),
          sourceTypeLabel(citation.source_type)
        ));

        var meta = make("div", "d24-evidence-meta");
        sourceMetaLines(citation).forEach(function (row) {
          var line = make("div", "d24-meta-row");
          line.appendChild(make("span", "d24-meta-label", row[0]));
          line.appendChild(make("span", "d24-meta-value", row[1]));
          meta.appendChild(line);
        });
        card.appendChild(meta);

        var quote = make("blockquote", "d24-quote");
        quote.textContent = citation.quote || "(\u043f\u0443\u0441\u0442\u0430\u044f \u0446\u0438\u0442\u0430\u0442\u0430)";
        card.appendChild(quote);

        var valid = make(
          "div",
          "d24-validation " + (citation.quote_valid ? "ok" : "err"),
          citation.quote_valid ? "quote verified: YES" : "quote verified: NO"
        );
        if (!citation.quote_valid && citation.validation_errors && citation.validation_errors.length) {
          valid.textContent += " \u2014 " + citation.validation_errors.join(", ");
        }
        card.appendChild(valid);
        var contextLinks = renderContextLinks(citation);
        if (contextLinks) {
          card.appendChild(contextLinks);
        }
        if (citation.link_note) {
          card.appendChild(make("div", "d24-link-note", citation.link_note));
        }

        var full = make("details", "d24-full-chunk");
        full.appendChild(make("summary", null, "Show full chunk"));
        var pre = make("pre", "d24-full-chunk-text", citation.chunk_text || "(chunk text unavailable)");
        full.appendChild(pre);
        card.appendChild(full);

        wrap.appendChild(card);
      });
      return wrap;
    }

    function renderRetrievalDebug(retrieval) {
      var details = make("details", "d24-debug");
      details.appendChild(make("summary", null, "Technical details (retrieval + gate)"));
      var body = make("div", "d24-debug-body");

      var rows = [
        ["Rewritten query", retrieval.rewritten_query || "—"],
        ["Retrieval Top-K", retrieval.retrieval_top_k],
        ["Final Top-K", retrieval.final_top_k],
        ["Similarity threshold", formatScore(retrieval.similarity_threshold)],
        ["Answer threshold", formatScore(retrieval.answer_threshold)],
        ["Gate", retrieval.gate_reason + (retrieval.gate_passed ? " (passed)" : " (refused)")],
        ["Best accepted score", formatScore(retrieval.best_score)],
        ["Retrieved", retrieval.retrieved_count],
        ["Passed filter", retrieval.accepted_count],
        ["Rejected", retrieval.rejected_count],
        ["In context", retrieval.context_count],
      ];
      rows.forEach(function (row) {
        var line = make("div", "d24-meta-row");
        line.appendChild(make("span", "d24-meta-label", row[0]));
        line.appendChild(make("span", "d24-meta-value", row[1]));
        body.appendChild(line);
      });

      if (retrieval.rejected_candidates && retrieval.rejected_candidates.length) {
        var rejTitle = make("h4", null, "Rejected retrieval candidates (NOT evidence)");
        body.appendChild(rejTitle);
        var rej = make("div", "d24-rejected");
        retrieval.rejected_candidates.forEach(function (c) {
          var row = make("div", "d24-rejected-row");
          row.appendChild(make("span", "d24-rejected-score", formatScore(c.similarity)));
          row.appendChild(make("span", null, "[" + (c.source_type || "?") + "] " + c.chunk_id));
          row.appendChild(make("span", "d24-rejected-text", previewText(c.text, 140)));
          rej.appendChild(row);
        });
        body.appendChild(rej);
      }
      details.appendChild(body);
      return details;
    }

    /* ------------------------------------------------------------------ */
    /* grounded result                                                     */
    /* ------------------------------------------------------------------ */
    function renderResult(result, mount) {
      clear(mount);
      var badge = make(
        "div",
        "d24-status-badge d24-status-" + result.status,
        result.status === "answered"
          ? "ANSWERED — evidence verified"
          : (result.status === "insufficient_context"
            ? "INSUFFICIENT CONTEXT"
            : "GROUNDING FAILED")
      );
      mount.appendChild(badge);

      var question = make("p", "d22-question-echo", result.question);
      mount.appendChild(question);

      if (result.status === "insufficient_context") {
        var refusal = make("section", "d24-refusal");
        refusal.appendChild(make("h3", null, "Я не знаю"));
        refusal.appendChild(make("p", null, result.message || "Недостаточно данных."));
        if (result.clarification_request) {
          refusal.appendChild(make("p", "d24-clarification", result.clarification_request));
        }
        refusal.appendChild(make(
          "p",
          "d24-llm-note",
          "Generation LLM invoked: " + (result.llm_called ? "YES" : "NO")
        ));
        mount.appendChild(refusal);
        if (result.retrieval) mount.appendChild(renderRetrievalDebug(result.retrieval));
        return;
      }

      var answerSection = make("section", "d24-answer");
      answerSection.appendChild(make("h3", null, "ANSWER"));
      answerSection.appendChild(make("div", "d24-answer-text", result.answer || ""));
      if (result.status === "grounding_failed") {
        answerSection.appendChild(make("p", "d24-grounding-failed", result.message || ""));
      }
      mount.appendChild(answerSection);

      if (result.sources && result.sources.length) {
        mount.appendChild(renderCompactSources(result.sources));
      }

      var evidenceSection = make("section", "d24-answer");
      evidenceSection.appendChild(make("h3", null, "EVIDENCE"));
      if (result.citations && result.citations.length) {
        evidenceSection.appendChild(renderEvidence(result.citations));
      } else {
        evidenceSection.appendChild(make(
          "p",
          "d24-no-evidence",
          "Evidence отсутствует — ответ нельзя считать подтверждённым."
        ));
      }
      if (result.validation_errors && result.validation_errors.length) {
        var errBox = make("ul", "d24-validation-errors");
        result.validation_errors.forEach(function (e) {
          errBox.appendChild(make("li", null, e));
        });
        evidenceSection.appendChild(errBox);
      }
      mount.appendChild(evidenceSection);

      var validation = make(
        "div",
        "d24-grounding-validation " + (result.grounding_valid ? "ok" : "err"),
        "Grounding validation: " + (result.grounding_valid ? "PASSED" : "FAILED")
      );
      mount.appendChild(validation);

      if (result.retrieval) mount.appendChild(renderRetrievalDebug(result.retrieval));
    }

    /* ------------------------------------------------------------------ */
    /* ask                                                                 */
    /* ------------------------------------------------------------------ */
    async function ask() {
      var question = ($("d24-question").value || "").trim();
      if (!question) { setError("d24-ask-error", "Введите вопрос."); return; }
      setError("d24-ask-error", "");
      show($("d24-ask-loading"), true);
      $("d24-ask").disabled = true;
      try {
        var payload = {
          question: question,
          retrieval_top_k: Number($("d24-retrieval-top-k").value) || 20,
          similarity_threshold: Number($("d24-similarity-threshold").value),
          final_top_k: Number($("d24-final-top-k").value) || 5,
        };
        var result = await api("POST", "/ask", payload);
        state.lastResult = result;
        if (result.pipeline && result.pipeline.length) {
          show($("d24-pipeline-panel"), true);
          $("d24-pipeline").textContent = result.pipeline.join("\n↓\n");
        } else {
          show($("d24-pipeline-panel"), false);
        }
        renderResult(result, $("d24-results"));
      } catch (err) {
        setError("d24-ask-error", err.message);
      } finally {
        show($("d24-ask-loading"), false);
        $("d24-ask").disabled = false;
      }
    }
    $("d24-ask").addEventListener("click", ask);

    /* ------------------------------------------------------------------ */
    /* evaluation                                                          */
    /* ------------------------------------------------------------------ */
    function renderEvalSummary() {
      var box = $("d24-eval-summary");
      clear(box);
      var s = state.summary;
      if (!s) { box.textContent = "Нет данных. Запустите оценку."; return; }
      var blocks = [
        ["Questions evaluated", s.evaluated + " / " + s.total_questions],
        ["Sources present", s.sources_present + " / " + s.total_questions],
        ["Quotes present", s.quotes_present + " / " + s.total_questions],
        ["Quotes verified", s.quotes_valid + " / " + s.total_questions],
        ["Grounded answers", s.answered],
        ["Insufficient context", s.insufficient_context],
        ["Grounding failed", s.grounding_failed],
      ];
      blocks.forEach(function (row) {
        var block = make("div", "d22-summary-block");
        block.appendChild(make("b", null, row[0]));
        block.appendChild(make("span", "d22-summary-item", row[1]));
        box.appendChild(block);
      });
      var supported = s.supported || {};
      var supportBlock = make("div", "d22-summary-block");
      supportBlock.appendChild(make("b", null, "Answer supported by evidence"));
      supportBlock.appendChild(make("span", "d22-summary-item",
        "PASS: " + (supported.pass || 0)
        + " · PARTIAL: " + (supported.partial || 0)
        + " · FAIL: " + (supported.fail || 0)));
      box.appendChild(supportBlock);
    }

    function renderEvalTable() {
      var tbody = $("d24-eval-table-body");
      clear(tbody);
      state.questions.forEach(function (question) {
        var record = state.results[question.id] || {};
        var tr = make("tr");

        tr.appendChild(make("td", "d22-eval-id", question.id));
        var qcell = make("td", "d22-eval-question");
        qcell.appendChild(make("div", null, question.question));
        if (question.expected_sources && question.expected_sources.length) {
          question.expected_sources.forEach(function (src) {
            qcell.appendChild(make("span", "d22-expected-source", formatExpected(src)));
          });
        }
        tr.appendChild(qcell);

        tr.appendChild(make("td", null, boolText(record.has_sources)));
        tr.appendChild(make("td", null, boolText(record.has_quotes)));
        tr.appendChild(make("td", null, boolText(record.all_quotes_valid)));

        var gradeCell = make("td");
        var select = make("select", "d24-grade-select");
        [["", "—"], ["pass", "PASS"], ["partial", "PARTIAL"], ["fail", "FAIL"]].forEach(function (opt) {
          var o = document.createElement("option");
          o.value = opt[0];
          o.textContent = opt[1];
          select.appendChild(o);
        });
        select.value = record.answer_supported_by_evidence || "";
        select.addEventListener("change", function () {
          saveGrade(question.id, select.value || null);
        });
        gradeCell.appendChild(select);
        tr.appendChild(gradeCell);

        var actionCell = make("td", "d22-eval-actions");
        var runBtn = make("button", "link-btn", "Run");
        runBtn.type = "button";
        runBtn.addEventListener("click", function () { runOne(question.id); });
        actionCell.appendChild(runBtn);
        tr.appendChild(actionCell);
        tbody.appendChild(tr);
      });
    }

    function formatExpected(src) {
      if (src.source_type === "telegram") {
        return "telegram: " + asList(src.message_ids).join(", ");
      }
      var text = src.source || src.source_type;
      if (src.page !== null && src.page !== undefined) text += ", p." + src.page;
      return text;
    }
    function boolText(value) {
      if (value === true) return "YES";
      if (value === false) return "NO";
      return "—";
    }

    async function loadEvaluation() {
      try {
        state.questions = await api("GET", "/evaluation/questions");
        var results = await api("GET", "/evaluation/results");
        state.results = results.results || {};
        state.summary = results.summary;
        state.evalLoaded = true;
        renderEvalSummary();
        renderEvalTable();
      } catch (err) {
        setError("d24-eval-error", err.message);
      }
    }

    async function saveGrade(questionId, value) {
      setError("d24-eval-error", "");
      try {
        var results = await api(
          "PUT",
          "/evaluation/result/" + encodeURIComponent(questionId),
          { answer_supported_by_evidence: value }
        );
        state.results = results.results || {};
        state.summary = results.summary;
        renderEvalSummary();
      } catch (err) {
        setError("d24-eval-error", err.message);
      }
    }

    async function runOne(questionId) {
      setError("d24-eval-error", "");
      show($("d24-eval-loading"), true);
      try {
        var payload = settingsPayload();
        var run = await api(
          "POST",
          "/evaluation/run/" + encodeURIComponent(questionId),
          payload
        );
        state.results[questionId] = Object.assign({}, state.results[questionId], run.checks);
        renderEvalTable();
        var detail = $("d24-eval-detail");
        clear(detail);
        detail.appendChild(make("h3", null, "Question " + questionId));
        renderResult(run.result, detail);
        var results = await api("GET", "/evaluation/results");
        state.results = results.results || {};
        state.summary = results.summary;
        renderEvalSummary();
        renderEvalTable();
      } catch (err) {
        setError("d24-eval-error", err.message);
      } finally {
        show($("d24-eval-loading"), false);
      }
    }

    async function runAll() {
      setError("d24-eval-error", "");
      show($("d24-eval-loading"), true);
      $("d24-eval-run-all").disabled = true;
      try {
        var response = await api("POST", "/evaluation/run-all", settingsPayload());
        if (response.summary) {
          state.summary = response.summary;
        }
        var results = await api("GET", "/evaluation/results");
        state.results = results.results || {};
        state.summary = results.summary;
        renderEvalSummary();
        renderEvalTable();
      } catch (err) {
        setError("d24-eval-error", err.message);
      } finally {
        show($("d24-eval-loading"), false);
        $("d24-eval-run-all").disabled = false;
      }
    }
    $("d24-eval-run-all").addEventListener("click", runAll);
    $("d24-eval-refresh").addEventListener("click", loadEvaluation);

    /* ------------------------------------------------------------------ */
    /* negative / refusal                                                  */
    /* ------------------------------------------------------------------ */
    function renderNegativeSummary(report) {
      var box = $("d24-negative-summary");
      clear(box);
      if (!report) { box.textContent = "Нет данных. Запустите проверку."; return; }
      var blocks = [
        ["Negative questions", report.questions],
        ["Correctly refused", report.correctly_refused + " / " + report.questions],
        ["Incorrectly answered", report.incorrectly_answered + " / " + report.questions],
        ["LLM invoked", report.llm_invoked],
      ];
      blocks.forEach(function (row) {
        var block = make("div", "d22-summary-block");
        block.appendChild(make("b", null, row[0]));
        block.appendChild(make("span", "d22-summary-item", row[1]));
        box.appendChild(block);
      });
      renderNegativeTable(report.runs || []);
    }

    function renderNegativeTable(runs) {
      var tbody = $("d24-negative-table-body");
      clear(tbody);
      runs.forEach(function (run) {
        var tr = make("tr");
        tr.appendChild(make("td", "d22-eval-id", run.question.id));
        tr.appendChild(make("td", "d22-eval-question", run.question.question));
        tr.appendChild(make("td", null, run.expected_status));
        tr.appendChild(make("td", null, run.actual_status));
        tr.appendChild(make("td", null, run.llm_called ? "YES" : "NO"));
        tr.appendChild(make("td", null, formatScore(run.best_score)));
        tr.appendChild(make("td", null, run.passed ? "PASS" : "FAIL"));
        tr.appendChild(make("td", "d24-observed", run.question.note || ""));
        tbody.appendChild(tr);
      });
    }

    async function runNegative() {
      setError("d24-negative-error", "");
      show($("d24-negative-loading"), true);
      $("d24-negative-run-all").disabled = true;
      try {
        var report = await api("POST", "/negative/run-all", settingsPayload());
        state.negativeLoaded = true;
        renderNegativeSummary(report);
      } catch (err) {
        setError("d24-negative-error", err.message);
      } finally {
        show($("d24-negative-loading"), false);
        $("d24-negative-run-all").disabled = false;
      }
    }

    async function loadNegative() {
      try {
        var report = await api("GET", "/negative/report" + settingsQuery());
        state.negativeLoaded = true;
        renderNegativeSummary(report);
      } catch (err) {
        setError("d24-negative-error", err.message);
      }
    }
    $("d24-negative-run-all").addEventListener("click", runNegative);
    $("d24-negative-refresh").addEventListener("click", loadNegative);

    function settingsPayload() {
      return {
        retrieval_top_k: Number($("d24-retrieval-top-k").value) || 20,
        similarity_threshold: Number($("d24-similarity-threshold").value),
        final_top_k: Number($("d24-final-top-k").value) || 5,
      };
    }
    function settingsQuery() {
      var p = settingsPayload();
      return "?retrieval_top_k=" + p.retrieval_top_k
        + "&similarity_threshold=" + p.similarity_threshold
        + "&final_top_k=" + p.final_top_k;
    }

    /* ------------------------------------------------------------------ */
    /* lazy loading on main-tab activation                                 */
    /* ------------------------------------------------------------------ */
    document.addEventListener("llmtabchange", function (event) {
      if (event && event.detail && event.detail.tab === "day24") {
        if (!state.status) loadStatus();
      }
    });

    loadStatus();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay24);
  } else {
    initDay24();
  }
})();
