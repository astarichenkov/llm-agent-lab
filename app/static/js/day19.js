/* LLM Agent Lab — Day 19: MCP tool composition pipeline.
 *
 * Vanilla JS. Talks to POST /api/week4/day19/pipeline. The backend owns the
 * whole chain: it discovers the real Pipeline MCP tools, performs three real
 * MCP `tools/call` requests in order (search_logs -> analyze_logs ->
 * save_report) and passes the actual data between them. This file only renders
 * the returned answer, the pipeline steps, the artifacts and the REAL trace.
 *
 * Raw stage logs are NEVER rendered: only counts, the structured analysis and
 * the artifact previews (analysis.md / metadata.json) are shown.
 */
(function () {
  "use strict";

  var MAIN_STEPS = ["search_logs", "analyze_logs", "save_report"];
  var STEP_LABELS = {
    search_logs: "search_logs",
    analyze_logs: "analyze_logs",
    save_report: "save_report"
  };

  function initDay19() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day19",
      "d19-service", "d19-level", "d19-since", "d19-limit",
      "d19-question", "d19-text", "d19-mask", "d19-run", "d19-loading", "d19-error",
      "d19-answer-section", "d19-answer",
      "d19-steps-section", "d19-steps",
      "d19-summary-section", "d19-logs-received", "d19-logs-analyzed",
      "d19-groups", "d19-analysis-summary",
      "d19-artifacts-section", "d19-artifacts",
      "d19-preview-analysis", "d19-preview-metadata", "d19-artifact-preview",
      "d19-trace-section", "d19-trace",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day19: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var inFlight = false;
    var currentArtifacts = null;

    function show(el, on) { el.classList.toggle("hidden", !on); }
    function clear(el) { el.innerHTML = ""; el.textContent = ""; }
    function setError(message) {
      var el = $("d19-error");
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }

    function outcomeFor(trace, name) {
      var found = null;
      (trace || []).forEach(function (step) {
        if (step.step === name && step.status !== "started") { found = step; }
      });
      return found;
    }

    function statusIcon(status) {
      if (status === "ok") return "✓";
      if (status === "error") return "✗";
      if (status === "skipped") return "-";
      return "…";
    }

    function stepDetail(name, step) {
      if (!step) return "";
      var details = step.details || {};
      if (name === "search_logs") {
        var count = details.logs_received != null ? details.logs_received
          : (details.logs_passed != null ? details.logs_passed : null);
        if (count != null) return count + " logs received";
      }
      if (name === "analyze_logs") {
        if (details.error_groups != null) return details.error_groups + " groups found";
        if (details.logs_passed != null) return details.logs_passed + " logs passed";
      }
      if (name === "save_report") {
        if (details.artifacts && details.artifacts.length) {
          return details.artifacts.length + " artifacts created";
        }
        if (details.run_id) return details.run_id;
      }
      return step.message || "";
    }

    function renderSteps(data) {
      var list = $("d19-steps");
      clear(list);
      show($("d19-steps-section"), true);
      var trace = data.trace || [];
      MAIN_STEPS.forEach(function (name) {
        var step = outcomeFor(trace, name);
        var status = step ? step.status : "skipped";
        var li = document.createElement("li");
        li.className = "d19-step d19-step-" + status;

        var icon = document.createElement("span");
        icon.className = "d19-step-icon";
        icon.textContent = statusIcon(status);
        li.appendChild(icon);

        var body = document.createElement("span");
        body.className = "d19-step-body";
        var title = document.createElement("b");
        title.textContent = STEP_LABELS[name];
        body.appendChild(title);
        var detail = document.createElement("span");
        detail.className = "d19-step-detail";
        detail.textContent = stepDetail(name, step);
        body.appendChild(detail);
        li.appendChild(body);
        list.appendChild(li);
      });
    }

    function renderSummary(data) {
      var search = data.search || {};
      var analysis = data.analysis || {};
      show($("d19-summary-section"), true);
      $("d19-logs-received").textContent =
        search.count != null ? search.count : 0;
      $("d19-logs-analyzed").textContent =
        analysis.logs_analyzed != null ? analysis.logs_analyzed : 0;
      $("d19-groups").textContent =
        analysis.error_groups_count != null ? analysis.error_groups_count : 0;
      $("d19-analysis-summary").textContent = analysis.summary || "";
    }

    function artifactUrl(runId, name) {
      return "/api/week4/day19/runs/" + encodeURIComponent(runId) +
        "/artifacts/" + encodeURIComponent(name);
    }

    function renderArtifacts(data) {
      var artifacts = data.artifacts;
      var list = $("d19-artifacts");
      clear(list);
      if (!artifacts) { show($("d19-artifacts-section"), false); return; }
      currentArtifacts = artifacts;
      show($("d19-artifacts-section"), true);
      ["raw.jsonl", "analysis.md", "metadata.json"].forEach(function (name) {
        var li = document.createElement("li");
        var link = document.createElement("a");
        link.href = artifactUrl(artifacts.run_id, name);
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = name;
        li.appendChild(link);
        var path = document.createElement("span");
        path.className = "d19-artifact-path";
        path.textContent = "  " + artifacts.root + "/" + artifacts.run_id + "/" + name;
        li.appendChild(path);
        list.appendChild(li);
      });
    }

    function renderTrace(steps) {
      var list = $("d19-trace");
      clear(list);
      show($("d19-trace-section"), true);
      (steps || []).forEach(function (step) {
        var li = document.createElement("li");
        li.className = "d16-trace-step d16-trace-" +
          (step.status === "error" ? "error" : "ok");
        var badge = document.createElement("span");
        badge.className = "d16-trace-badge";
        badge.textContent = (step.status || "ok").toUpperCase();
        li.appendChild(badge);
        var message = document.createElement("span");
        message.className = "d16-trace-message";
        message.textContent = step.message;
        li.appendChild(message);
        list.appendChild(li);
      });
    }

    function render(data) {
      data = data || {};
      show($("d19-answer-section"), true);
      $("d19-answer").textContent = data.answer || "";
      renderSteps(data);
      renderSummary(data);
      renderArtifacts(data);
      renderTrace(data.trace);
      if (data.status === "failed") {
        setError(data.error || "Pipeline failed.");
      } else {
        setError("");
      }
    }

    function preview(name) {
      if (!currentArtifacts) return;
      var pre = $("d19-artifact-preview");
      pre.textContent = "Loading…";
      show(pre, true);
      fetch(artifactUrl(currentArtifacts.run_id, name), {
        headers: { "Accept": "text/plain" }
      })
        .then(function (response) {
          if (!response.ok) { throw new Error("Preview failed (" + response.status + ")"); }
          return response.text();
        })
        .then(function (text) { pre.textContent = text; })
        .catch(function (err) { pre.textContent = err.message || "Preview failed"; });
    }

    function runPipeline() {
      if (inFlight) return;
      var service = ($("d19-service").value || "").trim();
      if (!service) { setError("Укажите service."); return; }
      var limit = parseInt($("d19-limit").value, 10);
      if (!Number.isInteger(limit) || limit < 1 || limit > 500) {
        setError("Limit: целое число от 1 до 500."); return;
      }
      var question = ($("d19-question").value || "").trim();
      if (!question) { setError("Введите вопрос для анализа."); return; }

      var payload = {
        service: service,
        since_minutes: parseInt($("d19-since").value, 10),
        limit: limit,
        question: question,
      };
      var level = ($("d19-level").value || "").trim();
      if (level) payload.level = level;
      var text = ($("d19-text").value || "").trim();
      if (text) payload.text_contains = text;
      if ($("d19-mask").checked) payload.mask_data = true;

      inFlight = true;
      setError("");
      $("d19-loading").classList.remove("hidden");
      $("d19-run").disabled = true;
      show($("d19-answer-section"), false);
      show($("d19-steps-section"), false);
      show($("d19-summary-section"), false);
      show($("d19-artifacts-section"), false);
      show($("d19-trace-section"), false);
      show($("d19-artifact-preview"), false);

      fetch("/api/week4/day19/pipeline", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify(payload),
      })
        .then(function (response) {
          return response.json().catch(function () { return {}; })
            .then(function (data) {
              return { ok: response.ok, status: response.status, data: data };
            });
        })
        .then(function (res) {
          if (!res.ok) {
            var detail = res.data && res.data.detail;
            if (Array.isArray(detail)) {
              detail = detail.map(function (d) { return d.msg; }).join("; ");
            }
            throw new Error(detail || ("Request failed (" + res.status + ")"));
          }
          render(res.data);
        })
        .catch(function (err) {
          setError(err.message || "Day 19 pipeline request failed.");
        })
        .finally(function () {
          inFlight = false;
          $("d19-loading").classList.add("hidden");
          $("d19-run").disabled = false;
        });
    }

    $("d19-run").addEventListener("click", runPipeline);
    $("d19-preview-analysis").addEventListener("click", function () {
      preview("analysis.md");
    });
    $("d19-preview-metadata").addEventListener("click", function () {
      preview("metadata.json");
    });

    setError("");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay19);
  } else {
    initDay19();
  }
})();
