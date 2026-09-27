/* LLM Agent Lab — Day 18: scheduled monitoring dashboard.
 *
 * Vanilla JS. Two independent flows share the same backend MonitoringService:
 *
 *  - the AGENT flow (POST /api/week4/day18/chat): the LLM selects a monitoring
 *    MCP tool; this file only renders the answer / real tool call / trace;
 *  - the DASHBOARD flow (REST): start / status / runs / summary / stop.
 *
 * Frontend polling only refreshes the DISPLAY; the actual scheduling happens
 * in the backend. Exactly one timer exists at a time.
 */
(function () {
  "use strict";

  var INTERVAL_LABELS = {
    10: "10 sec", 30: "30 sec", 60: "1 min",
    300: "5 min", 600: "10 min", 1800: "30 min", 3600: "1 hour"
  };
  var LOOKBACK_LABELS = {
    1: "1 min", 5: "5 min", 10: "10 min", 15: "15 min", 30: "30 min", 60: "1 hour"
  };
  var POLL_MS = 5000;

  function initDay18() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day18",
      "d18-agent-message", "d18-agent-send", "d18-agent-loading", "d18-error",
      "d18-answer-section", "d18-answer",
      "d18-toolcall-section", "d18-toolcall-server", "d18-toolcall-tool",
      "d18-toolcall-jobid", "d18-toolcall-args",
      "d18-service", "d18-level", "d18-interval", "d18-lookback",
      "d18-limit", "d18-text", "d18-start", "d18-loading",
      "d18-job-section", "d18-job-id", "d18-job-status", "d18-job-service",
      "d18-job-level", "d18-job-interval", "d18-job-lookback", "d18-job-runs",
      "d18-job-last", "d18-job-next", "d18-refresh", "d18-stop",
      "d18-summary-section", "d18-sum-runs", "d18-sum-success",
      "d18-sum-failed", "d18-sum-total", "d18-sum-avg", "d18-sum-max",
      "d18-sum-last", "d18-sum-trend",
      "d18-runs-section", "d18-runs-body",
      "d18-trace-section", "d18-trace",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day18: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var currentJobId = null;
    var currentStatus = null;
    var pollTimer = null;
    var agentInFlight = false;
    var startInFlight = false;

    function show(el, on) { el.classList.toggle("hidden", !on); }
    function clear(el) { el.innerHTML = ""; el.textContent = ""; }
    function setError(message) {
      var el = $("d18-error");
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }
    function intervalLabel(seconds) {
      return INTERVAL_LABELS[seconds] || (seconds + " sec");
    }
    function lookbackLabel(minutes) {
      return LOOKBACK_LABELS[minutes] || (minutes + " min");
    }
    function formatTime(value) {
      if (!value) return "—";
      var date = new Date(value);
      if (isNaN(date.getTime())) return String(value);
      var pad = function (n) { return (n < 10 ? "0" : "") + n; };
      return pad(date.getHours()) + ":" + pad(date.getMinutes()) + ":" + pad(date.getSeconds());
    }
    function formatDuration(ms) {
      if (ms == null) return "—";
      return Math.max(0, Math.round(ms)) + " ms";
    }

    function stopPolling() {
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }
    function startPolling() {
      stopPolling();
      pollTimer = setInterval(function () {
        if (!currentJobId) { stopPolling(); return; }
        if (currentStatus && currentStatus.status !== "active") {
          stopPolling(); return;
        }
        refreshDashboard(true);
      }, POLL_MS);
    }

    function jsonFetch(url, options) {
      return fetch(url, options).then(function (response) {
        return response.json().catch(function () { return {}; })
          .then(function (data) { return { ok: response.ok, status: response.status, data: data }; });
      });
    }

    /* ---------------- Agent flow ---------------- */
    function renderTrace(steps) {
      var list = $("d18-trace");
      clear(list);
      (steps || []).forEach(function (step) {
        var li = document.createElement("li");
        li.className = "d16-trace-step d16-trace-" + (step.status || "ok");
        var badge = document.createElement("span");
        badge.className = "d16-trace-badge";
        badge.textContent = step.status === "error" ? "ERROR" : "OK";
        li.appendChild(badge);
        var message = document.createElement("span");
        message.className = "d16-trace-message";
        message.textContent = step.message;
        li.appendChild(message);
        list.appendChild(li);
      });
    }

    function renderToolCall(toolCalls) {
      toolCalls = toolCalls || [];
      if (!toolCalls.length) { show($("d18-toolcall-section"), false); return; }
      show($("d18-toolcall-section"), true);
      var first = toolCalls[0];
      $("d18-toolcall-server").textContent = first.server || "monitoring";
      $("d18-toolcall-tool").textContent = first.tool || "—";
      $("d18-toolcall-args").textContent = JSON.stringify(first.arguments || {}, null, 2);
      var summary = first.result_summary || {};
      $("d18-toolcall-jobid").textContent = summary.job_id || "—";

      // If the agent started a job, load its dashboard right away.
      if (first.tool === "start_monitoring" && summary.job_id) {
        currentJobId = summary.job_id;
        refreshDashboard(false);
      } else if (summary.job_id) {
        currentJobId = summary.job_id;
        refreshDashboard(false);
      }
    }

    function sendAgent() {
      if (agentInFlight) return;
      var message = ($("d18-agent-message").value || "").trim();
      if (!message) { setError("Введите сообщение."); return; }
      agentInFlight = true;
      setError("");
      $("d18-agent-loading").classList.remove("hidden");
      $("d18-agent-send").disabled = true;
      show($("d18-answer-section"), false);
      show($("d18-toolcall-section"), false);
      show($("d18-trace-section"), false);

      jsonFetch("/api/week4/day18/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ message: message }),
      }).then(function (res) {
        if (!res.ok) {
          throw new Error((res.data && res.data.detail) || ("Request failed (" + res.status + ")"));
        }
        var data = res.data || {};
        show($("d18-answer-section"), true);
        $("d18-answer").textContent = data.answer || "";
        renderToolCall(data.tool_calls);
        show($("d18-trace-section"), true);
        renderTrace(data.trace);
        if (data.error) { setError(data.error); }
      }).catch(function (err) {
        show($("d18-trace-section"), false);
        setError(err.message || "Day 18 agent request failed.");
      }).finally(function () {
        agentInFlight = false;
        $("d18-agent-loading").classList.add("hidden");
        $("d18-agent-send").disabled = false;
        // The agent may have started/stopped a job: always reload the
        // dashboard so it reflects it without pressing Start.
        loadLatestJob(true);
      });
    }

    /* ---------------- Dashboard flow ---------------- */
    function selectLatestJob(jobs) {
      jobs = jobs || [];
      var activeJob = null;
      for (var i = 0; i < jobs.length; i++) {
        if (jobs[i] && jobs[i].status === "active") { activeJob = jobs[i]; break; }
      }
      return activeJob || jobs[0] || null;
    }

    function loadLatestJob(silent) {
      // Restore the dashboard after a page refresh (or after the agent created
      // a job through Send) by loading the most recent / active job.
      return jsonFetch("/api/week4/day18/monitoring").then(function (res) {
        if (!res.ok) return;
        var chosen = selectLatestJob(res.data);
        if (!chosen || !chosen.job_id) return;
        currentJobId = chosen.job_id;
        refreshDashboard(silent !== false);
      }).catch(function () { /* dashboard stays empty */ });
    }

    function renderJob(status) {
      currentStatus = status;
      show($("d18-job-section"), true);
      $("d18-job-id").textContent = status.job_id || "—";
      var badge = $("d18-job-status");
      var state = status.status || "active";
      badge.textContent = state;
      badge.className = "d18-badge d18-status-" + state;
      $("d18-job-service").textContent = status.service || "—";
      $("d18-job-level").textContent = status.level || "ANY";
      $("d18-job-interval").textContent = intervalLabel(status.interval_seconds);
      $("d18-job-lookback").textContent = lookbackLabel(status.lookback_minutes);
      $("d18-job-runs").textContent = status.runs_count != null ? status.runs_count : 0;
      $("d18-job-last").textContent = formatTime(status.last_run_at);
      $("d18-job-next").textContent = formatTime(status.next_run_at);
      $("d18-stop").disabled = state === "stopped";
      if (state === "active") { startPolling(); } else { stopPolling(); }
    }

    function renderSummary(summary) {
      if (!summary) { show($("d18-summary-section"), false); return; }
      show($("d18-summary-section"), true);
      $("d18-sum-runs").textContent = summary.runs != null ? summary.runs : 0;
      $("d18-sum-success").textContent = summary.successful_runs != null ? summary.successful_runs : 0;
      $("d18-sum-failed").textContent = summary.failed_runs != null ? summary.failed_runs : 0;
      $("d18-sum-total").textContent = summary.total_logs != null ? summary.total_logs : 0;
      $("d18-sum-avg").textContent =
        summary.average_logs_per_run != null ? summary.average_logs_per_run : 0;
      $("d18-sum-max").textContent = summary.max_logs_per_run != null ? summary.max_logs_per_run : 0;
      $("d18-sum-last").textContent = summary.last_run_logs != null ? summary.last_run_logs : 0;
      $("d18-sum-trend").textContent = summary.trend || "stable";
    }

    function renderRuns(runs) {
      var body = $("d18-runs-body");
      clear(body);
      runs = runs || [];
      if (!runs.length) { show($("d18-runs-section"), false); return; }
      show($("d18-runs-section"), true);
      runs.forEach(function (run, index) {
        var tr = document.createElement("tr");
        var num = document.createElement("td");
        num.textContent = run.id != null ? run.id : (runs.length - index);
        tr.appendChild(num);

        var time = document.createElement("td");
        time.textContent = formatTime(run.finished_at || run.started_at);
        tr.appendChild(time);

        var logs = document.createElement("td");
        logs.textContent = run.logs_count != null ? run.logs_count : 0;
        tr.appendChild(logs);

        var duration = document.createElement("td");
        duration.textContent = formatDuration(run.duration_ms);
        tr.appendChild(duration);

        var statusCell = document.createElement("td");
        var ok = run.status === "success";
        statusCell.className = ok ? "d18-run-ok" : "d18-run-error";
        if (ok) {
          statusCell.textContent = "✓";
        } else {
          statusCell.textContent = "✗";
          if (run.error_message) {
            var span = document.createElement("span");
            span.className = "d18-run-error-msg";
            span.textContent = " " + String(run.error_message).slice(0, 80);
            statusCell.appendChild(span);
          }
        }
        tr.appendChild(statusCell);
        body.appendChild(tr);
      });
    }

    function refreshDashboard(silent) {
      if (!currentJobId) return;
      var id = encodeURIComponent(currentJobId);
      Promise.all([
        jsonFetch("/api/week4/day18/monitoring/" + id),
        jsonFetch("/api/week4/day18/monitoring/" + id + "/runs?limit=20"),
        jsonFetch("/api/week4/day18/monitoring/" + id + "/summary?last_runs=20"),
      ]).then(function (results) {
        if (!results[0].ok) {
          if (!silent) setError("Не удалось получить статус job.");
          return;
        }
        renderJob(results[0].data);
        renderRuns(results[1].ok ? results[1].data.runs : []);
        renderSummary(results[2].ok ? results[2].data : null);
      }).catch(function (err) {
        if (!silent) setError(err.message || "Refresh failed.");
      });
    }

    function startMonitoring() {
      if (startInFlight) return;
      var service = ($("d18-service").value || "").trim();
      if (!service) { setError("Укажите service."); return; }
      var limit = parseInt($("d18-limit").value, 10);
      if (!Number.isInteger(limit) || limit < 1 || limit > 500) {
        setError("Limit: целое число от 1 до 500."); return;
      }
      var level = ($("d18-level").value || "").trim();
      var text = ($("d18-text").value || "").trim();
      var payload = {
        service: service,
        interval_seconds: parseInt($("d18-interval").value, 10),
        lookback_minutes: parseInt($("d18-lookback").value, 10),
        limit: limit,
      };
      if (level) payload.level = level;
      if (text) payload.text_contains = text;

      startInFlight = true;
      setError("");
      $("d18-loading").classList.remove("hidden");
      $("d18-start").disabled = true;
      jsonFetch("/api/week4/day18/monitoring", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(payload),
      }).then(function (res) {
        if (!res.ok) {
          var detail = res.data && res.data.detail;
          if (Array.isArray(detail)) { detail = detail.map(function (d) { return d.msg; }).join("; "); }
          throw new Error(detail || ("Request failed (" + res.status + ")"));
        }
        currentJobId = res.data.job_id;
        refreshDashboard(false);
      }).catch(function (err) {
        setError(err.message || "Не удалось создать monitoring job.");
      }).finally(function () {
        startInFlight = false;
        $("d18-loading").classList.add("hidden");
        $("d18-start").disabled = false;
      });
    }

    function stopMonitoring() {
      if (!currentJobId) return;
      jsonFetch(
        "/api/week4/day18/monitoring/" + encodeURIComponent(currentJobId) + "/stop",
        { method: "POST", headers: { "Accept": "application/json" } }
      ).then(function (res) {
        if (!res.ok) { throw new Error((res.data && res.data.detail) || "Stop failed"); }
        stopPolling();
        refreshDashboard(false);
      }).catch(function (err) {
        setError(err.message || "Не удалось остановить job.");
      });
    }

    $("d18-agent-send").addEventListener("click", sendAgent);
    $("d18-start").addEventListener("click", startMonitoring);
    $("d18-refresh").addEventListener("click", function () { refreshDashboard(false); });
    $("d18-stop").addEventListener("click", stopMonitoring);
    $("d18-interval").addEventListener("change", function () {
      // Keep the selected interval visible even before a job exists.
      $("d18-loading").classList.add("hidden");
    });

    // Stop display polling when leaving the Day 18 tab; resume when back.
    document.addEventListener("llmtabchange", function (event) {
      var tab = event && event.detail ? event.detail.tab : null;
      if (tab !== "day18") {
        stopPolling();
      } else {
        // Re-entering the tab: pick up jobs started via the agent.
        loadLatestJob(true);
      }
    });

    setError("");
    // On first load (including a full page refresh) restore the latest job.
    loadLatestJob(true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay18);
  } else {
    initDay18();
  }
})();
