/* LLM Agent Lab — Day 20: multi-server MCP orchestration.
 *
 * Vanilla JS. Talks to:
 *   - GET  /api/week4/day20/mcp/status      (registered servers + routing)
 *   - POST /api/week4/day20/investigate     (LLM-driven investigation)
 *
 * The backend owns MCP discovery and the agent loop. This file only renders
 * what the backend returned: the servers, the tool -> server routing table,
 * the REAL orchestration trace, the final answer and the saved report.
 *
 * No secrets are ever rendered here (there are none in the API responses).
 */
(function () {
  "use strict";

  function initDay20() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day20",
      "d20-servers", "d20-routing", "d20-refresh", "d20-loading", "d20-error",
      "d20-message", "d20-service", "d20-since", "d20-save", "d20-mask",
      "d20-run", "d20-run-loading", "d20-run-error",
      "d20-answer-section", "d20-status", "d20-id", "d20-iterations",
      "d20-toolcalls", "d20-answer",
      "d20-trace-section", "d20-trace",
      "d20-artifacts-section", "d20-artifacts",
      "d20-preview-investigation", "d20-preview-metadata", "d20-artifact-preview",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day20: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var discovering = false;
    var investigating = false;
    var currentArtifacts = null;

    function show(el, on) { el.classList.toggle("hidden", !on); }
    function clear(el) { el.innerHTML = ""; el.textContent = ""; }
    function setError(id, message) {
      var el = $(id);
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }
    function statusIcon(status) {
      if (status === "ok") return "✓";
      if (status === "error") return "✗";
      if (status === "timeout") return "⏱";
      if (status === "skipped") return "-";
      return "…";
    }

    /* ---------------- MCP servers + routing ---------------- */
    function renderServers(servers) {
      var container = $("d20-servers");
      clear(container);
      (servers || []).forEach(function (server) {
        var card = document.createElement("article");
        card.className = "d20-server-card " +
          (server.connected ? "d20-server-up" : "d20-server-down");

        var dot = document.createElement("span");
        dot.className = "d20-server-dot";
        dot.textContent = server.connected ? "●" : "○";
        card.appendChild(dot);

        var body = document.createElement("div");
        body.className = "d20-server-body";

        var title = document.createElement("b");
        title.textContent = server.label || server.name;
        body.appendChild(title);

        var meta = document.createElement("span");
        meta.className = "d20-server-meta";
        var count = server.tools_count != null ? server.tools_count : 0;
        meta.textContent = "  " + count + (count === 1 ? " tool" : " tools") +
          " · " + (server.transport || "stdio");
        body.appendChild(meta);

        var tools = document.createElement("div");
        tools.className = "d20-server-tools";
        tools.textContent = (server.tools || []).map(function (tool) {
          return tool.name;
        }).join(", ");
        body.appendChild(tools);

        if (!server.connected && server.error) {
          var err = document.createElement("div");
          err.className = "d20-server-error";
          err.textContent = server.error;
          body.appendChild(err);
        }

        card.appendChild(body);
        container.appendChild(card);
      });
    }

    function renderRouting(routes) {
      var list = $("d20-routing");
      clear(list);
      (routes || []).forEach(function (route) {
        var li = document.createElement("li");
        li.className = "d20-route";

        var tool = document.createElement("code");
        tool.className = "d20-route-tool";
        tool.textContent = route.tool;
        li.appendChild(tool);

        var arrow = document.createElement("span");
        arrow.className = "d20-route-arrow";
        arrow.textContent = "→";
        li.appendChild(arrow);

        var server = document.createElement("span");
        server.className = "d20-route-server";
        server.textContent = route.label || route.server;
        li.appendChild(server);

        list.appendChild(li);
      });
    }

    function discover() {
      if (discovering) return;
      discovering = true;
      setError("d20-error", "");
      $("d20-loading").classList.remove("hidden");
      $("d20-refresh").disabled = true;

      fetch("/api/week4/day20/mcp/status", {
        headers: { "Accept": "application/json" },
      })
        .then(function (response) {
          return response.json().catch(function () { return {}; })
            .then(function (data) {
              return { ok: response.ok, status: response.status, data: data };
            });
        })
        .then(function (res) {
          if (!res.ok) {
            throw new Error(
              (res.data && res.data.detail) ||
              ("Request failed (" + res.status + ")")
            );
          }
          renderServers(res.data.servers);
          renderRouting(res.data.routing);
          if (res.data.error) { setError("d20-error", res.data.error); }
        })
        .catch(function (err) {
          renderServers([]);
          renderRouting([]);
          setError("d20-error", err.message || "MCP discovery failed.");
        })
        .finally(function () {
          discovering = false;
          $("d20-loading").classList.add("hidden");
          $("d20-refresh").disabled = false;
        });
    }

    /* ---------------- Investigation ---------------- */
    function renderTrace(trace) {
      var list = $("d20-trace");
      clear(list);
      show($("d20-trace-section"), true);
      (trace || []).forEach(function (step) {
        var li = document.createElement("li");
        li.className = "d20-trace-step d20-trace-" + (step.status || "ok");

        var icon = document.createElement("span");
        icon.className = "d20-trace-icon";
        icon.textContent = statusIcon(step.status);
        li.appendChild(icon);

        var body = document.createElement("div");
        body.className = "d20-trace-body";

        var title = document.createElement("b");
        var server = step.server_label || step.server;
        if (step.kind === "tool_call") {
          title.textContent = (server ? server + " · " : "") + (step.tool || "tool");
        } else if (step.kind === "llm") {
          title.textContent = "Agent";
        } else if (step.kind === "discovery") {
          title.textContent = (server || "MCP") + " (discovery)";
        } else {
          title.textContent = "Agent";
        }
        body.appendChild(title);

        if (step.result_summary) {
          var summary = document.createElement("span");
          summary.className = "d20-trace-summary";
          summary.textContent = "  " + step.result_summary;
          body.appendChild(summary);
        }
        if (step.reason) {
          var reason = document.createElement("div");
          reason.className = "d20-trace-reason";
          reason.textContent = step.reason;
          body.appendChild(reason);
        }
        if (step.duration_ms != null) {
          var duration = document.createElement("span");
          duration.className = "d20-trace-duration";
          duration.textContent = "  " + step.duration_ms + " ms";
          body.appendChild(duration);
        }
        if (step.error) {
          var error = document.createElement("div");
          error.className = "d20-trace-error";
          error.textContent = step.error;
          body.appendChild(error);
        }

        li.appendChild(body);
        list.appendChild(li);
      });
    }

    function artifactUrl(investigationId, name) {
      return "/api/week4/day20/investigations/" +
        encodeURIComponent(investigationId) +
        "/artifacts/" + encodeURIComponent(name);
    }

    function renderArtifacts(data) {
      var artifacts = data.artifacts;
      var list = $("d20-artifacts");
      clear(list);
      if (!artifacts) {
        show($("d20-artifacts-section"), false);
        currentArtifacts = null;
        return;
      }
      currentArtifacts = artifacts;
      show($("d20-artifacts-section"), true);
      [artifacts.investigation || "investigation.md",
       artifacts.metadata || "metadata.json"].forEach(function (name) {
        var li = document.createElement("li");
        var link = document.createElement("a");
        link.href = artifactUrl(artifacts.investigation_id, name);
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = name;
        li.appendChild(link);
        var path = document.createElement("span");
        path.className = "d19-artifact-path";
        path.textContent = "  " + artifacts.root + "/" +
          artifacts.investigation_id + "/" + name;
        li.appendChild(path);
        list.appendChild(li);
      });
    }

    function renderResult(data) {
      data = data || {};
      show($("d20-answer-section"), true);
      $("d20-status").textContent = data.status || "—";
      $("d20-id").textContent = data.investigation_id || "—";
      $("d20-iterations").textContent =
        data.iterations != null ? data.iterations : 0;
      $("d20-toolcalls").textContent =
        data.tool_calls ? data.tool_calls.length : 0;
      $("d20-answer").textContent = data.answer || "";
      renderTrace(data.trace);
      renderArtifacts(data);
      if (data.status === "failed") {
        setError("d20-run-error", data.error || "Investigation failed.");
      } else if (data.error) {
        setError("d20-run-error", data.error);
      } else {
        setError("d20-run-error", "");
      }
    }

    function preview(name) {
      if (!currentArtifacts) return;
      var pre = $("d20-artifact-preview");
      pre.textContent = "Loading…";
      show(pre, true);
      fetch(artifactUrl(currentArtifacts.investigation_id, name), {
        headers: { "Accept": "text/plain" },
      })
        .then(function (response) {
          if (!response.ok) {
            throw new Error("Preview failed (" + response.status + ")");
          }
          return response.text();
        })
        .then(function (text) { pre.textContent = text; })
        .catch(function (err) {
          pre.textContent = err.message || "Preview failed";
        });
    }

    function investigate() {
      if (investigating) return;
      var message = ($("d20-message").value || "").trim();
      if (!message) { setError("d20-run-error", "Введите запрос."); return; }

      var payload = {
        message: message,
        since_minutes: parseInt($("d20-since").value, 10),
        save_report: $("d20-save").checked,
      };
      if ($("d20-mask").checked) payload.mask_data = true;
      var service = ($("d20-service").value || "").trim();
      if (service) payload.service = service;

      investigating = true;
      setError("d20-run-error", "");
      $("d20-run-loading").classList.remove("hidden");
      $("d20-run").disabled = true;
      show($("d20-answer-section"), false);
      show($("d20-trace-section"), false);
      show($("d20-artifacts-section"), false);
      show($("d20-artifact-preview"), false);

      fetch("/api/week4/day20/investigate", {
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
          renderResult(res.data);
        })
        .catch(function (err) {
          setError("d20-run-error", err.message || "Investigation request failed.");
        })
        .finally(function () {
          investigating = false;
          $("d20-run-loading").classList.add("hidden");
          $("d20-run").disabled = false;
        });
    }

    $("d20-refresh").addEventListener("click", discover);
    $("d20-run").addEventListener("click", investigate);
    $("d20-preview-investigation").addEventListener("click", function () {
      preview("investigation.md");
    });
    $("d20-preview-metadata").addEventListener("click", function () {
      preview("metadata.json");
    });
    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day20") { discover(); }
    });

    setError("d20-error", "");
    setError("d20-run-error", "");
    discover();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay20);
  } else {
    initDay20();
  }
})();
