/* LLM Agent Lab — Day 16: MCP connection & tool discovery.
 *
 * Vanilla JS. Talks to GET /api/week4/day16/mcp/status. The backend owns the
 * MCP transport: every refresh really spawns the local Demo MCP server over
 * stdio, initializes a session and calls tools/list. This file only renders
 * whatever the backend returned — it never hardcodes the tool list.
 */
(function () {
  "use strict";

  function initDay16() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "panel-day16",
      "d16-status", "d16-transport", "d16-server", "d16-tools-count",
      "d16-refresh", "d16-loading", "d16-error",
      "d16-tools", "d16-tools-empty", "d16-trace",
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day16: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var inFlight = false;
    var STATUS_CLASSES = "d16-status-connected d16-status-connecting d16-status-failed";

    function setStatus(text, className) {
      var el = $("d16-status");
      el.textContent = text;
      el.className = "d16-status-value " + (className || "");
    }
    function setError(message) {
      var el = $("d16-error");
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }
    function setLoading(on) {
      inFlight = on;
      $("d16-loading").classList.toggle("hidden", !on);
      $("d16-refresh").disabled = on;
    }

    function formatSchema(schema) {
      if (!schema || !schema.properties ||
          Object.keys(schema.properties).length === 0) {
        return "(no parameters)";
      }
      var requiredKeys = schema.required || [];
      return Object.keys(schema.properties).map(function (key) {
        var prop = schema.properties[key] || {};
        var type = prop.type || "any";
        var req = requiredKeys.indexOf(key) !== -1 ? ", required" : "";
        return key + ": " + type + req;
      }).join("\n");
    }

    function clear(el) {
      el.innerHTML = "";
      el.textContent = "";
    }

    function renderTools(tools) {
      var container = $("d16-tools");
      clear(container);
      tools = tools || [];
      if (tools.length === 0) {
        $("d16-tools-empty").classList.remove("hidden");
        return;
      }
      $("d16-tools-empty").classList.add("hidden");
      tools.forEach(function (tool) {
        var card = document.createElement("article");
        card.className = "d16-tool-card";

        var name = document.createElement("h3");
        name.className = "d16-tool-name";
        name.textContent = tool.name;
        card.appendChild(name);

        var desc = document.createElement("p");
        desc.className = "d16-tool-desc";
        desc.textContent = tool.description || "(no description)";
        card.appendChild(desc);

        var label = document.createElement("p");
        label.className = "d16-tool-schema-label";
        label.textContent = "Input schema:";
        card.appendChild(label);

        var pre = document.createElement("pre");
        pre.className = "d16-tool-schema";
        pre.textContent = formatSchema(tool.input_schema);
        card.appendChild(pre);

        container.appendChild(card);
      });
    }

    function renderTrace(steps) {
      var list = $("d16-trace");
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

    function refresh() {
      if (inFlight) return;
      setLoading(true);
      setError("");
      setStatus("Connecting…", "d16-status-connecting");

      fetch("/api/week4/day16/mcp/status", {
        headers: { "Accept": "application/json" },
      })
        .then(function (response) {
          return response.json().then(function (data) {
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
          var data = res.data || {};
          $("d16-transport").textContent =
            (data.server && data.server.transport) || "stdio";
          $("d16-server").textContent =
            (data.server && data.server.name) || "—";
          $("d16-tools-count").textContent =
            data.tools_count != null ? data.tools_count : 0;
          renderTools(data.tools);
          renderTrace(data.trace);
          if (data.connected) {
            setStatus("Connected", "d16-status-connected");
            setError("");
          } else {
            setStatus("Connection failed", "d16-status-failed");
            setError(data.error || "MCP connection failed.");
          }
        })
        .catch(function (err) {
          setStatus("Connection failed", "d16-status-failed");
          renderTools([]);
          renderTrace([]);
          setError(err.message || "MCP connection failed.");
        })
        .finally(function () {
          setLoading(false);
        });
    }

    $("d16-refresh").addEventListener("click", refresh);
    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day16") {
        refresh();
      }
    });

    refresh();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay16);
  } else {
    initDay16();
  }
})();
