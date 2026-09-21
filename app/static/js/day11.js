/* LLM Agent Lab — Day 11: agent memory layers.
 * Vanilla JS. Talks to /api/day11/*. All memory logic (classifier, storage,
 * context building) lives on the backend; this file only drives the UI and
 * makes the three layers observable.
 */
(function () {
  "use strict";

  function initDay11() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day11", "panel-day11",
      "d11-strategy", "d11-window-input", "d11-use-memory", "d11-classify",
      "d11-include-lt", "d11-include-working",
      "d11-model", "d11-session-id",
      "d11-demo-run", "d11-seed", "d11-new-session",
      "d11-clear-st", "d11-clear-working", "d11-clear-lt",
      "d11-loading", "d11-progress", "d11-note",
      "d11-messages", "d11-input", "d11-send", "d11-error",
      "d11-decision-status", "d11-decision-list", "d11-decision-error",
      "d11-save-st", "d11-save-working", "d11-save-lt", "d11-save-summary",
      "d11-st-count", "d11-st-list",
      "d11-working-count", "d11-working-block", "d11-working-json",
      "d11-lt-count", "d11-lt-block", "d11-lt-json",
      "d11-ctx-summary", "d11-ctx-layers", "d11-ctx-total", "d11-ctx-sent",
      "d11-ctx-dropped", "d11-ctx-dropped-list", "d11-ctx-blocks"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day11: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = { inFlight: false };

    /* ---------------- helpers ---------------- */
    function fmt(n) {
      if (n === null || n === undefined) return "—";
      return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
    }
    function setError(m) {
      $("d11-error").textContent = m || "";
      $("d11-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d11-note").textContent = m || "";
      $("d11-note").classList.toggle("hidden", !m);
    }
    function setProgress(m) { $("d11-progress").textContent = m || "—"; }
    function setLoading(on) {
      state.inFlight = on;
      $("d11-send").disabled = on;
      $("d11-demo-run").disabled = on;
      $("d11-seed").disabled = on;
      $("d11-loading").classList.toggle("hidden", !on);
    }
    function http(path, opts) {
      return fetch(path, opts).then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, status: r.status, data: d };
        });
      });
    }
    function buildBody(text) {
      var body = {
        message: text,
        strategy: $("d11-strategy").value,
        use_memory: $("d11-use-memory").checked,
        classify: $("d11-classify").checked,
        include_long_term: $("d11-include-lt").checked,
        include_working: $("d11-include-working").checked
      };
      var w = parseInt($("d11-window-input").value, 10);
      if (Number.isInteger(w)) body.window_size = w;
      return body;
    }

    /* ---------------- rendering ---------------- */
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
      $("d11-messages").appendChild(bubble);
      $("d11-messages").scrollTop = $("d11-messages").scrollHeight;
    }
    function renderMessages(messages) {
      $("d11-messages").innerHTML = "";
      (messages || []).forEach(function (m) { addBubble(m.role, m.content); });
    }
    function messageList(el, messages) {
      el.innerHTML = "";
      if (!messages || !messages.length) {
        var empty = document.createElement("div");
        empty.className = "d10-msg-empty";
        empty.textContent = "(нет)";
        el.appendChild(empty);
        return;
      }
      messages.forEach(function (m) {
        var row = document.createElement("div");
        row.className = "d10-msg-row";
        var role = document.createElement("span");
        role.className = "d10-msg-role";
        role.textContent = m.role === "user" ? "Вы" : "Агент";
        var text = document.createElement("span");
        text.className = "d10-msg-text";
        text.textContent = m.content;
        row.appendChild(role);
        row.appendChild(text);
        el.appendChild(row);
      });
    }
    function renderMemory(memory) {
      if (!memory) return;
      $("d11-session-id").textContent = memory.session_id || "—";

      // SHORT-TERM
      $("d11-st-count").textContent = fmt(memory.short_term_count);
      messageList($("d11-st-list"), memory.short_term);

      // WORKING
      var working = memory.working || {};
      var wcount =
        (working.goal ? 1 : 0) +
        ((working.constraints || []).length) +
        ((working.requirements || []).length) +
        ((working.decisions || []).length) +
        (Object.keys(working.data || {}).length);
      $("d11-working-count").textContent = fmt(wcount);
      $("d11-working-block").textContent = workingToText(working);
      $("d11-working-json").textContent = JSON.stringify(working, null, 2);

      // LONG-TERM
      var entries = (memory.long_term && memory.long_term.entries) || {};
      var keys = Object.keys(entries);
      $("d11-lt-count").textContent = fmt(keys.length);
      $("d11-lt-block").textContent = keys.length
        ? keys.map(function (k) { return "- " + k + " = " + entries[k]; }).join("\n")
        : "(пусто)";
      $("d11-lt-json").textContent = JSON.stringify(entries, null, 2);

      renderDecision(memory.last_decision);
    }
    function workingToText(working) {
      var lines = [];
      if (working.goal) lines.push("goal = " + working.goal);
      ["constraints", "requirements", "decisions"].forEach(function (key) {
        (working[key] || []).forEach(function (v) {
          lines.push(key.slice(0, -1) + " = " + v);
        });
      });
      var data = working.data || {};
      Object.keys(data).forEach(function (k) { lines.push(k + " = " + data[k]); });
      return lines.length ? lines.join("\n") : "(пусто)";
    }
    function workingDeltaCount(working) {
      var w = working || {};
      var count = w.goal ? 1 : 0;
      count += (w.constraints || []).length;
      count += (w.requirements || []).length;
      count += (w.decisions || []).length;
      count += Object.keys(w.data || {}).length;
      return count;
    }
    function longTermDeltaCount(decision) {
      return Object.keys((decision && decision.long_term_memory) || {}).length;
    }
    function workingSaved(decision) {
      return !!decision && decision.performed && !decision.nothing_to_save &&
        workingDeltaCount(decision.working_memory) > 0;
    }
    function longTermSaved(decision) {
      return !!decision && decision.performed && !decision.nothing_to_save &&
        longTermDeltaCount(decision) > 0;
    }
    function workingSaveText(decision) {
      var w = (decision && decision.working_memory) || {};
      var lines = [];
      if (w.goal) lines.push("goal: " + w.goal);
      ["constraints", "requirements", "decisions"].forEach(function (key) {
        (w[key] || []).forEach(function (v) {
          lines.push(key + ":\n- " + v);
        });
      });
      var data = w.data || {};
      Object.keys(data).forEach(function (k) {
        lines.push(k + ": " + data[k]);
      });
      return "✓ Сохранено:\n" + lines.join("\n");
    }
    function longTermSaveText(decision) {
      var lt = (decision && decision.long_term_memory) || {};
      var lines = Object.keys(lt).map(function (k) { return k + ": " + lt[k]; });
      return "✓ Сохранено:\n" + lines.join("\n");
    }
    function saveSummary(decision) {
      var w = workingSaved(decision);
      var lt = longTermSaved(decision);
      if (w && lt) return "Short-term + Working + Long-term";
      if (w) return "Short-term + Working";
      if (lt) return "Short-term + Long-term";
      return "только Short-term";
    }
    function decisionLines(decision) {
      if (!decision) return ["—"];
      if (decision.error) return ["Ошибка: " + decision.error];
      if (!decision.performed) return ["Классификация не выполнялась."];
      if (decision.nothing_to_save) return ["nothing_to_save — память не изменялась."];
      var lines = [];
      var w = decision.working_memory || {};
      if (w.goal) lines.push("Working: goal = " + w.goal);
      ["constraints", "requirements", "decisions"].forEach(function (key) {
        (w[key] || []).forEach(function (v) {
          lines.push("Working: " + key.slice(0, -1) + " = " + v);
        });
      });
      var data = w.data || {};
      Object.keys(data).forEach(function (k) {
        lines.push("Working: " + k + " = " + data[k]);
      });
      var lt = decision.long_term_memory || {};
      Object.keys(lt).forEach(function (k) {
        lines.push("Long-term: " + k + " = " + lt[k]);
      });
      return lines.length ? lines : ["Память не изменялась."];
    }
    function renderDecision(decision) {
      if (!decision) {
        $("d11-decision-status").textContent = "—";
        $("d11-decision-list").innerHTML = "";
        $("d11-decision-error").classList.add("hidden");
        $("d11-save-st").textContent = "—";
        $("d11-save-working").textContent = "— Ничего";
        $("d11-save-lt").textContent = "— Ничего";
        $("d11-save-summary").textContent = "—";
        return;
      }
      if (decision.error) {
        $("d11-decision-status").textContent = "Ошибка классификации (graceful degradation)";
      } else if (!decision.performed) {
        $("d11-decision-status").textContent = "Классификация не выполнялась";
      } else if (decision.nothing_to_save) {
        $("d11-decision-status").textContent = "nothing_to_save";
      } else {
        $("d11-decision-status").textContent = "Решение принято: память обновлена";
      }
      var box = $("d11-decision-list");
      box.innerHTML = "";
      var lines = ["Short-term: message stored (детерминированно)"].concat(
        decisionLines(decision)
      );
      lines.forEach(function (line) {
        var row = document.createElement("div");
        row.className = "d11-decision-row";
        row.textContent = line;
        box.appendChild(row);
      });
      $("d11-decision-error").textContent = decision.error || "";
      $("d11-decision-error").classList.toggle("hidden", !decision.error);

      $("d11-save-st").textContent = "✓ Сохранено текущее сообщение";
      $("d11-save-working").textContent = workingSaved(decision)
        ? workingSaveText(decision)
        : "— Ничего";
      $("d11-save-lt").textContent = longTermSaved(decision)
        ? longTermSaveText(decision)
        : "— Ничего";
      $("d11-save-summary").textContent = saveSummary(decision);
    }
    function renderContext(ctx, memory) {
      if (!ctx) return;
      $("d11-ctx-summary").textContent =
        "Стратегия: " + ctx.strategy + " · отправлено short-term: " +
        fmt(ctx.sent_short_term_messages) + " · выпало: " + fmt(ctx.dropped_messages);
      $("d11-ctx-total").textContent = fmt(ctx.total_short_term_messages);
      $("d11-ctx-sent").textContent = fmt(ctx.sent_short_term_messages);
      $("d11-ctx-dropped").textContent = fmt(ctx.dropped_messages);
      messageList($("d11-ctx-dropped-list"), ctx.dropped_preview);

      var chips = $("d11-ctx-layers");
      chips.innerHTML = "";
      (ctx.included_layers || []).forEach(function (layer) {
        var chip = document.createElement("span");
        chip.className = "d11-layer-chip";
        chip.textContent = layer;
        chips.appendChild(chip);
      });

      var parts = [];
      parts.push("LONG-TERM block:");
      parts.push(ctx.long_term_block || "(не включён)");
      parts.push("");
      parts.push("WORKING block:");
      parts.push(ctx.working_block || "(не включён)");
      $("d11-ctx-blocks").textContent = parts.join("\n");
    }

    /* ---------------- state loading ---------------- */
    function loadState() {
      return http("/api/day11/state").then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Ошибка состояния.");
        $("d11-model").textContent = res.data.model;
        renderMemory(res.data.memory);
      }).catch(function (err) { setError(err.message); });
    }

    /* ---------------- chat ---------------- */
    function sendMessage(text, opts) {
      opts = opts || {};
      return http("/api/day11/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildBody(text))
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
        var d = res.data;
        renderMemory(d.memory);
        renderContext(d.context, d.memory);
        if (!opts.silent) {
          addBubble("assistant", d.answer);
        }
        if (d.classifier_error) {
          setNote(d.classifier_error);
        }
        return d;
      });
    }
    function onSend() {
      if (state.inFlight) return;
      var text = $("d11-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      setError(""); setNote("");
      addBubble("user", text);
      setLoading(true);
      sendMessage(text).then(function () {
        $("d11-input").value = "";
      }).catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- demo scenario ---------------- */
    function delay(ms) {
      return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }
    function onDemoRun() {
      if (state.inFlight) return;
      setLoading(true); setError(""); setNote("");
      setProgress("Запуск demo-сценария…");
      var scenario = null;
      http("/api/day11/scenario").then(function (res) {
        if (!res.ok) throw new Error("Не удалось получить сценарий.");
        scenario = res.data;
        return http("/api/day11/session", { method: "POST" });
      }).then(function () {
        renderMessages([]);
        var chain = Promise.resolve();
        scenario.messages.forEach(function (text, idx) {
          chain = chain.then(function () {
            setProgress("Demo: " + (idx + 1) + "/" + scenario.messages.length +
              " — " + text.slice(0, 60));
            addBubble("user", text);
            return sendMessage(text, { silent: false });
          });
        });
        return chain.then(function () {
          setNote("Сценарий завершён. Сравните short-term (часть сообщений выпала) " +
            "с working / long-term памятью.");
        });
      }).catch(function (err) {
        setError(err.message || "Сценарий не удалось выполнить.");
      }).finally(function () { setLoading(false); setProgress("—"); });
    }

    /* ---------------- memory management ---------------- */
    function applyStateResponse(res, message) {
      if (!res.ok) throw new Error(res.data.detail || "Ошибка операции.");
      renderMemory(res.data.memory);
      if (message) setNote(message);
      setError("");
    }
    function onNewSession() {
      if (state.inFlight) return;
      http("/api/day11/session", { method: "POST" }).then(function (res) {
        applyStateResponse(res, "Новая сессия: short-term и working пусты, long-term сохранён.");
        renderMessages([]);
        $("d11-ctx-summary").textContent = "—";
        $("d11-ctx-layers").innerHTML = "";
        $("d11-ctx-total").textContent = "0";
        $("d11-ctx-sent").textContent = "0";
        $("d11-ctx-dropped").textContent = "0";
      }).catch(function (err) { setError(err.message); });
    }
    function onSeed() {
      if (state.inFlight) return;
      setLoading(true); setError(""); setNote("");
      http("/api/day11/memory/seed", { method: "POST" }).then(function (res) {
        applyStateResponse(res, "Демо-память установлена без вызова API.");
        renderMessages(res.data.memory.short_term);
      }).catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }
    function onClear(path, label) {
      if (state.inFlight) return;
      http(path, { method: "DELETE" }).then(function (res) {
        applyStateResponse(res, label + " очищено (независимо от других слоёв).");
      }).catch(function (err) { setError(err.message); });
    }

    /* ---------------- wiring ---------------- */
    $("d11-send").addEventListener("click", onSend);
    $("d11-demo-run").addEventListener("click", onDemoRun);
    $("d11-seed").addEventListener("click", onSeed);
    $("d11-new-session").addEventListener("click", onNewSession);
    $("d11-clear-st").addEventListener("click", function () {
      onClear("/api/day11/memory/short-term", "Short-term");
    });
    $("d11-clear-working").addEventListener("click", function () {
      onClear("/api/day11/memory/working", "Working memory");
    });
    $("d11-clear-lt").addEventListener("click", function () {
      onClear("/api/day11/memory/long-term", "Long-term memory");
    });

    $("d11-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        $("d11-send").click();
      }
    });

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day11") {
        loadState();
      }
    });

    loadState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay11);
  } else {
    initDay11();
  }
})();
