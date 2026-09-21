/* LLM Agent Lab — Day 12: personalization / user profile.
 * Vanilla JS. Talks to /api/day12/*. The backend owns profile->instructions
 * conversion; this file only edits the structured profile and shows the
 * effect (active profile, instructions, personalized chat, profile compare).
 */
(function () {
  "use strict";

  function initDay12() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "tab-day12", "panel-day12",
      "d12-active-style", "d12-active-badges", "d12-presets",
      "d12-name", "d12-style",
      "d12-f-structured", "d12-f-lists", "d12-f-examples", "d12-f-code",
      "d12-c-no-emoji", "d12-c-no-intro", "d12-c-no-lists", "d12-lang",
      "d12-custom",
      "d12-save", "d12-save-status", "d12-instructions",
      "d12-use-memory", "d12-classify",
      "d12-messages", "d12-input", "d12-send", "d12-loading", "d12-error",
      "d12-note",
      "d12-context-summary", "d12-context-layers",
      "d12-memory-summary", "d12-memory-list",
      "d12-demo-prompt", "d12-compare-run", "d12-compare-results"
    ];
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length) {
      console.error("Day12: missing DOM elements, disabled:", missing.join(", "));
      return;
    }

    var state = { inFlight: false, profile: null, presets: [] };

    /* ---------------- helpers ---------------- */
    function setError(m) {
      $("d12-error").textContent = m || "";
      $("d12-error").classList.toggle("hidden", !m);
    }
    function setNote(m) {
      $("d12-note").textContent = m || "";
      $("d12-note").classList.toggle("hidden", !m);
    }
    function setSaveStatus(m) { $("d12-save-status").textContent = m || ""; }
    function setLoading(on) {
      state.inFlight = on;
      $("d12-save").disabled = on;
      $("d12-send").disabled = on;
      $("d12-compare-run").disabled = on;
      $("d12-loading").classList.toggle("hidden", !on);
    }
    function http(path, opts) {
      return fetch(path, opts).then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, status: r.status, data: d };
        });
      });
    }

    /* ---------------- profile <-> form ---------------- */
    function readForm() {
      return {
        name: $("d12-name").value.trim() || null,
        style: $("d12-style").value,
        format: {
          structured: $("d12-f-structured").checked,
          use_lists: $("d12-f-lists").checked,
          use_examples: $("d12-f-examples").checked,
          use_code: $("d12-f-code").checked
        },
        constraints: {
          no_emoji: $("d12-c-no-emoji").checked,
          no_intro: $("d12-c-no-intro").checked,
          no_lists: $("d12-c-no-lists").checked,
          language: $("d12-lang").value
        },
        custom_instructions: $("d12-custom").value.trim() || null
      };
    }
    function fillForm(profile) {
      $("d12-name").value = profile.name || "";
      $("d12-style").value = profile.style;
      var fmt = profile.format || {};
      $("d12-f-structured").checked = !!fmt.structured;
      $("d12-f-lists").checked = !!fmt.use_lists;
      $("d12-f-examples").checked = !!fmt.use_examples;
      $("d12-f-code").checked = !!fmt.use_code;
      var cons = profile.constraints || {};
      $("d12-c-no-emoji").checked = !!cons.no_emoji;
      $("d12-c-no-intro").checked = !!cons.no_intro;
      $("d12-c-no-lists").checked = !!cons.no_lists;
      $("d12-lang").value = cons.language || "ru";
      $("d12-custom").value = profile.custom_instructions || "";
    }

    /* ---------------- rendering ---------------- */
    function chip(text) {
      var span = document.createElement("span");
      span.className = "d11-layer-chip";
      span.textContent = text;
      return span;
    }
    function renderActive(profile, instructions) {
      state.profile = profile;
      $("d12-active-style").textContent = "Стиль: " + profile.style;
      var badges = $("d12-active-badges");
      badges.innerHTML = "";
      if (profile.name) badges.appendChild(chip("имя: " + profile.name));
      var fmt = profile.format || {};
      if (fmt.structured) badges.appendChild(chip("структурированный"));
      if (fmt.use_lists) badges.appendChild(chip("списки"));
      if (fmt.use_examples) badges.appendChild(chip("примеры"));
      if (fmt.use_code) badges.appendChild(chip("код"));
      var cons = profile.constraints || {};
      if (cons.no_emoji) badges.appendChild(chip("без emoji"));
      if (cons.no_intro) badges.appendChild(chip("без вступления"));
      if (cons.no_lists) badges.appendChild(chip("без списков"));
      badges.appendChild(chip("язык: " + (cons.language || "ru")));
      if (profile.custom_instructions) badges.appendChild(chip("свои инструкции"));
      $("d12-instructions").textContent = instructions || "—";
    }
    function renderPresets(presets) {
      state.presets = presets || [];
      var box = $("d12-presets");
      box.innerHTML = "";
      state.presets.forEach(function (preset) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "link-btn";
        btn.textContent = preset.title;
        btn.title = preset.description;
        btn.addEventListener("click", function () { onApplyPreset(preset.id); });
        box.appendChild(btn);
      });
    }
    function renderMemory(memory) {
      if (!memory) return;
      var entries = (memory.long_term && memory.long_term.entries) || {};
      var keys = Object.keys(entries);
      $("d12-memory-summary").textContent =
        "short-term: " + (memory.short_term_count || 0) +
        " · long-term: " + keys.length + " записей";
      $("d12-memory-list").innerHTML = "";
      keys.forEach(function (key) {
        var row = document.createElement("div");
        row.className = "d10-msg-row";
        var role = document.createElement("span");
        role.className = "d10-msg-role";
        role.textContent = "memory";
        var text = document.createElement("span");
        text.className = "d10-msg-text";
        text.textContent = key + " = " + entries[key];
        row.appendChild(role);
        row.appendChild(text);
        $("d12-memory-list").appendChild(row);
      });
      if (!keys.length) {
        var empty = document.createElement("div");
        empty.className = "d10-msg-empty";
        empty.textContent = "(long-term память пуста)";
        $("d12-memory-list").appendChild(empty);
      }
    }
    function renderContext(ctx) {
      if (!ctx) return;
      var layers = ctx.included_layers || [];
      $("d12-context-summary").textContent =
        "personalization: " + (ctx.personalization_included ? "да" : "нет") +
        " · всего слоёв: " + layers.length;
      var box = $("d12-context-layers");
      box.innerHTML = "";
      layers.forEach(function (layer) { box.appendChild(chip(layer)); });
    }
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
      $("d12-messages").appendChild(bubble);
      $("d12-messages").scrollTop = $("d12-messages").scrollHeight;
    }

    /* ---------------- profile loading / saving ---------------- */
    function applyProfileResponse(data, note) {
      renderActive(data.profile, data.instructions);
      if (!state._loadedOnce) {
        fillForm(data.profile);
        state._loadedOnce = true;
      }
      if (data.presets && data.presets.length) renderPresets(data.presets);
      if (data.demo_prompt && !$("d12-demo-prompt").value.trim()) {
        $("d12-demo-prompt").value = data.demo_prompt;
      }
      if (note) setNote(note);
    }
    function loadProfile() {
      return http("/api/day12/profile").then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || "Не удалось загрузить профиль.");
        applyProfileResponse(res.data);
      }).catch(function (err) { setError(err.message); });
    }
    function onSave() {
      if (state.inFlight) return;
      setLoading(true); setError(""); setNote(""); setSaveStatus("");
      http("/api/day12/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readForm())
      }).then(function (res) {
        if (!res.ok) {
          var detail = res.data.detail;
          if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg; }).join("; ");
          throw new Error(detail || "Профиль отклонён.");
        }
        state._loadedOnce = true;
        renderActive(res.data.profile, res.data.instructions);
        setSaveStatus("Профиль сохранён и будет применяться автоматически.");
      }).catch(function (err) {
        setError(err.message || "Не удалось сохранить профиль.");
      }).finally(function () { setLoading(false); });
    }
    function onApplyPreset(presetId) {
      if (state.inFlight) return;
      setLoading(true); setError(""); setNote(""); setSaveStatus("");
      http("/api/day12/profile/preset/" + encodeURIComponent(presetId), { method: "POST" })
        .then(function (res) {
          if (!res.ok) throw new Error(res.data.detail || "Пресет не найден.");
          state._loadedOnce = true;
          fillForm(res.data.profile);
          renderActive(res.data.profile, res.data.instructions);
          setSaveStatus("Применён готовый профиль.");
        }).catch(function (err) { setError(err.message); })
        .finally(function () { setLoading(false); });
    }

    /* ---------------- personalized chat ---------------- */
    function onSend() {
      if (state.inFlight) return;
      var text = $("d12-input").value.trim();
      if (!text) { setError("Введите сообщение."); return; }
      setError(""); setNote("");
      addBubble("user", text);
      setLoading(true);
      http("/api/day12/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          use_memory: $("d12-use-memory").checked,
          classify: $("d12-classify").checked
        })
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
        var d = res.data;
        addBubble("assistant", d.answer);
        renderActive(d.profile, d.personalization_block);
        renderContext(d.context);
        renderMemory(d.memory);
        $("d12-input").value = "";
      }).catch(function (err) {
        setError(err.message || "Что-то пошло не так.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- compare ---------------- */
    function renderCompare(data) {
      var box = $("d12-compare-results");
      box.innerHTML = "";
      (data.items || []).forEach(function (item) {
        var card = document.createElement("article");
        card.className = "d12-compare-card";
        var title = document.createElement("h3");
        title.textContent = item.title + " (" + item.profile.style + ")";
        var answer = document.createElement("div");
        answer.className = "d12-compare-answer";
        answer.textContent = item.answer;
        var meta = document.createElement("p");
        meta.className = "field-hint";
        meta.textContent = "finish_reason: " + (item.finish_reason || "—");
        var details = document.createElement("details");
        var summary = document.createElement("summary");
        summary.textContent = "Показать инструкции профиля";
        var pre = document.createElement("pre");
        pre.className = "d10-facts-block";
        pre.textContent = item.personalization_block;
        details.appendChild(summary);
        details.appendChild(pre);
        card.appendChild(title);
        card.appendChild(answer);
        card.appendChild(meta);
        card.appendChild(details);
        box.appendChild(card);
      });
    }
    function onCompare() {
      if (state.inFlight) return;
      var text = $("d12-demo-prompt").value.trim();
      if (!text) { setError("Введите вопрос для сравнения."); return; }
      setError(""); setNote("");
      setLoading(true);
      http("/api/day12/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      }).then(function (res) {
        if (!res.ok) throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
        renderCompare(res.data);
        setNote("Один и тот же вопрос — три профиля. Разница вызвана только профилем.");
      }).catch(function (err) {
        setError(err.message || "Не удалось выполнить сравнение.");
      }).finally(function () { setLoading(false); });
    }

    /* ---------------- wiring ---------------- */
    $("d12-save").addEventListener("click", onSave);
    $("d12-send").addEventListener("click", onSend);
    $("d12-compare-run").addEventListener("click", onCompare);

    document.addEventListener("llmtabchange", function (event) {
      if (event.detail && event.detail.tab === "day12") {
        loadProfile();
      }
    });

    loadProfile();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDay12);
  } else {
    initDay12();
  }
})();
