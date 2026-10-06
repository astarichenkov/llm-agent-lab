/* Mitsubishi Xpander Assistant — standalone product page.
 *
 * Talks to the thin /api/xpander facade, which reuses the existing Day 25
 * chat service (sessions + task memory + grounded answers). This file only
 * presents data: it never invents an answer or a source.
 */
(function () {
  "use strict";

  var API = "/api/xpander";

  /* Curated "similar topics" suggestions per detected category (static UI
   * content, not knowledge-base data). */
  var CATEGORY_SIMILAR = {
    vibration: ["Вибрация при старте", "Шумы на холостом ходу", "Тряска на малой скорости", "Проверка опор двигателя"],
    cvt: ["Замена масла CVT", "Рывки при разгоне", "Перегрев вариатора", "Прогрев вариатора зимой"],
    maintenance: ["Регламент ТО", "Моторное масло", "Тормозная жидкость", "Воздушный фильтр"],
    suspension: ["Стук подвески", "Стойки и опоры", "Давление в шинах", "Развал-схождение"],
    electric: ["Предохранители", "Ошибки на панели", "Аккумулятор", "Освещение"],
    owners: ["Пластик и мойка", "Термостат", "Обогрев салона", "Зимняя эксплуатация"]
  };

  function initXpander() {
    var $ = function (id) { return document.getElementById(id); };
    var required = [
      "xp-chat-list", "xp-messages", "xp-form", "xp-input",
      "xp-error", "xp-topic-known", "xp-topic-similar", "xp-topic-title",
    ];
    // The page is standalone; guard against partial DOMs but do not hard-fail
    // when optional widgets are absent.
    var missing = required.filter(function (id) { return !$(id); });
    if (missing.length && !$("xp-chat-list")) {
      return;
    }

    var state = { status: null, sessions: [], current: null, busy: false, filter: "" };

    function make(tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    }
    function show(el, on) { if (el) el.classList.toggle("hidden", !on); }
    function clear(el) { while (el && el.firstChild) el.removeChild(el.firstChild); }
    function setError(message) {
      var el = $("xp-error");
      if (!el) return;
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
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
    /* formatting helpers                                                  */
    /* ------------------------------------------------------------------ */
    function formatKm(value) {
      var n = Number(value);
      if (isNaN(n)) return String(value);
      return n.toLocaleString("ru-RU") + " км";
    }
    function relativeTime(iso) {
      if (!iso) return "";
      var then = new Date(iso);
      if (isNaN(then.getTime())) return "";
      var diff = (Date.now() - then.getTime()) / 1000;
      if (diff < 60) return "только что";
      if (diff < 3600) return Math.floor(diff / 60) + " мин";
      if (diff < 86400) {
        return String(then.getHours()).padStart(2, "0") + ":" + String(then.getMinutes()).padStart(2, "0");
      }
      return String(then.getDate()).padStart(2, "0") + "." + String(then.getMonth() + 1).padStart(2, "0");
    }
    function truncate(text, limit) {
      var t = (text || "").replace(/\s+/g, " ").trim();
      return t.length > limit ? t.slice(0, limit - 1) + "…" : t;
    }

    /* ------------------------------------------------------------------ */
    /* chat list                                                           */
    /* ------------------------------------------------------------------ */
    function chatIconSvg() {
      return '<svg viewBox="0 0 24 24" width="17" height="17"><path d="M4 5 H20 V16 H9 L4 20 Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';
    }

    function lastPreview(session) {
      if (session.__preview === undefined) return "";
      return session.__preview || "";
    }

    function renderChatList() {
      var list = $("xp-chat-list");
      if (!list) return;
      clear(list);
      var sessions = state.sessions.filter(function (s) {
        if (!state.filter) return true;
        return (s.title || "").toLowerCase().indexOf(state.filter) !== -1;
      });
      if (!sessions.length) {
        list.appendChild(make("p", "xp-empty-hint", state.filter ? "Ничего не найдено." : "Пока нет чатов. Начните новый."));
        return;
      }
      sessions.forEach(function (session) {
        var item = make("div", "xp-chat-item");
        if (state.current && state.current.id === session.id) item.classList.add("active");

        var icon = make("span", "xp-chat-item__icon");
        icon.innerHTML = chatIconSvg();
        item.appendChild(icon);

        var main = make("div", "xp-chat-item__main");
        main.appendChild(make("div", "xp-chat-item__title", session.title || "Новый чат"));
        if (session.__preview) {
          main.appendChild(make("div", "xp-chat-item__preview", truncate(session.__preview, 46)));
        }
        item.appendChild(main);
        item.appendChild(make("div", "xp-chat-item__time", relativeTime(session.updated_at)));

        var del = make("button", "xp-chat-item__del", "×");
        del.type = "button";
        del.title = "Удалить чат";
        del.addEventListener("click", function (event) {
          event.stopPropagation();
          deleteSession(session.id);
        });
        item.appendChild(del);

        item.addEventListener("click", function () { selectSession(session.id); });
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

    /* ------------------------------------------------------------------ */
    /* current topic panel                                                 */
    /* ------------------------------------------------------------------ */
    function detectCategory() {
      var session = state.current;
      var text = "";
      if (session) text += (session.title || "") + " ";
      if (state.currentState) {
        text += (state.currentState.goal || "") + " ";
        (state.currentState.known_facts || []).forEach(function (f) { text += (f.text || "") + " "; });
      }
      var t = text.toLowerCase();
      if (/вибрац|шум|стук|тряск/.test(t)) return "vibration";
      if (/вариатор|cvt|atf|коробк|трансмисс/.test(t)) return "cvt";
      if (/подвеск|стойк|амортиз|шин/.test(t)) return "suspension";
      if (/электр|предохранит|аккумул|батар|ошибк|ошибок/.test(t)) return "electric";
      if (/владел|сообществ|telegram|опыт/.test(t)) return "owners";
      if (/обслуж|масл|жидкост|фильтр|регламент/.test(t)) return "maintenance";
      return "maintenance";
    }

    function knownCards(taskState) {
      var cards = [];
      var vehicle = (taskState && taskState.vehicle) || {};
      if (vehicle.transmission) cards.push({ label: "Коробка", value: vehicle.transmission });
      if (vehicle.year) cards.push({ label: "Год", value: String(vehicle.year) });
      if (vehicle.mileage_km !== null && vehicle.mileage_km !== undefined) {
        cards.push({ label: "Пробег", value: formatKm(vehicle.mileage_km) });
      }
      ((taskState && taskState.known_facts) || []).forEach(function (fact) {
        if (fact.key && fact.key.indexOf("vehicle_") === 0) return;
        cards.push({ label: "", value: fact.text });
      });
      ((taskState && taskState.hypotheses) || []).forEach(function (h) {
        cards.push({ label: "Предположение", value: h.text });
      });
      return cards.slice(0, 4);
    }

    function renderCurrentTopic() {
      var session = state.current;
      var title = (session && session.title) || "Mitsubishi Xpander";
      $("xp-topic-title").textContent = title;
      $("xp-topic-vehicle").textContent = "Mitsubishi Xpander · Обсуждение";

      var known = $("xp-topic-known");
      clear(known);
      var cards = knownCards(state.currentState);
      if (!cards.length) {
        var hint = make("div", "xp-known-card");
        hint.appendChild(make("span", "xp-known-card__dot"));
        var box = make("div");
        box.appendChild(make("span", "xp-known-card__label", "Известно"));
        box.appendChild(make("span", null, "Спросите ассистента — детали появятся здесь."));
        hint.appendChild(box);
        known.appendChild(hint);
      } else {
        cards.forEach(function (card) {
          var el = make("div", "xp-known-card");
          el.appendChild(make("span", "xp-known-card__dot"));
          var box = make("div");
          if (card.label) box.appendChild(make("span", "xp-known-card__label", card.label));
          box.appendChild(make("span", null, card.value));
          el.appendChild(box);
          known.appendChild(el);
        });
      }

      var similar = $("xp-topic-similar");
      clear(similar);
      (CATEGORY_SIMILAR[detectCategory()] || CATEGORY_SIMILAR.maintenance).forEach(function (topic) {
        var li = make("li");
        var btn = make("button", null, topic);
        btn.type = "button";
        btn.addEventListener("click", function () {
          startTopicChat(topic, topic);
        });
        li.appendChild(btn);
        similar.appendChild(li);
      });
    }

    /* ------------------------------------------------------------------ */
    /* messages                                                            */
    /* ------------------------------------------------------------------ */
    var LIST_RE = /^\s*(?:[-•*✓]|\d+[.)])\s+(.*)$/;

    function renderAnswer(content) {
      var wrap = document.createDocumentFragment();
      var lines = String(content || "").split(/\r?\n/);
      var paragraph = [];
      var checks = [];

      function flushParagraph() {
        if (!paragraph.length) return;
        var p = make("p", null, paragraph.join(" "));
        wrap.appendChild(p);
        paragraph = [];
      }
      function flushChecks() {
        if (!checks.length) return;
        var ul = make("ul", "xp-checks");
        checks.forEach(function (item) { ul.appendChild(make("li", null, item)); });
        wrap.appendChild(ul);
        checks = [];
      }

      lines.forEach(function (raw) {
        var line = raw.trim();
        if (!line) {
          flushParagraph();
          flushChecks();
          return;
        }
        var match = LIST_RE.exec(line);
        if (match) {
          flushParagraph();
          checks.push(match[1]);
        } else {
          flushChecks();
          paragraph.push(line);
        }
      });
      flushParagraph();
      flushChecks();
      if (!wrap.childNodes.length) wrap.appendChild(make("p", null, ""));
      return wrap;
    }

    function sourceTypeLabel(sourceType) {
      if (sourceType === "manual") return "База знаний";
      if (sourceType === "telegram") return "Опыт владельцев";
      return "Источник";
    }

    function renderEvidence(evidence) {
      var details = make("details", "xp-sources");
      details.appendChild(make("summary", null, "Источники (" + evidence.length + ")"));

      var chips = make("div", "xp-sources__chips");
      var types = {};
      evidence.forEach(function (e) { types[e.source_type || "other"] = true; });
      if (types.manual) {
        chips.appendChild(make("span", "xp-chip xp-chip--manual", "Официальные рекомендации"));
        chips.appendChild(make("span", "xp-chip xp-chip--manual", "База знаний"));
      }
      if (types.telegram) {
        chips.appendChild(make("span", "xp-chip xp-chip--telegram", "Опыт владельцев"));
      }
      if (!Object.keys(types).length) {
        chips.appendChild(make("span", "xp-chip", "Источники"));
      }
      details.appendChild(chips);

      evidence.forEach(function (entry) {
        var card = make("article", "xp-evidence");
        var meta = make("div", "xp-evidence__meta");
        var src = make("span");
        src.appendChild(document.createTextNode(sourceTypeLabel(entry.source_type) + ": "));
        // For Telegram show the human-readable chat/topic name instead of the
        // raw export file path.
        var sourceLabel = entry.source_type === "telegram"
          ? (entry.chat_name || entry.source || "обсуждение владельцев")
          : (entry.source || "источник");
        src.appendChild(make("b", null, sourceLabel));
        meta.appendChild(src);

        if (entry.section) {
          var sec = make("span");
          sec.appendChild(document.createTextNode("Раздел: "));
          sec.appendChild(make("b", null, entry.section));
          meta.appendChild(sec);
        }
        if (entry.page !== null && entry.page !== undefined) {
          var pg = make("span");
          pg.appendChild(document.createTextNode("Стр.: "));
          pg.appendChild(make("b", null, String(entry.page)));
          meta.appendChild(pg);
        }
        if (entry.message_ids && entry.message_ids.length) {
          var msg = make("span");
          msg.appendChild(document.createTextNode("Сообщения: "));
          msg.appendChild(make("b", null, entry.message_ids.join(", ")));
          meta.appendChild(msg);
        }
        card.appendChild(meta);

        card.appendChild(make("blockquote", null, entry.quote || "(цитата недоступна)"));

        if (entry.chunk_text) {
          var full = make("details");
          full.appendChild(make("summary", null, "Показать полный фрагмент"));
          full.appendChild(make("pre", null, entry.chunk_text));
          card.appendChild(full);
        }
        details.appendChild(card);
      });
      return details;
    }

    function renderMessage(message) {
      var article = make("article", "xp-msg xp-msg--" + (message.role === "user" ? "user" : "assistant"));
      if (message.role === "assistant") {
        article.appendChild(make("div", "xp-msg__avatar", "X"));
      }
      var bubble = make("div", "xp-msg__bubble");
      if (message.role === "user") {
        bubble.appendChild(make("p", null, message.content));
      } else {
        bubble.appendChild(renderAnswer(message.content));
        if (message.evidence && message.evidence.length) {
          bubble.appendChild(renderEvidence(message.evidence));
        }
      }
      article.appendChild(bubble);
      return article;
    }

    function ensureMessageArea() {
      var box = $("xp-messages");
      if (box.querySelector("#xp-empty")) clear(box);
      return box;
    }

    function renderMessages(messages) {
      var box = $("xp-messages");
      clear(box);
      if (!messages.length) {
        var empty = make("div", "xp-empty");
        empty.id = "xp-empty";
        empty.innerHTML =
          '<div class="xp-empty__mark">' +
          '<svg viewBox="0 0 48 48" width="42" height="42">' +
          '<path d="M24 4 L43 14 L24 44 L5 14 Z" fill="none" stroke="#e60012" stroke-width="2.5"/>' +
          '<path d="M24 16 L34 21 L24 36 L14 21 Z" fill="#e60012" opacity="0.8"/></svg></div>' +
          '<h3>Чем помочь по вашему Xpander?</h3>' +
          '<p>Задайте вопрос или выберите популярную тему выше.</p>';
        box.appendChild(empty);
        return;
      }
      messages.forEach(function (message) { box.appendChild(renderMessage(message)); });
      // Deep link: /xpander?expandSources=1 opens every sources block.
      if (window.location.search.indexOf("expandSources=1") !== -1) {
        box.querySelectorAll(".xp-sources").forEach(function (node) { node.open = true; });
      }
      box.scrollTop = box.scrollHeight;
    }

    /* ------------------------------------------------------------------ */
    /* session actions                                                     */
    /* ------------------------------------------------------------------ */
    async function selectSession(id) {
      setError("");
      try {
        var detail = await api("GET", "/sessions/" + id);
        state.current = detail.session;
        state.currentState = detail.state;
        $("xp-chat-title").textContent = detail.session.title || "Mitsubishi Xpander";
        $("xp-chat-subtitle").textContent = "Mitsubishi Xpander · Обсуждение";
        renderMessages(detail.messages || []);
        renderCurrentTopic();
        renderChatList();
        closeDrawer();
        var box = $("xp-messages");
        if (box) box.scrollTop = box.scrollHeight;
      } catch (err) {
        setError(err.message);
      }
    }

    async function createSession(title) {
      var session = await api("POST", "/sessions", title ? { title: title } : {});
      session.__preview = "";
      state.sessions.unshift(session);
      renderChatList();
      return session;
    }

    async function deleteSession(id) {
      try {
        await api("DELETE", "/sessions/" + id);
        state.sessions = state.sessions.filter(function (s) { return s.id !== id; });
        if (state.current && state.current.id === id) {
          state.current = null;
          state.currentState = null;
          renderMessages([]);
          renderCurrentTopic();
        }
        renderChatList();
      } catch (err) {
        setError(err.message);
      }
    }

    async function newChat(title, prompt) {
      setError("");
      try {
        var session = await createSession(title || "");
        state.current = session;
        state.currentState = null;
        $("xp-chat-title").textContent = session.title || "Mitsubishi Xpander";
        $("xp-chat-subtitle").textContent = "Mitsubishi Xpander · Новый чат";
        renderMessages([]);
        renderCurrentTopic();
        renderChatList();
        closeDrawer();
        if (prompt) {
          $("xp-input").value = prompt;
          autoGrow();
          $("xp-input").focus();
        }
        return session;
      } catch (err) {
        setError(err.message);
      }
    }

    function startTopicChat(title, prompt) {
      newChat(title, prompt);
    }

    /* ------------------------------------------------------------------ */
    /* send                                                                */
    /* ------------------------------------------------------------------ */
    function autoGrow() {
      var input = $("xp-input");
      if (!input) return;
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 140) + "px";
    }

    async function sendMessage(event) {
      if (event) event.preventDefault();
      if (state.busy) return;
      var input = $("xp-input");
      var content = (input.value || "").trim();
      if (!content) return;

      state.busy = true;
      show($("xp-send-loading"), true);
      setError("");
      try {
        if (!state.current) {
          var title = truncate(content, 48);
          var session = await createSession(title);
          state.current = session;
          $("xp-chat-title").textContent = session.title;
        }
        var response = await api("POST", "/sessions/" + state.current.id + "/messages", { content: content });
        input.value = "";
        autoGrow();

        state.current = response.session;
        state.currentState = response.state;
        var box = ensureMessageArea();
        box.appendChild(renderMessage(response.user_message));
        box.appendChild(renderMessage(response.assistant_message));
        box.scrollTop = box.scrollHeight;

        var updated = state.sessions.filter(function (s) { return s.id !== response.session.id; });
        response.session.__preview = content;
        updated.unshift(response.session);
        state.sessions = updated;
        $("xp-chat-title").textContent = response.session.title || state.current.title;
        renderCurrentTopic();
        renderChatList();
      } catch (err) {
        setError(err.message);
      } finally {
        state.busy = false;
        show($("xp-send-loading"), false);
      }
    }

    /* ------------------------------------------------------------------ */
    /* responsive drawer                                                   */
    /* ------------------------------------------------------------------ */
    function openDrawer() { var el = $("xp-chatlist-panel"); if (el) el.classList.add("open"); }
    function closeDrawer() { var el = $("xp-chatlist-panel"); if (el) el.classList.remove("open"); }

    /* ------------------------------------------------------------------ */
    /* wiring                                                              */
    /* ------------------------------------------------------------------ */
    var newBtn = $("xp-new-chat");
    if (newBtn) newBtn.addEventListener("click", function () { newChat("", ""); });

    var chatToggle = $("xp-chat-toggle");
    if (chatToggle) chatToggle.addEventListener("click", openDrawer);
    var chatClose = $("xp-chatlist-close");
    if (chatClose) chatClose.addEventListener("click", closeDrawer);

    var form = $("xp-form");
    if (form) form.addEventListener("submit", sendMessage);

    var input = $("xp-input");
    if (input) {
      input.addEventListener("input", autoGrow);
      input.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendMessage();
        }
      });
    }

    var search = $("xp-search");
    if (search) {
      search.addEventListener("input", function () {
        state.filter = (search.value || "").trim().toLowerCase();
        renderChatList();
      });
    }

    document.querySelectorAll(".xp-topic-card").forEach(function (card) {
      card.addEventListener("click", function () {
        var topic = card.getAttribute("data-topic");
        var prompt = card.getAttribute("data-prompt");
        var labels = { noise: "Вибрация и шумы", cvt: "Вариатор и трансмиссия", maintenance: "Обслуживание", suspension: "Подвеска", electric: "Электрика", owners: "Опыт владельцев" };
        startTopicChat(labels[topic] || "Новая тема", prompt);
        var section = $("xp-chats");
        if (section) section.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    document.querySelectorAll("[data-nav]").forEach(function (link) {
      link.addEventListener("click", function () {
        var target = link.getAttribute("data-nav");
        if (target === "chats") {
          var chats = $("xp-chats"); if (chats) chats.scrollIntoView({ behavior: "smooth" });
          return;
        }
        if (target === "topics") {
          var topics = $("xp-topics"); if (topics) topics.scrollIntoView({ behavior: "smooth" });
          return;
        }
        if (target === "owners") {
          startTopicChat("Опыт владельцев", "Что владельцы в Telegram рассказывают о типичных проблемах Mitsubishi Xpander?");
          return;
        }
        if (target === "knowledge") {
          startTopicChat("База знаний", "Что официальная документация Mitsubishi Xpander рекомендует по обслуживанию вариатора?");
        }
      });
    });

    var heroOpen = $("xp-hero-open");
    if (heroOpen) heroOpen.addEventListener("click", function () {
      if (!state.current) { newChat("", ""); }
      var chats = $("xp-chats"); if (chats) chats.scrollIntoView({ behavior: "smooth" });
      if (input) input.focus();
    });
    var heroTopics = $("xp-hero-topics");
    if (heroTopics) heroTopics.addEventListener("click", function () {
      var topics = $("xp-topics"); if (topics) topics.scrollIntoView({ behavior: "smooth" });
    });

    async function boot() {
      try { state.status = await api("GET", "/status"); } catch (err) { state.status = null; }
      await loadSessions();
      var params = new URLSearchParams(window.location.search);
      var wanted = params.get("session");
      if (wanted && state.sessions.some(function (s) { return s.id === wanted; })) {
        await selectSession(wanted);
      } else if (state.sessions.length) {
        await selectSession(state.sessions[0].id);
      } else {
        renderCurrentTopic();
      }
    }
    boot();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initXpander);
  } else {
    initXpander();
  }
})();
