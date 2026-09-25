/* Minimal DOM/fetch harness for the Day 6 / Day 7 frontend regression tests.
 *
 * It loads the REAL app/static/js/*.js files in a vm sandbox with a tiny fake
 * DOM, stubs fetch, and asserts on the resulting DOM. Run with:
 *
 *     node tests/js/frontend_harness.js
 *
 * Exit code 0 = all checks passed, 1 = at least one failed.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const JS_DIR = path.resolve(__dirname, "..", "..", "app", "static", "js");

/* ------------------------------------------------------------------ */
/* Fake DOM                                                            */
/* ------------------------------------------------------------------ */
class FakeElement {
  constructor(tag) {
    this.tagName = (tag || "div").toUpperCase();
    this.children = [];
    this._text = "";
    this.className = "";
    this.style = {};
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.disabled = false;
    this.value = "";
    this._listeners = {};
    this._classes = new Set();
    const self = this;
    this.classList = {
      add(c) { self._classes.add(c); },
      remove(c) { self._classes.delete(c); },
      contains(c) { return self._classes.has(c); },
      toggle(c, on) {
        if (on === undefined) {
          self._classes.has(c) ? self._classes.delete(c) : self._classes.add(c);
        } else if (on) {
          self._classes.add(c);
        } else {
          self._classes.delete(c);
        }
      },
    };
  }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute() {}
  getAttribute() { return null; }
  addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); }
  removeEventListener() {}
  querySelector(sel) {
    const cls = sel.replace(/^\./, "");
    const search = (el) => {
      if ((el.className || "").split(/\s+/).indexOf(cls) !== -1) return el;
      for (const c of el.children) { const found = search(c); if (found) return found; }
      return null;
    };
    for (const c of this.children) { const found = search(c); if (found) return found; }
    return null;
  }
  querySelectorAll() { return []; }
  get textContent() { return this._text; }
  set textContent(v) { this._text = v == null ? "" : String(v); }
  get innerHTML() { return this.children.map((c) => c.textContent || "").join(""); }
  set innerHTML(v) { if (v === "") this.children = []; }
}

function makeDocument(ids) {
  const store = {};
  ids.forEach((id) => { store[id] = new FakeElement("div"); });
  return {
    readyState: "complete",
    getElementById: (id) => store[id] || null,
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (ns, tag) => new FakeElement(tag),
    querySelectorAll: () => [],
    addEventListener: () => {},
    dispatchEvent: () => {},
    _store: store,
  };
}

function runScript(document, fetchImpl, file) {
  const sandbox = {
    document,
    fetch: fetchImpl,
    console,
    setTimeout,
    clearTimeout,
    CustomEvent: function (type, opts) { return { type, detail: opts && opts.detail }; },
  };
  const context = vm.createContext(sandbox);
  const code = fs.readFileSync(path.join(JS_DIR, file), "utf8");
  vm.runInContext(code, context, { filename: file });
}

function jsonResponse(data, ok = true) {
  return Promise.resolve({
    ok,
    status: ok ? 200 : 500,
    json: () => Promise.resolve(data),
  });
}

function tick(ms = 20) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ------------------------------------------------------------------ */
/* Checks                                                             */
/* ------------------------------------------------------------------ */
const results = [];
function check(name, condition) {
  results.push({ name, ok: !!condition });
}

async function testDay6Metadata() {
  const ids = [
    "tab-day6", "panel-day6", "d6-agent-name", "d6-agent-provider",
    "d6-agent-model", "d6-agent-max-tokens", "d6-agent-thinking",
    "d6-input", "d6-send", "d6-loading", "d6-error",
    "d6-answer", "d6-meta",
  ];
  const document = makeDocument(ids);
  const calls = [];
  const fetchImpl = (url) => {
    calls.push(url);
    if (url === "/api/day6/agent") {
      return jsonResponse({
        agent_id: "day6-agent",
        name: "Simple Agent (transient)",
        provider: "deepseek",
        model: "deepseek-v4-flash",
        max_tokens: 8192,
        thinking: false,
        stateless: true,
      });
    }
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day6.js");
  await tick();

  const store = document._store;
  check("day6: fetch metadata on init", calls.indexOf("/api/day6/agent") !== -1);
  check("day6: does NOT POST chat on init", calls.indexOf("/api/day6/agent/chat") === -1);
  check("day6: provider filled before any request", store["d6-agent-provider"].textContent === "deepseek");
  check("day6: model filled before any request", store["d6-agent-model"].textContent === "deepseek-v4-flash");
  check("day6: name filled from backend", store["d6-agent-name"].textContent === "Simple Agent (transient)");
  check("day6: max output filled from backend", store["d6-agent-max-tokens"].textContent === "8192 tokens");
  check("day6: reasoning shown OFF from backend", store["d6-agent-thinking"].textContent === "OFF");
}

async function testDay7FullHistory() {
  const ids = [
    "tab-day7", "panel-day7", "d7-agent-name", "d7-agent-provider",
    "d7-agent-model", "d7-agent-max-tokens", "d7-agent-thinking",
    "d7-context-count", "d7-messages", "d7-input",
    "d7-send", "d7-clear", "d7-loading", "d7-error",
  ];
  const document = makeDocument(ids);
  const history = {
    agent: {
      agent_id: "default", name: "Default Agent", provider: "deepseek",
      model: "deepseek-v4-flash", max_tokens: 8192, thinking: false,
    },
    messages: [
      { role: "user", content: "M1" },
      { role: "assistant", content: "M2" },
      { role: "user", content: "M3" },
      { role: "assistant", content: "M4" },
      { role: "user", content: "M5" },
    ],
    count: 5,
  };
  const fetchImpl = (url) => {
    if (url === "/api/chat/history") return jsonResponse(history);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day7.js");
  await tick();

  const store = document._store;
  const messagesEl = store["d7-messages"];
  const bubbles = messagesEl.children;
  const texts = bubbles.map((b) => b.children[1].textContent);

  check("day7: renders ALL 5 messages", bubbles.length === 5);
  check("day7: messages in original order", JSON.stringify(texts) === JSON.stringify(["M1", "M2", "M3", "M4", "M5"]));
  check("day7: first message present", texts[0] === "M1");
  check("day7: last message present", texts[texts.length - 1] === "M5");
  check("day7: count shows 5", /^5\b/.test(store["d7-context-count"].textContent));
  check("day7: max output filled from backend", store["d7-agent-max-tokens"].textContent === "8192 tokens");
  check("day7: reasoning shown OFF from backend", store["d7-agent-thinking"].textContent === "OFF");
}

async function testDay9Diagnostics() {
  const ids = [
    "tab-day9", "panel-day9", "d9-model", "d9-recent", "d9-batch", "d9-cycles",
    "d9-mode-full", "d9-mode-compressed", "d9-recent-input", "d9-batch-input",
    "d9-seed", "d9-ask-control", "d9-clear", "d9-loading",
    "d9-messages", "d9-input", "d9-send", "d9-error", "d9-compression-note",
    "d9-full-count", "d9-summarized-count", "d9-recent-count", "d9-pending-count",
    "d9-full-tokens", "d9-summary-tokens", "d9-recent-tokens",
    "d9-compressed-tokens", "d9-saved", "d9-saved-percent", "d9-cycles-diag",
    "d9-summary-details", "d9-summary-text", "d9-rows", "d9-chart",
    "d9-answer-full", "d9-answer-compressed",
  ];
  const document = makeDocument(ids);
  const diag = {
    full_history_messages: 38,
    summarized_messages: 32,
    recent_messages_sent: 6,
    full_history_tokens_estimated: 5000,
    summary_tokens_estimated: 120,
    recent_tokens_estimated: 300,
    compressed_context_tokens_estimated: 420,
    tokens_saved: 4580,
    tokens_saved_percent: 91.6,
    compression_cycles: 1,
    summary: "ORION facts",
    recent_messages_limit: 6,
    compression_batch_size: 10,
    pending_messages: 0,
  };
  const fetchImpl = (url) => {
    if (url === "/api/day9/diagnostics") return jsonResponse({ diagnostics: diag, last_answer: null });
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day9.js");
  await tick();

  const store = document._store;
  check("day9: full history count shown", store["d9-full-count"].textContent === "38");
  check("day9: summarized count shown", store["d9-summarized-count"].textContent === "32");
  check("day9: recent count shown", store["d9-recent-count"].textContent === "6");
  check("day9: compressed tokens shown", store["d9-compressed-tokens"].textContent === "≈ 420");
  check("day9: saved percent shown", store["d9-saved-percent"].textContent === "91.6%");
  check("day9: cycles shown", store["d9-cycles-diag"].textContent === "1");
  check("day9: summary text rendered", store["d9-summary-text"].textContent === "ORION facts");
  // Transparency: the remembered summary is rendered INSIDE the dialog.
  const memCard = store["d9-messages"].children[0];
  check("day9: memory card inside dialog",
    memCard.className.indexOf("d9-memory-card") !== -1);
  check("day9: memory card shows remembered summary",
    memCard.children[1].textContent === "ORION facts");
}

async function testDay10State() {
  const ids = [
    "tab-day10", "panel-day10",
    "d10-s-sliding", "d10-s-sticky", "d10-s-branching",
    "d10-strategy-name", "d10-model", "d10-history-count", "d10-active-branch",
    "d10-window-input", "d10-recent-input", "d10-update-facts",
    "d10-seed-branches", "d10-demo-run", "d10-compare-run", "d10-clear",
    "d10-loading", "d10-progress", "d10-note",
    "d10-branch-panel", "d10-checkpoint", "d10-branch-list",
    "d10-branch-name", "d10-branch-create",
    "d10-messages", "d10-input", "d10-send", "d10-error",
    "d10-sys", "d10-facts-tokens", "d10-recent-tokens", "d10-cur",
    "d10-est-input", "d10-api-input", "d10-api-output", "d10-api-total",
    "d10-facts-extract", "d10-cost",
    "d10-cum-requests", "d10-cum-extractions", "d10-cum-input",
    "d10-cum-output", "d10-cum-total", "d10-cum-extract",
    "d10-sliding-panel", "d10-sw-summary", "d10-full-count",
    "d10-sent-count", "d10-dropped-count", "d10-dropped-list",
    "d10-facts-panel", "d10-facts-count", "d10-facts-recent",
    "d10-facts-block", "d10-facts-json", "d10-facts-error",
    "d10-compare-rows",
  ];
  const document = makeDocument(ids);
  const state = {
    strategy: "sliding_window",
    model: "deepseek-v4-flash",
    context: {
      strategy: "sliding_window",
      total_history_messages: 14,
      sent_history_messages: 6,
      dropped_messages: 8,
      dropped_preview: [{ role: "user", content: "old" }],
      sent_preview: [{ role: "user", content: "recent" }],
      window_size: 6,
      branches: [],
    },
  };
  const fetchImpl = (url) => {
    if (url.indexOf("/api/day10/state") === 0) return jsonResponse(state);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day10.js");
  await tick();

  const store = document._store;
  check("day10: strategy name rendered", store["d10-strategy-name"].textContent === "sliding_window");
  check("day10: model rendered", store["d10-model"].textContent === "deepseek-v4-flash");
  check("day10: sliding summary shows dropped/sent",
    store["d10-sw-summary"].textContent.indexOf("Dropped: 8 messages") !== -1);
  check("day10: sent count rendered", store["d10-sent-count"].textContent === "6 messages");
  check("day10: dropped count rendered", store["d10-dropped-count"].textContent === "8 messages");
}

async function testDay9DialogZones() {
  const ids = [
    "tab-day9", "panel-day9", "d9-model", "d9-recent", "d9-batch", "d9-cycles",
    "d9-mode-full", "d9-mode-compressed", "d9-recent-input", "d9-batch-input",
    "d9-seed", "d9-ask-control", "d9-clear", "d9-loading",
    "d9-messages", "d9-input", "d9-send", "d9-error", "d9-compression-note",
    "d9-full-count", "d9-summarized-count", "d9-recent-count", "d9-pending-count",
    "d9-full-tokens", "d9-summary-tokens", "d9-recent-tokens",
    "d9-compressed-tokens", "d9-saved", "d9-saved-percent", "d9-cycles-diag",
    "d9-summary-details", "d9-summary-text", "d9-rows", "d9-chart",
    "d9-answer-full", "d9-answer-compressed",
  ];
  const document = makeDocument(ids);
  const seedDiag = {
    full_history_messages: 8, summarized_messages: 0, recent_messages_sent: 4,
    pending_messages: 4, full_history_tokens_estimated: 100,
    summary_tokens_estimated: 0, recent_tokens_estimated: 20,
    compressed_context_tokens_estimated: 20, tokens_saved: 80,
    tokens_saved_percent: 80, compression_cycles: 0, summary: null,
    recent_messages_limit: 4, compression_batch_size: 2,
  };
  const chatDiag = Object.assign({}, seedDiag, {
    summarized_messages: 4, recent_messages_sent: 4, pending_messages: 0,
    summary_tokens_estimated: 10, flags: undefined,
    compression_cycles: 1,
    summary: "PROJECT ORION: FastAPI, PostgreSQL, no Redis, stateless, JWT",
  });
  const messages = [];
  for (let i = 0; i < 4; i++) {
    messages.push({ role: "user", content: "fact " + i });
    messages.push({ role: "assistant", content: "stored " + i });
  }
  const fetchImpl = (url) => {
    if (url === "/api/day9/diagnostics") {
      return jsonResponse({ diagnostics: seedDiag, last_answer: null });
    }
    if (url === "/api/day9/seed") {
      return jsonResponse({ messages: messages, count: messages.length, diagnostics: seedDiag });
    }
    if (url === "/api/day9/chat") {
      return jsonResponse({
        answer: "Итог: Orion", mode: "compressed", model: "deepseek-v4-flash",
        provider: "deepseek", finish_reason: "stop",
        usage: {
          full_context_tokens_estimated: 120,
          compressed_context_tokens_estimated: 30,
          tokens_saved: 90, tokens_saved_percent: 75,
        },
        diagnostics: chatDiag, compression_performed: true, compression_error: null,
      });
    }
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day9.js");
  await tick();

  const store = document._store;
  // seed the demo dialog
  store["d9-seed"]._listeners["click"][0]();
  await tick(40);
  check("day9-flow: dialog rendered after seed", store["d9-messages"].children.length > 1);
  check("day9-flow: memory card present after seed",
    store["d9-messages"].querySelector(".d9-memory-card") !== null);

  // ask the control question -> compression cycle
  store["d9-ask-control"]._listeners["click"][0]();
  await tick(60);
  const card = store["d9-messages"].querySelector(".d9-memory-card");
  check("day9-flow: remembered summary in dialog",
    card.children[1].textContent === "PROJECT ORION: FastAPI, PostgreSQL, no Redis, stateless, JWT");
  check("day9-flow: compression event shown in dialog",
    store["d9-messages"].querySelector(".d9-system-bubble") !== null);
  check("day9-flow: zone labels shown",
    store["d9-messages"].querySelector(".d9-zone-label") !== null);
}

async function testDay11State() {
  const ids = [
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
    "d11-ctx-dropped", "d11-ctx-dropped-list", "d11-ctx-blocks",
  ];
  const document = makeDocument(ids);
  const state = {
    model: "deepseek-v4-flash",
    memory: {
      session_id: "sess-test",
      short_term: [
        { role: "user", content: "Я Антон" },
        { role: "assistant", content: "Понял" },
      ],
      short_term_count: 2,
      working: { goal: "Сервис бронирования", constraints: ["Не использовать PostgreSQL"] },
      long_term: { entries: { preferred_language: "Python", answer_style: "concise" } },
      last_decision: {
        performed: true,
        nothing_to_save: false,
        error: null,
        working_memory: { goal: "Сервис бронирования", constraints: ["Не использовать PostgreSQL"] },
        long_term_memory: { preferred_language: "Python" },
      },
    },
  };
  const fetchImpl = (url) => {
    if (url === "/api/day11/state") return jsonResponse(state);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day11.js");
  await tick();

  const store = document._store;
  check("day11: model rendered", store["d11-model"].textContent === "deepseek-v4-flash");
  check("day11: session id rendered", store["d11-session-id"].textContent === "sess-test");
  check("day11: short-term count rendered", store["d11-st-count"].textContent === "2");
  check("day11: working count rendered", store["d11-working-count"].textContent === "2");
  check("day11: long-term count rendered", store["d11-lt-count"].textContent === "2");
  check("day11: long-term block shows key=value",
    store["d11-lt-block"].textContent.indexOf("preferred_language = Python") !== -1);
  check("day11: decision listed",
    store["d11-decision-list"].children.length > 0);
  check("day11: save decision short-term shown",
    store["d11-save-st"].textContent === "✓ Сохранено текущее сообщение");
  check("day11: save decision summary shows all layers",
    store["d11-save-summary"].textContent === "Short-term + Working + Long-term");
}

async function testDay11SaveDecision() {
  const ids = [
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
    "d11-ctx-dropped", "d11-ctx-dropped-list", "d11-ctx-blocks",
  ];
  const document = makeDocument(ids);
  const store = document._store;
  let currentDecision = null;

  const state = {
    model: "deepseek-v4-flash",
    memory: {
      session_id: "sess-test",
      short_term: [],
      short_term_count: 0,
      working: {},
      long_term: { entries: {} },
      last_decision: null,
    },
  };

  const chatResponse = () => ({
    session_id: "sess-test",
    answer: "OK",
    model: "deepseek-v4-flash",
    finish_reason: "stop",
    usage: {},
    cost: {},
    context: { strategy: "full", included_layers: [] },
    memory: {
      session_id: "sess-test",
      short_term: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "OK" },
      ],
      short_term_count: 2,
      working: (currentDecision && currentDecision.working_memory) || {},
      long_term: { entries: (currentDecision && currentDecision.long_term_memory) || {} },
      last_decision: currentDecision,
    },
    last_decision: currentDecision,
    classifier_error: null,
  });

  const fetchImpl = (url) => {
    if (url === "/api/day11/state") return jsonResponse(state);
    if (url === "/api/day11/chat") return jsonResponse(chatResponse());
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day11.js");
  await tick();

  store["d11-input"].value = "hello";

  const cases = [
    {
      name: "Short-term only",
      decision: { performed: true, nothing_to_save: true, error: null, working_memory: {}, long_term_memory: {} },
      summary: "только Short-term",
    },
    {
      name: "Short-term + Working",
      decision: { performed: true, nothing_to_save: false, error: null, working_memory: { goal: "G", constraints: ["C"] }, long_term_memory: {} },
      summary: "Short-term + Working",
    },
    {
      name: "Short-term + Long-term",
      decision: { performed: true, nothing_to_save: false, error: null, working_memory: {}, long_term_memory: { preferred_language: "Python" } },
      summary: "Short-term + Long-term",
    },
    {
      name: "Short-term + Working + Long-term",
      decision: { performed: true, nothing_to_save: false, error: null, working_memory: { goal: "G", constraints: ["C"] }, long_term_memory: { preferred_language: "Python" } },
      summary: "Short-term + Working + Long-term",
    },
  ];

  for (const c of cases) {
    currentDecision = c.decision;
    store["d11-input"].value = "hello";
    store["d11-send"]._listeners["click"][0]();
    await tick(60);
    check("day11-save-decision: " + c.name + " summary",
      store["d11-save-summary"].textContent === c.summary);
    check("day11-save-decision: " + c.name + " short-term shown",
      store["d11-save-st"].textContent === "✓ Сохранено текущее сообщение");
  }
}

async function testDay12Profile() {
  const ids = [
    "tab-day12", "panel-day12",
    "d12-active-style", "d12-active-badges", "d12-presets",
    "d12-name", "d12-style",
    "d12-f-structured", "d12-f-lists", "d12-f-examples", "d12-f-code",
    "d12-c-no-emoji", "d12-c-no-intro", "d12-c-no-lists", "d12-lang",
    "d12-custom",
    "d12-save", "d12-save-status", "d12-instructions",
    "d12-use-memory", "d12-classify",
    "d12-messages", "d12-input", "d12-send", "d12-loading", "d12-error",
    "d12-note", "d12-context-summary", "d12-context-layers",
    "d12-memory-summary", "d12-memory-list",
    "d12-demo-prompt", "d12-compare-run", "d12-compare-results",
  ];
  const document = makeDocument(ids);
  const profileResponse = {
    profile: {
      name: "Антон",
      style: "technical",
      format: { structured: true, use_lists: true, use_examples: false, use_code: true },
      constraints: { no_emoji: true, no_intro: true, no_lists: false, language: "ru" },
      custom_instructions: "Пиши только транслитом.",
    },
    instructions: "USER PROFILE (presentation preferences)\nRespond in Russian.",
    presets: [
      { id: "concise", title: "Краткий", description: "d", profile: { name: null, style: "concise", format: {}, constraints: {} } },
      { id: "detailed", title: "Подробный", description: "d", profile: { name: null, style: "detailed", format: {}, constraints: {} } },
      { id: "technical", title: "Технический", description: "d", profile: { name: null, style: "technical", format: {}, constraints: {} } },
    ],
    demo_prompt: "Помоги Антону разобраться: как устроен индекс в базе данных?",
  };
  const fetchImpl = (url) => {
    if (url === "/api/day12/profile") return jsonResponse(profileResponse);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day12.js");
  await tick();

  const store = document._store;
  check("day12: active style rendered",
    store["d12-active-style"].textContent.indexOf("technical") !== -1);
  check("day12: instructions rendered",
    store["d12-instructions"].textContent.indexOf("USER PROFILE") !== -1);
  check("day12: name filled from backend", store["d12-name"].value === "Антон");
  check("day12: custom instructions filled from backend",
    store["d12-custom"].value === "Пиши только транслитом.");
  check("day12: three presets rendered", store["d12-presets"].children.length === 3);
  check("day12: demo prompt filled",
    store["d12-demo-prompt"].value.indexOf("Антону") !== -1);
  check("day12: format badge rendered",
    store["d12-active-badges"].children.length > 0);
}

async function testDay13TaskState() {
  const ids = [
    "tab-day13", "panel-day13",
    "d13-goal", "d13-create", "d13-reset",
    "d13-loading", "d13-error", "d13-note",
    "d13-fsm",
    "d13-stage-planning", "d13-stage-execution",
    "d13-stage-pause",
    "d13-stage-validation", "d13-stage-done",
    "d13-stage", "d13-step", "d13-expected", "d13-status",
    "d13-pause", "d13-resume", "d13-allowed",
    "d13-plan-list", "d13-completed-list", "d13-raw",
    "d13-messages", "d13-input", "d13-send", "d13-context-block",
  ];
  const document = makeDocument(ids);
  const state = {
    model: "deepseek-v4-flash",
    has_task: true,
    allowed_transitions: ["validation"],
    state: {
      task_id: "task-abc123",
      goal: "Спроектировать REST API",
      stage: "execution",
      current_step: 2,
      expected_action: "Определить API endpoints",
      plan: [
        "Определить структуру данных",
        "Определить API endpoints",
        "Описать обработку ошибок",
        "Проверить итоговое решение",
      ],
      completed_steps: ["Определить структуру данных"],
      paused: false,
      created_at: "2026-01-01T00:00:00+00:00",
      updated_at: "2026-01-01T00:00:00+00:00",
    },
  };
  const fetchImpl = (url) => {
    if (url === "/api/day13/state") return jsonResponse(state);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day13.js");
  await tick();

  const store = document._store;
  check("day13: stage rendered uppercase", store["d13-stage"].textContent === "EXECUTION");
  check("day13: current step rendered", store["d13-step"].textContent === "2 / 4");
  check("day13: expected action rendered",
    store["d13-expected"].textContent === "Определить API endpoints");
  check("day13: status ACTIVE", store["d13-status"].textContent === "ACTIVE");
  check("day13: active FSM stage highlighted",
    store["d13-stage-execution"].classList.contains("active"));
  check("day13: past FSM stage marked passed",
    store["d13-stage-planning"].classList.contains("passed"));
  check("day13: allowed transitions shown",
    store["d13-allowed"].textContent === "validation");
  check("day13: completed steps listed",
    store["d13-completed-list"].children.length === 1);
  check("day13: raw state is JSON",
    store["d13-raw"].textContent.indexOf('"current_step": 2') !== -1);
  check("day13: pause enabled while active", store["d13-pause"].disabled === false);
  check("day13: resume disabled while active", store["d13-resume"].disabled === true);
}

async function testDay13PauseState() {
  const ids = [
    "tab-day13", "panel-day13",
    "d13-goal", "d13-create", "d13-reset",
    "d13-loading", "d13-error", "d13-note",
    "d13-fsm",
    "d13-stage-planning", "d13-stage-execution",
    "d13-stage-pause",
    "d13-stage-validation", "d13-stage-done",
    "d13-stage", "d13-step", "d13-expected", "d13-status",
    "d13-pause", "d13-resume", "d13-allowed",
    "d13-plan-list", "d13-completed-list", "d13-raw",
    "d13-messages", "d13-input", "d13-send", "d13-context-block",
  ];
  const document = makeDocument(ids);
  const state = {
    model: "deepseek-v4-flash",
    has_task: true,
    allowed_transitions: [],
    state: {
      task_id: "task-abc123",
      goal: "Спроектировать REST API",
      stage: "execution",
      current_step: 2,
      expected_action: "Определить API endpoints",
      plan: [
        "Определить структуру данных",
        "Определить API endpoints",
        "Описать обработку ошибок",
        "Проверить итоговое решение",
      ],
      completed_steps: ["Определить структуру данных"],
      paused: true,
      created_at: "2026-01-01T00:00:00+00:00",
      updated_at: "2026-01-01T00:00:00+00:00",
    },
  };
  const fetchImpl = (url) => {
    if (url === "/api/day13/state") return jsonResponse(state);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day13.js");
  await tick();

  const store = document._store;
  check("day13-pause: stage keeps saved workflow stage",
    store["d13-stage"].textContent === "EXECUTION");
  check("day13-pause: current step preserved",
    store["d13-step"].textContent === "2 / 4");
  check("day13-pause: expected action preserved",
    store["d13-expected"].textContent === "Определить API endpoints");
  check("day13-pause: status PAUSED",
    store["d13-status"].textContent === "PAUSED");
  check("day13-pause: pause node highlighted",
    store["d13-stage-pause"].classList.contains("active"));
  check("day13-pause: saved workflow stage NOT highlighted",
    !store["d13-stage-execution"].classList.contains("active"));
  check("day13-pause: pause button disabled",
    store["d13-pause"].disabled === true);
  check("day13-pause: resume button enabled",
    store["d13-resume"].disabled === false);
}

async function testDay16Mcp() {
  const ids = [
    "panel-day16",
    "d16-status", "d16-transport", "d16-server", "d16-tools-count",
    "d16-refresh", "d16-loading", "d16-error",
    "d16-tools", "d16-tools-empty", "d16-trace",
  ];
  const document = makeDocument(ids);
  const payload = {
    connected: true,
    server: { name: "Week 4 Demo MCP", transport: "stdio" },
    tools_count: 2,
    tools: [
      {
        name: "echo",
        description: "Returns supplied text.",
        input_schema: {
          type: "object",
          properties: { text: { type: "string" } },
          required: ["text"],
        },
      },
      {
        name: "get_server_info",
        description: "Returns demo MCP server information.",
        input_schema: { type: "object", properties: {} },
      },
    ],
    trace: [
      { step: "connect", status: "ok", message: "Connected via stdio" },
      { step: "initialize", status: "ok", message: "MCP session initialized" },
      { step: "list_tools", status: "ok", message: "Received 2 tools" },
    ],
    error: null,
  };
  const fetchImpl = (url) => {
    if (url === "/api/week4/day16/mcp/status") return jsonResponse(payload);
    return Promise.reject(new Error("unexpected fetch " + url));
  };

  runScript(document, fetchImpl, "day16.js");
  await tick();

  const store = document._store;
  check("day16: status Connected", store["d16-status"].textContent === "Connected");
  check("day16: transport stdio", store["d16-transport"].textContent === "stdio");
  check("day16: server name rendered",
    store["d16-server"].textContent === "Week 4 Demo MCP");
  check("day16: tools count rendered", store["d16-tools-count"].textContent === "2");
  check("day16: two tool cards rendered", store["d16-tools"].children.length === 2);
  check("day16: first tool is echo",
    store["d16-tools"].children[0].children[0].textContent === "echo");
  check("day16: second tool is get_server_info",
    store["d16-tools"].children[1].children[0].textContent === "get_server_info");
  check("day16: echo schema shows required text",
    store["d16-tools"].children[0].children[3].textContent === "text: string, required");
  check("day16: trace steps rendered", store["d16-trace"].children.length === 3);
  check("day16: error hidden on success",
    store["d16-error"].classList.contains("hidden"));
}

async function testDay16McpFailure() {
  const ids = [
    "panel-day16",
    "d16-status", "d16-transport", "d16-server", "d16-tools-count",
    "d16-refresh", "d16-loading", "d16-error",
    "d16-tools", "d16-tools-empty", "d16-trace",
  ];
  const document = makeDocument(ids);
  const payload = {
    connected: false,
    server: { name: "Week 4 Demo MCP", transport: "stdio" },
    tools_count: 0,
    tools: [],
    trace: [{ step: "error", status: "error", message: "Connection closed" }],
    error: "Connection closed",
  };
  const fetchImpl = () => jsonResponse(payload);

  runScript(document, fetchImpl, "day16.js");
  await tick();

  const store = document._store;
  check("day16-fail: status Connection failed",
    store["d16-status"].textContent === "Connection failed");
  check("day16-fail: no tool cards", store["d16-tools"].children.length === 0);
  check("day16-fail: empty message shown",
    !store["d16-tools-empty"].classList.contains("hidden"));
  check("day16-fail: error shown",
    !store["d16-error"].classList.contains("hidden") &&
    store["d16-error"].textContent === "Connection closed");
}

async function main() {
  await testDay6Metadata();
  await testDay7FullHistory();
  await testDay9Diagnostics();
  await testDay9DialogZones();
  await testDay10State();
  await testDay11State();
  await testDay11SaveDecision();
  await testDay12Profile();
  await testDay13TaskState();
  await testDay13PauseState();
  await testDay16Mcp();
  await testDay16McpFailure();

  let failures = 0;
  results.forEach((r) => {
    console.log((r.ok ? "PASS" : "FAIL") + " " + r.name);
    if (!r.ok) failures += 1;
  });
  console.log(failures === 0 ? "ALL FRONTEND CHECKS PASSED" : failures + " CHECK(S) FAILED");
  process.exitCode = failures === 0 ? 0 : 1;
}

main().catch((err) => {
  console.error("HARNESS ERROR:", err && err.stack ? err.stack : err);
  process.exitCode = 1;
});
