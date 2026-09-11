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

async function main() {
  await testDay6Metadata();
  await testDay7FullHistory();

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
