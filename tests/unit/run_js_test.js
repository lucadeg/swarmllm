// harness/run-js.js: the run_js document and result format, with a fake runner (the real frame is
// tests/e2e/preview_browser.mjs).
import { MemoryWorkspace, watch } from "../../harness/workspace.js";
import { PreviewServer } from "../../harness/preview.js";
import { buildPreviewDoc } from "../../harness/preview-build.js";
import { runJsTool, runSnapshot, formatRun, browserRunner, RUN_LINES, RUN_CHARS, GRACE_MS } from "../../harness/run-js.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const dec = new TextDecoder();
const project = () => watch(new MemoryWorkspace({
  "index.html": "<!doctype html><body><canvas id=c></canvas><script type=module src=game.js></script></body>",
  "game.js": "export const add = (a, b) => a + b;\n",
  "sub/page.html": "<p>x</p>",
}));

Deno.test("runSnapshot: a blank page or the given page, with the loader and __run.js next to it", () => {
  const files = new Map([["index.html", { bytes: new TextEncoder().encode("<body><p>hi</p></body>") }], ["game.js", { bytes: new TextEncoder().encode("export const x = 1;") }]]);
  const a = runSnapshot({ files }, { code: "import { x } from './game.js'; console.log(x);" });
  eq(a.entry, "__run.html");
  ok(dec.decode(a.files.get("__run.js").bytes).includes("console.log(x)"));
  const b = runSnapshot({ files }, { code: "1", page: "index.html" });
  const html = dec.decode(b.files.get("index.html").bytes);
  ok(/<p>hi<\/p><script type="module">[\s\S]*import\("\.\/__run\.js"\)[\s\S]*<\/script><\/body>/.test(html), html);
  ok(!dec.decode(files.get("index.html").bytes).includes("__run"), "the project's own snapshot is untouched");
  // built like any preview: the snippet's import of game.js and the loader's import are data URLs
  const { html: doc, missing } = buildPreviewDoc(a, { nonce: "n" });
  eq(missing, []);
  ok(!/"\.\/__run\.js"/.test(doc) && !/from '\.\/game\.js'/.test(doc), "imports rewritten");
  ok(doc.includes("__pvDone"), "the capture script offers the done hook");
});

Deno.test("run_js: runs through the runner with the project's current files; result format", async () => {
  const ws = project(), s = new PreviewServer(ws);
  let got = null;
  const runner = async (snap, { timeout }) => { got = { snap, timeout }; return { logs: [{ level: "log", text: "3", src: "__run.js", line: 1, col: 9, ms: 12 }], done: { ok: true, ms: 4 } }; };
  const t = runJsTool(s, { runner });
  eq(await t.run({ code: "import { add } from './game.js'; console.log(add(1, 2));" }), "ok in 4 ms\n3", "printed values as they are");
  eq(got.timeout, 3000);
  ok(got.snap.files.has("game.js") && got.snap.files.has("__run.js"));
  eq(await t.run({ code: "1", page: "nope.html" }), "error: no nope.html in the project (page is a path from the project root)");
  eq(await t.run({ code: " " }), "error: code is empty");
  await t.run({ code: "1", page: "./sub/page.html" });
  ok(got.snap.files.has("sub/__run.js") && got.snap.entry === "sub/page.html", "next to the page");
  s.close();
});

Deno.test("formatRun: ok without output, errors, timeouts, caps", () => {
  eq(formatRun({ logs: [], done: { ok: true, ms: 3 } }), "ok in 3 ms (no output; await async work and print with console.log)");
  eq(formatRun({ logs: [{ level: "error", text: "Error: bad\n    at f (lib.js:4:3)", src: "__run.js", line: 2, col: 1, ms: 5 }], done: { ok: false, ms: 5 } }),
    "error in 5 ms\n[0.0s] error __run.js:2:1 Error: bad\n  at f (lib.js:4:3)");
  eq(formatRun({ logs: [{ level: "error", text: "Error: no board\n    at bad (lib/t.js:5:9)\n    at __run.js:2:1", src: "__run.html", line: 127, col: 5, ms: 1 }], done: { ok: false, ms: 1 } }),
    "error in 1 ms\n[0.0s] error lib/t.js:5:9 Error: no board\n  at bad (lib/t.js:5:9)\n  at __run.js:2:1", "a throw is placed at its first stack frame, not the loader");
  eq(formatRun({ logs: [{ level: "error", text: "late", ms: 1 }], done: { ok: true, ms: 1 } }).split("\n")[0], "error in 1 ms", "an error logged by the page counts");
  eq(formatRun({ logs: [], done: null }), "timed out after 3 s (a loop that never ends, or an await that never resolves?)");
  eq(formatRun({ logs: [], done: null, hung: true }), "timed out after 3 s (the page hung: a loop that never ends?)");
  const many = formatRun({ logs: Array.from({ length: 100 }, (_, i) => ({ level: "log", text: "line " + i, ms: 1 })), done: { ok: true, ms: 1 } }).split("\n");
  eq(many.length, RUN_LINES + 2); eq(many.at(-1), `(${100 - RUN_LINES} more lines)`);
  const big = formatRun({ logs: Array.from({ length: 10 }, (_, i) => ({ level: "log", text: i + "x".repeat(900), ms: 1 })), done: { ok: true, ms: 1 } });
  ok(big.length < RUN_CHARS + 40, "char cap " + big.length);
  ok(big.split("\n")[2].endsWith("…(cut)"), "the row that overflows is cut, not dropped");
  const tailErr = formatRun({ logs: [...Array.from({ length: 40 }, (_, i) => ({ level: "log", text: "row " + i + " " + "y".repeat(60), ms: 1 })), { level: "error", text: "Error: last", src: "__run.js", line: 9, col: 1, ms: 2 }], done: { ok: false, ms: 2 } }).split("\n");
  ok(/Error: last$/.test(tailErr.at(-2)) && /more lines\)$/.test(tailErr.at(-1)), "the last error is always kept: " + tailErr.slice(-3).join(" | "));
  eq(formatRun({ logs: [{ level: "error", text: "Error: x\n    at f (lib.js:4:3)", src: "sub/page.html", line: 30, col: 1, ms: 1 }], done: { ok: false, ms: 1 } }, 3000, "sub/page.html").split("\n")[1],
    "[0.0s] error lib.js:4:3 Error: x", "a throw from the loader in a given page is placed at its stack frame too");
  eq(formatRun({ nohost: true }), "error: run_js needs the isolated preview host, which this page does not have");
  eq(formatRun({ stopped: true }), "stopped by the user");
});

Deno.test("run_js: page resolves from the project root, else from a served folder", async () => {
  const ws = watch(new MemoryWorkspace({ "game/index.html": "<body></body>", "game/logic.js": "export const x = 1;" })), s = new PreviewServer(ws);
  let got = null;
  const runner = async (snap) => { got = snap; return { logs: [], done: { ok: true, ms: 1 } }; };
  const t = runJsTool(s, { runner });
  await t.run({ code: "1", page: "game/index.html" });
  eq([got.dir, got.entry], ["", "game/index.html"]);
  ok(got.files.has("game/__run.js"));
  await s.serve({ dir: "game", port: 5173 });
  await t.run({ code: "1", page: "index.html" });
  eq([got.dir, got.entry], ["game", "index.html"], "the served folder's page");
  s.close();
});

// a fake mountPreview: the test drives its callbacks
function fakeDoc() {
  const el = () => ({ style: {}, append() {}, remove() { this.removed = true; } });
  return { createElement: el, body: el() };
}
Deno.test("browserRunner: one result; done waits GRACE_MS for late output; a stale done token is ignored; hung; nohost; abort", async () => {
  const doc = fakeDoc();
  let cb = null, destroyed = 0;
  const mount = (_el, _src, _port, o) => { cb = o; return { destroy() { destroyed++; } }; };
  const snap = runSnapshot({ files: new Map() }, { code: "1" });
  const R = browserRunner(doc, { mount });
  let p = R(snap, { timeout: 1000 });
  ok(cb.run, "a run frame");
  cb.onShow(); cb.onShow();   // shown again after a nav: one timer
  cb.onDone({ ok: true, ms: 2, tok: "not-this-run" });
  cb.onDone({ ok: true, ms: 3, tok: snap.runTok });
  cb.onLog({ level: "error", text: "late tick", src: "game.js" });   // inside the grace window
  cb.onStatus({ state: "hung" });   // after done: the result is still done
  let r = await p;
  eq(r.done, { ok: true, ms: 3 });
  ok(!r.hung || r.done, "done wins");
  eq(r.logs.map((e) => e.text), ["late tick"]);
  await new Promise((q) => setTimeout(q, 0));
  eq(destroyed, 1);
  ok(GRACE_MS >= 100);

  p = R(snap, { timeout: 1000 }); cb.onShow(); cb.onStatus({ state: "hung" }); cb.onDone({ ok: true, ms: 1, tok: snap.runTok });
  r = await p; eq([r.done, r.hung], [null, true]);
  p = R(snap, { timeout: 1000 }); cb.onStatus({ state: "nohost" });
  r = await p; ok(r.nohost);
  const ac = new AbortController();
  p = R(snap, { timeout: 5000, signal: ac.signal }); cb.onShow(); ac.abort();
  r = await p; ok(r.stopped, "Stop ends the run at once");
  p = R(snap, { timeout: 30 }); cb.onShow();
  r = await p; eq([r.done, r.hung], [null, false], "timeout");
});
