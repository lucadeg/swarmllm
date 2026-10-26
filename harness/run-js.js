// run_js: the agent runs a JS module in the preview sandbox and reads what it printed
// (docs/design/harness-light.md B.2). It checks logic ("does clearLines() drop a full row?"),
// not only "the page has no console errors".
//
// The project is snapshotted like serve does (without registering a port or touching the visible
// preview), the snippet is added as __run.js next to the page, and a loader module that imports it
// goes at the end of that page (or of a blank page). The document is built by buildPreviewDoc
// (imports rewritten, capture script, CSP) and mounted by mountPreview in a hidden, fresh,
// sandboxed frame, always through the relay (another site, so another process): `while (true) {}`
// cannot freeze the room. Without a relay there is no run_js (runJsAvailable(); the runner answers
// "nohost"). Console output and errors are collected until the loader says done plus GRACE_MS
// (a timer or the first frame that throws right after), or for TIMEOUT_MS.
//
//   runJsTool(server, { runner }) -> the tool; runner(snapshot, { timeout, signal }) ->
//     { logs: [{ level, text, src, line, col, ms }], done: { ok, ms } | null, hung, nohost?, stopped? }
//   runSnapshot(snapshot, { code, page }) -> the snapshot to build (pure; unit-tested)
//
// The eval suite's checks (tests/eval/) run through the same runner:
//   probeSnapshot(ws, dir?) -> { dir, entry, files }   the project as serve would see it
//   runProbe(snap, { code, page, timeout, width, height, doc, runner })
//     -> { status: "ok"|"error"|"timeout", ms, logs, done, hung, timeout }
//   formatProbe(result) -> formatRun's text
import { mountPreview, relayUrl } from "./preview-frame.js";
import { fold } from "./preview-tools.js";
import { mimeFor } from "./preview-build.js";
import { PreviewServer } from "./preview.js";

export const TIMEOUT_MS = 3000, GRACE_MS = 150, RUN_LINES = 30, RUN_CHARS = 1200;
const enc = new TextEncoder(), dec = new TextDecoder();
const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/") + 1) : "");
const base = (p) => String(p || "").split("/").pop();

// run_js only where the snippet runs in another process than the room
export const runJsAvailable = (doc = globalThis.document) => !!relayUrl(doc);

// Waits for the page's load (at most 1 s: a page can hang its own load), imports the snippet, and
// reports done through the capture script (window.__pvDone) with this run's token, so a page that
// calls the hook itself does not end the run.
const loader = (tok) => `<script type="module">
const done = (ok, t0) => window.__pvDone?.(ok, Math.round(performance.now() - t0), "${tok}");
if (document.readyState !== "complete") await Promise.race([new Promise((r) => addEventListener("load", r, { once: true })), new Promise((r) => setTimeout(r, 1000))]);
await new Promise((r) => setTimeout(r, 0));
const t0 = performance.now();
try { await import("./__run.js"); done(true, t0); } catch (e) { console.error(e); done(false, t0); }
</script>`;

export function runSnapshot(snap, { code, page = "" }) {
  const files = new Map(snap.files);
  const runTok = Array.from(crypto.getRandomValues(new Uint32Array(2)), (x) => x.toString(36)).join("");
  let entry = page;
  let html;
  if (page) html = dec.decode(files.get(page).bytes);
  else { entry = "__run.html"; html = "<!doctype html><html><head><meta charset=\"utf-8\"></head><body></body></html>"; }
  const L = loader(runTok);
  const i = html.search(/<\/body\s*>(?![\s\S]*<\/body\s*>)/i);
  html = i >= 0 ? html.slice(0, i) + L + html.slice(i) : html + L;
  files.set(entry, { type: "text/html", bytes: enc.encode(html), hash: "run" });
  files.set(dirOf(entry) + "__run.js", { type: mimeFor("x.js"), bytes: enc.encode(code), hash: "run" });
  return Object.freeze({ port: 0, dir: snap.dir || "", entry, rev: 1, bytes: 0, files, runTok });
}

// The browser runner: a hidden run frame through mountPreview (relay only), destroyed when done
// (+ GRACE_MS), after `timeout`, on a hang, or when `signal` aborts. `mount` is for tests.
export function browserRunner(doc = globalThis.document, { width = 480, height = 360, mount = mountPreview } = {}) {
  return (snap, { timeout = TIMEOUT_MS, signal = null } = {}) => new Promise((resolve) => {
    const el = doc.createElement("div");
    // on screen (throttled rendering off-screen would stop requestAnimationFrame) but invisible
    el.style.cssText = `position:fixed;left:0;top:0;width:${width}px;height:${height}px;opacity:0;pointer-events:none;z-index:-1;overflow:hidden`;
    doc.body.append(el);
    const logs = [];
    let view = null, timer = 0, grace = 0, fin = false, done = null;
    const end = (r) => {
      if (fin) return;
      fin = true; clearTimeout(timer); clearTimeout(grace);
      signal?.removeEventListener("abort", onAbort);
      queueMicrotask(() => { view?.destroy(); el.remove(); });
      resolve({ logs, done, hung: false, ...r });
    };
    const onAbort = () => { done = null; end({ stopped: true }); };
    if (signal?.aborted) return onAbort();
    signal?.addEventListener("abort", onAbort, { once: true });
    const source = {
      snapshot: () => snap, onUpdate: () => () => {}, attach: () => () => {}, frameEvent() {},
      pushLog() {},
    };
    view = mount(el, source, 0, {
      run: true,
      onLog: (e) => { if (fin) return; if (e.src !== "(preview)") logs.push(e); else if (/hung/.test(e.text)) end(done ? {} : { hung: true }); },
      onStatus: (s) => {
        if (s.state === "hung") end(done ? {} : { hung: true });
        else if (s.state === "nohost") end({ nohost: true });
      },
      onShow: () => { if (!timer) timer = setTimeout(() => end({}), timeout); },
      onDone: (d) => {
        if (done || fin || d?.tok !== snap.runTok) return;
        done = { ok: !!d.ok, ms: Math.max(0, Number(d.ms) || 0) };
        grace = setTimeout(() => end({}), GRACE_MS);
      },
    });
  });
}

// the snapshot run_js builds: the project root (or, when it is too big or has no such page, a
// served port's folder), and `page` relative to it
async function pickFiles(server, page) {
  let err = null;
  for (const dir of ["", ...server.ports().map((p) => p.dir).filter(Boolean).reverse()]) {
    let files;
    try { files = await server._read(dir, null, null); } catch (e) { err = e; continue; }
    const rel = dir && page.startsWith(dir + "/") ? page.slice(dir.length + 1) : page;
    if (!page || files.has(rel)) return { dir, files, page: rel };
  }
  return { error: page ? `error: no ${page} in the project (page is a path from the project root)` : `error: ${err?.message || "no files"}` };
}

export function runJsTool(server, { runner = null, timeout = TIMEOUT_MS } = {}) {
  return {
    name: "run_js", mutates: false,
    description: "Run a JS module in the sandbox, next to page (loaded first) if given, else at the project root; it can import project files. Use top-level await; print with console.log.",
    parameters: { type: "object", properties: { code: { type: "string" }, page: { type: "string" } }, required: ["code"] },
    async run({ code = "", page = "" } = {}, { signal } = {}) {
      if (!String(code).trim()) return "error: code is empty";
      const f = await pickFiles(server, String(page || "").replace(/^\.?\//, ""));
      if (f.error) return f.error;
      const snap = runSnapshot({ dir: f.dir, files: f.files }, { code: String(code), page: f.page });
      const r = await (runner || browserRunner())(snap, { timeout, signal });
      return formatRun(r, timeout, snap.entry);
    },
  };
}

export function formatRun({ logs = [], done = null, hung = false, nohost = false, stopped = false }, timeout = TIMEOUT_MS, entry = "__run.html") {
  if (nohost) return "error: run_js needs the isolated preview host, which this page does not have";
  if (stopped) return "stopped by the user";
  const errs = logs.some((e) => e.level === "error");
  const head = done ? `${done.ok && !errs ? "ok" : "error"} in ${done.ms} ms`
    : `timed out after ${timeout / 1000} s${hung ? " (the page hung: a loop that never ends?)" : " (a loop that never ends, or an await that never resolves?)"}`;
  if (!logs.length) return done?.ok ? `${head} (no output; await async work and print with console.log)` : head;
  // printed values as they are; warnings and errors with where they came from
  const rows = fold(logs.map((e) => ({ ...e, t: e.ms ?? 0, rev: 0, ...atThrow(e, entry) })))
    .map((r) => (r.e.level === "log" || r.e.level === "info" ? { ...r, text: r.e.text + (r.n > 1 ? ` ×${r.n}` : "") } : r));
  // within the caps: the row that overflows is cut, not dropped, and the last error is always kept
  const lastErr = rows.findLastIndex((r) => r.e.level === "error");
  const clip = (t, n) => (t.length <= n ? t : t.slice(0, Math.max(0, n - 6)) + "…(cut)");
  const out = [head];
  let size = head.length, n = 0, skipped = 0;
  for (let i = 0; i < rows.length; i++) {
    const keepErr = lastErr > i ? Math.min(rows[lastErr].text.length, 400) + 1 : 0;   // room saved for it
    const room = RUN_CHARS - size - keepErr - 1;
    if (i === lastErr || (n < RUN_LINES && room >= 40)) {
      const t = clip(rows[i].text, i === lastErr ? Math.max(400, RUN_CHARS - size - 1) : room);
      out.push(t); size += t.length + 1; n++;
    } else skipped++;
  }
  if (skipped) out.push(`(${skipped} more lines)`);
  return out.join("\n");
}

// the loader logs a thrown error from its own line in the entry page: name the stack's first frame
function atThrow(e, entry) {
  if (e.level !== "error" || !e.src || base(e.src) !== base(entry)) return {};
  const m = /^\s+at .*?\(?([^\s()]+):(\d+):(\d+)\)?\s*$/m.exec(e.text || "");
  return m ? { src: m[1], line: +m[2], col: +m[3] } : {};
}

// ---- the eval suite's probe: the same runner, with a status for pass/fail
export async function probeSnapshot(ws, dir = "") {
  const s = new PreviewServer(ws);
  try { return { dir, entry: "index.html", files: await s._read(dir, null, null) }; } finally { s.close(); }
}

export async function runProbe(snap, { code, page = null, timeout = TIMEOUT_MS, width = 800, height = 600, doc = globalThis.document, runner = null } = {}) {
  const t0 = performance.now();
  page = page && snap.files.has(page) ? page : "";
  const run = runSnapshot(snap, { code: String(code), page });
  const r = await (runner || browserRunner(doc, { width, height }))(run, { timeout });
  const errs = r.logs.some((e) => e.level === "error");
  const status = r.done ? (r.done.ok && !errs ? "ok" : "error") : "timeout";
  return { ...r, status, ms: r.done ? r.done.ms : Math.round(performance.now() - t0), timeout, entry: run.entry };
}

export const formatProbe = (r) => formatRun(r, r.timeout ?? TIMEOUT_MS, r.entry);
