// The preview server: virtual ports over a workspace (docs/design/harness-app.md B.2, B.6, B.7).
// A port is only a key. serve({ dir, port, entry }) snapshots the files under `dir` and registers
// them as `port`; frames (harness/preview-frame.js) build a document from the snapshot and report
// console output back through pushLog, which the agent reads with preview_logs.
//
// Snapshot = { port, dir, entry, rev, bytes, files: Map<path, { type, bytes, hash }> }, immutable;
// paths are relative to dir. Writes under a served dir (through watch(ws)) re-snapshot 250 ms
// after the last one, re-reading only the written paths, and emit update { port, rev, changed }.
//
// PreviewSource (the part a frame needs; PreviewSubscriber on peers implements it too):
//   snapshot(port), onUpdate(fn), pushLog(port, entry), attach(port) -> detach, frameEvent(port, ev)
import { normPath, SKIP_DIRS } from "./workspace.js";
import { mimeFor } from "./preview-build.js";

export const DEFAULT_PORT = 5173;
const RING = 500;

export async function hashBytes(u8) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", u8));
  return [...h.slice(0, 10)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`);
const skipped = (p) => p.split("/").some((s) => s.startsWith(".") || SKIP_DIRS.has(s));
const under = (p, dir) => !dir || p === dir || p.startsWith(dir + "/");

export class PreviewServer {
  constructor(ws, { maxFiles = 400, maxFile = 2 << 20, maxBytes = 8 << 20, debounce = 250, frameWait = 500 } = {}) {
    this.ws = ws; this.limits = { maxFiles, maxFile, maxBytes }; this.debounce = debounce; this.frameWait = frameWait;
    this.served = new Map();   // port -> { snap, dirty: Set, timer, busy, logs: [], seq, frames, ready: {rev, ms, at}, idle: {rev, ms}, waiters: [] }
    this.lastRev = new Map();  // port -> rev, so a re-served port keeps counting up
    this.lastSeq = new Map();  // port -> log seq, so a cursor the agent holds stays valid after stop + serve
    this.fns = new Set();
    this.unwatch = ws.onChange ? ws.onChange((e) => this._changed(e)) : null;
  }
  ports() { return [...this.served.values()].map(({ snap: s }) => ({ port: s.port, dir: s.dir, entry: s.entry, rev: s.rev })).sort((a, b) => a.port - b.port); }
  snapshot(port) { return this.served.get(+port)?.snap || null; }
  servedPorts(path) { const p = normPath(path); return this.ports().filter((s) => under(p, s.dir)).map((s) => s.port); }
  onUpdate(fn) { this.fns.add(fn); return () => this.fns.delete(fn); }
  _emit(e) { for (const f of [...this.fns]) { try { f(e); } catch (err) { console.error(err); } } }

  async serve({ dir = "", port = DEFAULT_PORT, entry = "index.html" } = {}) {
    port = Number(port);
    if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error(`port must be an integer from 1024 to 65535, not ${port}`);
    dir = normPath(dir); entry = normPath(entry || "index.html");
    if (dir && !(await this.ws.exists(dir))) throw new Error(`no folder ${dir}`);
    const files = await this._read(dir, null, null);
    if (!files.has(entry)) throw new Error(`no ${entry} in ${dir || "the project root"}; write it first or pass entry`);
    const prev = this.served.get(port);
    if (prev) clearTimeout(prev.timer);
    const rev = (this.lastRev.get(port) || 0) + 1;
    this.lastRev.set(port, rev);
    const st = prev || { logs: [], seq: this.lastSeq.get(port) || 0, frames: 0, waiters: [] };
    Object.assign(st, { snap: this._snap(port, dir, entry, rev, files), dirty: new Set(), timer: 0, busy: null, ready: null, idle: null });
    this.served.set(port, st);
    this._emit({ port, rev, changed: [...files.keys()] });
    return st.snap;
  }
  stop(port) {
    const st = this.served.get(+port);
    if (!st) return false;
    clearTimeout(st.timer);
    this.served.delete(+port);
    this.lastSeq.set(+port, st.seq);
    for (const w of st.waiters.splice(0)) w(null);
    this._emit({ port: +port, rev: st.snap.rev, changed: [], stopped: true });
    return true;
  }
  // re-read every file (edits made outside this tab, e.g. on disk in a picked folder)
  refresh(port) { return this._resnap(+port, true); }
  // re-snapshot now instead of after the debounce (a tool about to say "reloaded"):
  // -> { rev } the port now serves, or { error } when the snapshot failed, or null (not served)
  async flush(port) {
    const st = this.served.get(+port);
    if (!st) return null;
    clearTimeout(st.timer);
    st.err = null;
    const snap = await this._resnap(+port, false);
    if (st.err) return { error: st.err };
    return { rev: (snap || this.served.get(+port)?.snap)?.rev || 0 };
  }
  close() { this.unwatch?.(); for (const p of [...this.served.keys()]) this.stop(p); }

  _snap(port, dir, entry, rev, files) {
    let bytes = 0;
    for (const f of files.values()) bytes += f.bytes.length;
    return Object.freeze({ port, dir, entry, rev, bytes, files });
  }
  // files under dir, reusing `old` entries for paths not in `dirty` (dirty null: read all)
  async _read(dir, old, dirty) {
    const pre = dir ? dir + "/" : "", L = this.limits, out = new Map();
    const paths = (await this.ws.walk()).filter((p) => under(p, dir) && p !== dir).map((p) => p.slice(pre.length)).filter((p) => !skipped(p));
    if (paths.length > L.maxFiles) throw new Error(`too many files under ${dir || "the project root"} (${paths.length}, max ${L.maxFiles}); serve a smaller folder`);
    let total = 0;
    for (const p of paths.sort()) {
      let f = old && !dirty?.has(p) ? old.get(p) : null;
      if (!f) {
        const bytes = await this.ws.readBytes(pre + p);
        if (bytes.length > L.maxFile) throw new Error(`${p} is ${size(bytes.length)} (max ${size(L.maxFile)} per file)`);
        f = { type: mimeFor(p), bytes, hash: await hashBytes(bytes) };
      }
      total += f.bytes.length;
      if (total > L.maxBytes) throw new Error(`the files under ${dir || "the project root"} are over ${size(L.maxBytes)}; serve a smaller folder`);
      out.set(p, f);
    }
    return out;
  }
  _changed({ path }) {
    for (const [port, st] of this.served) {
      if (!under(path, st.snap.dir)) continue;
      st.dirty.add(st.snap.dir ? path.slice(st.snap.dir.length + 1) : path);
      clearTimeout(st.timer);
      st.timer = setTimeout(() => this._resnap(port, false), this.debounce);
    }
  }
  async _resnap(port, all) {
    const st = this.served.get(port);
    if (!st) return null;
    if (st.busy) { await st.busy; return this._resnap(port, all); }   // one at a time; later writes land in the next
    const run = async () => {
      const dirty = st.dirty; st.dirty = new Set();
      const old = st.snap;
      let files;
      try { files = await this._read(old.dir, all ? null : old.files, all ? null : dirty); }
      catch (e) { st.err = e.message; this.pushLog(port, { level: "error", text: `preview not updated: ${e.message}`, src: "(server)" }); return null; }
      const changed = [];
      for (const [p, f] of files) if (old.files.get(p)?.hash !== f.hash) changed.push(p);
      for (const p of old.files.keys()) if (!files.has(p)) changed.push(p);
      if (!changed.length || this.served.get(port) !== st) return null;
      if (!files.has(old.entry)) this.pushLog(port, { level: "error", text: `${old.entry} was removed; the preview shows a 404 page`, src: "(server)" });
      const rev = this.lastRev.get(port) + 1;
      this.lastRev.set(port, rev);
      st.snap = this._snap(port, old.dir, old.entry, rev, files);
      st.ready = st.idle = null;
      this._emit({ port, rev, changed: changed.sort() });
      return st.snap;
    };
    st.busy = run();
    try { return await st.busy; } finally { st.busy = null; }
  }

  // ---- console: frames report here, the agent reads with logs()
  // entry: { level, text, src, line, col, ms (since the frame's load), rev (the rev it was built from) }
  pushLog(port, e) {
    const st = this.served.get(+port);
    if (!st) return;
    st.logs.push({ seq: ++st.seq, t: Math.max(0, Math.round(e.ms ?? e.t ?? 0)), rev: e.rev ?? st.snap.rev, level: e.level || "log", text: String(e.text ?? ""), src: e.src || "", line: e.line || 0, col: e.col || 0 });
    if (st.logs.length > RING) st.logs.splice(0, st.logs.length - RING);
  }
  // entries with seq > since, oldest first, and the cursor for the next call
  logs(port, since = 0) {
    const st = this.served.get(+port);
    if (!st) return { lines: [], next: since, dropped: 0 };
    if (since > st.seq) since = 0;   // a cursor from before a reload of the page: start over
    const lines = st.logs.filter((e) => e.seq > since);
    const first = st.logs[0]?.seq ?? st.seq + 1;
    return { lines, next: st.seq, dropped: Math.max(0, first - since - 1) };
  }
  cursor(port) { return this.served.get(+port)?.seq ?? this.lastSeq.get(+port) ?? 0; }

  // ---- frames: mountPreview attaches while it shows a port and reports ready/idle per rev
  attach(port) {
    const st = this.served.get(+port);
    if (!st) return () => {};
    st.frames++;
    let done = false;
    return () => { if (!done) { done = true; st.frames--; } };
  }
  hasFrame(port) { return (this.served.get(+port)?.frames || 0) > 0; }
  frameEvent(port, ev) {
    const st = this.served.get(+port);
    if (!st || ev.rev !== st.snap.rev) return;   // a frame still showing an older rev
    if (ev.t === "ready") st.ready = { rev: ev.rev, ms: ev.ms, at: Date.now() };
    if (ev.t === "idle") {
      st.idle = { rev: ev.rev, ms: ev.ms };
      for (const w of st.waiters.splice(0)) w({ loadedMs: st.ready?.ms ?? ev.ms });
    }
  }
  loadedAt(port) { return this.served.get(+port)?.ready?.at || 0; }
  // resolves when a mounted frame has run the current rev for 500 ms ({ loadedMs }), or null
  // after `ms`, or sooner when no frame attaches within frameWait (headless, pane closed)
  whenIdle(port, ms = 2000) {
    const st = this.served.get(+port);
    if (!st) return Promise.resolve(null);
    if (st.idle?.rev === st.snap.rev) return Promise.resolve({ loadedMs: st.ready?.ms ?? st.idle.ms });
    return new Promise((res) => {
      let t1 = 0, t2 = 0;
      const fin = (v) => { clearTimeout(t1); clearTimeout(t2); const i = st.waiters.indexOf(fin); if (i >= 0) st.waiters.splice(i, 1); res(v); };
      st.waiters.push(fin);
      t1 = setTimeout(() => fin(null), ms);
      if (this.frameWait < ms) t2 = setTimeout(() => { if (!st.frames) fin(null); }, this.frameWait);
    });
  }
}
