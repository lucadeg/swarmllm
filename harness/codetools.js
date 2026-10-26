// Coding tools for the Code mode agent over a workspace (harness/workspace.js). Each tool is
// { name, description, parameters (JSON schema), mutates, run(args, ctx) -> string,
//   preview?(args) -> { path, before, after } }.
// Results are plain text written for the model: short, with line numbers, capped so one call
// never eats the context (a 16k window with Code mode's 8k answer cap leaves room for ~12k of
// conversation), and with errors that say how
// to recover. Tools that change files are marked `mutates` so the agent can ask the user first;
// preview() gives the approval card its before/after without touching the file.
//
// codingTools(ws, { server }): with a PreviewServer, a write under a served folder waits for the
// preview's new snapshot and says its rev (or why it did not update), so the model knows to check
// preview_logs rather than serve again.
import { normPath } from "./workspace.js";

export const MAX_PATH = 200, SEARCH_MS = 2000;
export const MAX_LINES = 200, MAX_CHARS = 8000, MAX_LINE = 1000, MAX_HITS = 30, HIT_CHARS = 160, MAX_ENTRIES = 200;

const kb = (n) => (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`);
const cut = (s, n) => (s.length > n ? s.slice(0, n) + `…(${s.length - n} chars cut)` : s);
const lineCount = (t) => (t === "" ? 0 : t.split("\n").length - (t.endsWith("\n") ? 1 : 0));

// pi-style normalisation for a second try: NFKC, smart quotes / dashes / odd spaces to ASCII,
// trailing whitespace dropped
const fuzz = (l) => l.normalize("NFKC").replace(/[\u2018\u2019\u201A\u201B]/g, "'").replace(/[\u201C\u201D\u201E\u201F]/g, '"')
  .replace(/[\u2010-\u2015\u2212]/g, "-").replace(/[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g, " ").replace(/\s+$/, "");
const indentOf = (l) => /^[ \t]*/.exec(l)[0];
const asLines = (s) => { const l = s.split("\n"); if (l.length > 1 && l[l.length - 1] === "") l.pop(); return l; };
// start indexes of the windows of `F` whose lines equal `O`'s under `eq`
function windows(F, O, eq) {
  const at = [];
  for (let i = 0; i + O.length <= F.length && at.length < 2; i++) {
    let k = 0;
    while (k < O.length && eq(F[i + k], O[k])) k++;
    if (k === O.length) at.push(i);
  }
  return at;
}
// "lines 40-46 are closest:" + those lines numbered: the window sharing the most trimmed lines with old
function closest(F, O) {
  const want = new Set(O.map((l) => l.trim()).filter(Boolean));
  if (!want.size) return "";
  let best = -1, score = 0;
  for (let i = 0; i + Math.min(O.length, F.length) <= F.length; i++) {
    let k = 0;
    for (let j = i; j < Math.min(F.length, i + O.length); j++) if (want.has(F[j].trim())) k++;
    if (k > score) { score = k; best = i; }
  }
  if (best < 0) return "";
  const from = Math.max(0, best - (O.length < 3 ? 1 : 0)), to = Math.min(F.length, from + Math.max(O.length + (O.length < 3 ? 2 : 0), 1), from + 12);
  return `; lines ${from + 1}-${to} are closest:\n` + F.slice(from, to).map((l, i) => `${from + i + 1}|${cut(l, 200)}`).join("\n");
}

// read_file's "12|" line prefixes, which are not part of the file
const NUMBERED = /^[ \t]*\d+\|/;
export const unnumber = (t) => t.split("\n").map((l) => l.replace(NUMBERED, "")).join("\n");

// edit_file's search/replace; also used by its preview so the card shows exactly what run does.
// An exact match first; else whole lines equal after fuzz(), else after trim() (the new text then
// takes the file's indentation); else an error naming the closest lines.
function applyEdit(text, path, o, n) {
  if (!o) return { error: "error: old is empty; use write_file to create or replace a whole file" };
  // old copied from read_file with its "12|" line numbers (Qwen3 1.7B): match without them
  if (!text.includes(o) && NUMBERED.test(o) && o.split("\n").every((l) => !l.trim() || NUMBERED.test(l))) {
    const r = applyEdit(text, path, unnumber(o), unnumber(n));
    return r.error ? r : { ...r, how: [r.how, "the line numbers"].filter(Boolean).join(" and ") };
  }
  if (o === n) return { error: "error: old and new are the same" };
  const k = text.split(o).length - 1;
  if (k > 1) return { error: `error: old appears ${k} times in ${path}; include more surrounding lines so it is unique` };
  if (k === 1) {
    const at = text.indexOf(o);
    return { after: text.slice(0, at) + n + text.slice(at + o.length), line: text.slice(0, at).split("\n").length };
  }
  const F = text.split("\n"), O = asLines(o);
  for (const [eq, how] of [[(a, b) => fuzz(a) === fuzz(b), "whitespace"], [(a, b) => a.trim() === b.trim(), "indentation"]]) {
    if (how === "indentation" && !O.some((l) => l.trim())) continue;   // (blank lines match blank lines; old needs one that is not)
    const at = windows(F, O, eq);
    if (at.length > 1) return { error: `error: old appears more than once in ${path} (ignoring ${how}); include more surrounding lines so it is unique` };
    if (!at.length) continue;
    const i = at[0];
    let N = n === "" ? [] : asLines(n);
    if (how === "indentation") {
      const j = O.findIndex((l) => l.trim()), fi = indentOf(F[i + j]), oi = indentOf(O[j]);
      if (fi !== oi) N = N.map((l) => (l && l.startsWith(oi) ? fi + l.slice(oi.length) : l));
    }
    return { after: [...F.slice(0, i), ...N, ...F.slice(i + O.length)].join("\n"), line: i + 1, how };
  }
  const near = closest(F, O);
  return { error: `error: old not found in ${path}` + (near || "; read_file it again and copy the text exactly (whitespace included)") };
}

// a path the approval card shows exactly as it is written (normalised, not too long to read)
function cleanPath(p) {
  const n = normPath(p);
  if (!n) throw new Error("path is empty (pass the file's path, e.g. index.html)");
  if (n.length > MAX_PATH) throw new Error(`path is ${n.length} characters (max ${MAX_PATH})`);
  return n;
}
const isBinary = (u8, text) => u8.includes(0) || text.includes("\uFFFD");

// The search itself; runs in a Worker in a page (below) so a pathological pattern cannot freeze the
// room's tab. Serialized with toString(): must not close over anything.
export function grepFiles(files, pattern, flags, { maxLine, maxHits, hitChars }) {
  const re = new RegExp(pattern, flags), hits = [];
  let total = 0;
  for (const [f, text] of files) {
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i].length > maxLine ? lines[i].slice(0, maxLine) : lines[i];
      if (!re.test(l)) continue;
      if (++total <= maxHits) { const t = lines[i].trim(); hits.push(`${f}:${i + 1}: ${t.length > hitChars ? t.slice(0, hitChars) + `…(${t.length - hitChars} chars cut)` : t}`); }
    }
    if (total > 2000) break;   // enough to say "(+N more)"; a runaway pattern stops here
  }
  return { hits, total };
}
// In a page: a Worker, terminated after `ms` (a catastrophic backtracking pattern otherwise holds
// the main thread, and with it the room, for minutes). Elsewhere (Deno tests): inline.
function grep(files, pattern, flags, ms) {
  const opts = { maxLine: MAX_LINE, maxHits: MAX_HITS, hitChars: HIT_CHARS };
  if (typeof document === "undefined" || typeof Worker === "undefined") return Promise.resolve(grepFiles(files, pattern, flags, opts));
  const src = `const grepFiles = ${grepFiles.toString()};\nonmessage = (e) => { try { postMessage(grepFiles(e.data.files, e.data.pattern, e.data.flags, e.data.opts)); } catch (err) { postMessage({ error: String(err && err.message || err) }); } };`;
  const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
  const w = new Worker(url);
  return new Promise((res) => {
    const fin = (v) => { clearTimeout(t); w.terminate(); URL.revokeObjectURL(url); res(v); };
    const t = setTimeout(() => fin({ slow: true }), ms);
    w.onmessage = (e) => fin(e.data);
    w.onerror = (e) => { e.preventDefault?.(); fin({ error: e.message || "search failed" }); };
    w.postMessage({ files, pattern, flags, opts });
  });
}

// a rewrite that kept most of a long file: " · 190 of 200 old lines unchanged" (harness/cards.js
// then suggests edit_file); "" otherwise
export function kept(old, text) {
  if (old == null) return "";
  const O = asLines(old);
  if (O.length <= 40) return "";
  const have = new Map();
  for (const l of asLines(text)) have.set(l, (have.get(l) || 0) + 1);
  let k = 0;
  for (const l of O) { const c = have.get(l); if (c) { k++; have.set(l, c - 1); } }
  return k >= 0.8 * O.length ? ` · ${k} of ${O.length} old lines unchanged` : "";
}

// the new text is a fragment of the HTML page it would replace: the page has <html>/<!doctype>
// and a <body>, the new text has no <html>/<!doctype> (a bare "<body>...</body>" counts) and is shorter
export function fragmentOf(old, text) {
  if (old == null) return false;
  const page = (t) => /<!doctype html|<html[\s>]/i.test(t) && /<body[\s>]/i.test(t);
  return page(old) && !/<!doctype html|<html[\s>]/i.test(text) && text.length < old.length && lineCount(old) >= 5;
}

// the new text starts with the file's own first two lines (non-trivial): a whole-file rewrite
export function sameStart(old, text) {
  const a = asLines(old), b = asLines(text);
  if (a.length < 2 || b.length < 2) return false;
  return a[0].trim().length >= 6 && a[0] === b[0] && a[1] === b[1];
}

// edit_file takes old/new or old_string/new_string (the XML parser already trims one newline
// at each end of a parameter)
const oldOf = (a) => a.old ?? a.old_string ?? "", newOf = (a) => a.new ?? a.new_string ?? "";

export function codingTools(ws, { server = null, searchMs = SEARCH_MS } = {}) {
  // after a write: wait for each served port's new snapshot, and say what the preview now runs
  const reloadNote = async (path) => {
    const ports = server?.servedPorts?.(path) || [];
    if (!ports.length) return "";
    const out = [];
    for (const port of ports) {
      const r = server.flush ? await server.flush(port) : null;
      out.push(r?.error ? `preview :${port} not updated: ${r.error}` : `preview :${port} reloaded${r?.rev ? ` (rev ${r.rev})` : ""}`);
    }
    return " · " + out.join(" · ");
  };
  const readOr = async (p) => ((await ws.exists(p)) ? ws.read(p) : null);
  // "the project has: a, b" / "the project is empty", for errors about a missing file
  const have = async () => {
    let fs = [];
    try { fs = await ws.walk(50); } catch { /* none */ }
    return fs.length ? `the project has: ${fs.slice(0, 8).join(", ")}${fs.length > 8 ? ` (+${fs.length - 8} more)` : ""}` : "the project is empty";
  };
  // normalised, and allowed by the workspace (a folder on disk refuses hidden and secret files)
  const clean = (p) => { const n = cleanPath(p); ws.check?.(n); return n; };
  return [
    {
      name: "list_dir", mutates: false,
      description: "List a folder (default: project root).",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      async run({ path = "" } = {}) {
        let es;
        try { es = await ws.list(path); }
        catch (e) {
          // a file named as a folder (small models list "index.html" to see if it exists)
          if (path && (await ws.exists(path).catch(() => false))) return `${normPath(path)} is a file, not a folder; read_file shows its lines`;
          if (/no such/i.test(e?.message || "")) return `error: there is no folder ${path} (${await have()})`;
          throw e;
        }
        if (!es.length) return "(empty)";
        const out = es.slice(0, MAX_ENTRIES).map((e) => e.name + (e.dir ? "/" : ""));
        if (es.length > MAX_ENTRIES) out.push(`(+${es.length - MAX_ENTRIES} more)`);
        return out.join("\n");
      },
    },
    {
      name: "read_file", mutates: false,
      description: `Read lines with numbers (max ${MAX_LINES} per call).`,
      parameters: { type: "object", properties: { path: { type: "string" }, start_line: { type: "integer" }, end_line: { type: "integer" } }, required: ["path"] },
      async run({ path, start_line, end_line }) {
        let u8;
        try { u8 = await ws.readBytes(path); }
        catch (e) {
          // the browser's own "A requested file or directory could not be found..." says nothing useful to the model
          if (e?.name === "NotFoundError" || /could not be found|not found|no such file/i.test(e?.message || "")) return `error: ${path} does not exist yet (${await have()}); create it with write_file`;
          throw e;
        }
        const text = new TextDecoder().decode(u8);
        if (isBinary(u8, text)) return `(${path} is a binary file, ${kb(u8.length)})`;
        if (text === "") return `(${path} is empty)`;
        const lines = text.split("\n");
        if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();   // a final newline is not a line
        const from = Math.max(1, Math.floor(start_line) || 1);
        if (from > lines.length) return `error: ${path} has ${lines.length} lines`;
        const last = Math.min(lines.length, Math.floor(end_line) || Infinity, from + MAX_LINES - 1);
        const out = [];
        let size = 0, to = from - 1;
        for (let i = from; i <= last; i++) {
          const l = `${i}|${cut(lines[i - 1], MAX_LINE)}`;
          if (out.length && size + l.length + 1 > MAX_CHARS) break;
          out.push(l); size += l.length + 1; to = i;
        }
        if (to < lines.length && !(end_line && to >= end_line)) out.push(`(lines ${from}-${to} of ${lines.length}; read on with start_line=${to + 1})`);
        return out.join("\n");
      },
    },
    {
      name: "search", mutates: false,
      description: "Regex search in the project (or under path); returns path:line: text.",
      parameters: { type: "object", properties: { pattern: { type: "string" }, path: { type: "string" }, ignore_case: { type: "boolean" } }, required: ["pattern"] },
      async run({ pattern, path = "", ignore_case = false }) {
        const flags = ignore_case ? "i" : "";
        try { new RegExp(pattern, flags); } catch (e) { return `error: bad pattern: ${e.message}`; }
        const pre = path && path !== "." ? path.replace(/\/+$/, "") + "/" : "";
        const files = [];
        for (const f of await ws.walk()) {
          if (pre && !f.startsWith(pre)) continue;
          let text;
          try { text = await ws.read(f); } catch { continue; }
          if (text.includes("\u0000") || text.includes("\uFFFD")) continue;   // binary
          files.push([f, text]);
        }
        const r = await grep(files, pattern, flags, searchMs);
        if (r.slow) return `error: pattern too slow (over ${searchMs / 1000} s); use a simpler pattern without nested repeats`;
        if (r.error) return `error: ${r.error}`;
        const { hits, total } = r;
        if (!hits.length) return "no matches";
        return hits.join("\n") + (total > MAX_HITS ? `\n(+${total - MAX_HITS}${total > 2000 ? "+" : ""} more; narrow the pattern or path)` : "");
      },
    },
    {
      name: "edit_file", mutates: true,
      description: "Replace text: old must match the file exactly, once.",
      parameters: { type: "object", properties: { path: { type: "string" }, old: { type: "string" }, new: { type: "string" } }, required: ["path", "old", "new"] },
      async preview(args) {
        let path;
        try { path = clean(args.path); } catch (e) { return { path: String(args.path ?? "").slice(0, MAX_PATH), before: null, after: "", error: `error: ${e.message}` }; }
        let before;
        try { before = await ws.read(path); } catch (e) { return { path, before: null, after: "", error: `error: ${e.message}` }; }
        const r = applyEdit(before, path, oldOf(args), newOf(args));
        return { path, before, after: r.error ? before : r.after, ...(r.error ? { error: r.error } : {}) };
      },
      async run(args) {
        const path = clean(args.path), o = oldOf(args), n = newOf(args);
        if (!(await ws.exists(path))) return `error: no such file: ${path}; use write_file to create it`;
        const r = applyEdit(await ws.read(path), path, o, n);
        if (r.error) return r.error;
        await ws.write(path, r.after);
        const a = o.split("\n").length, b = n.split("\n").length, end = r.line + Math.max(b, 1) - 1;
        return `edited ${path} ${end > r.line ? `lines ${r.line}-${end}` : `line ${r.line}`} (${a} -> ${b} lines${r.how ? `, matched ignoring ${r.how}` : ""})` + await reloadNote(path);
      },
    },
    {
      name: "write_file", mutates: true,
      description: "Create or replace a file; append: true adds to its end.",
      parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" }, append: { type: "boolean" } }, required: ["path", "content"] },
      async preview({ path, content = "", append = false }) {
        try { path = clean(path); } catch (e) { return { path: String(path ?? "").slice(0, MAX_PATH), before: null, after: "", error: `error: ${e.message}` }; }
        let before;
        try { before = await readOr(path); } catch (e) { return { path, before: null, after: "", error: `error: ${e.message}` }; }
        return { path, before, after: append && before != null && !sameStart(before, String(content)) ? before + content : content };
      },
      async run({ path, content, append = false }) {
        if (path == null || !String(path).trim()) return 'error: write_file needs a path, e.g. {"path": "index.html", "content": "..."}; nothing was written';
        path = clean(path);
        // a call with no content would empty the file (a small model's content under a name it made up)
        if (content == null) return `error: write_file needs content (the whole text of ${path}); nothing was written`;
        const old = await readOr(path).catch(() => null);
        // an "append" that starts like the file itself is the whole file again (a small model
        // rewriting after an error): replace, don't double the file
        const whole = append && old != null && sameStart(old, String(content));
        if (whole) append = false;
        const before = append ? old : null;
        const text = before != null ? before + content : String(content);
        // the same file again (a small model after a failed step): say so, so it moves on
        // a whole HTML page replaced by a piece of one (a small model "adding" a button with
        // write_file): refuse, the page would lose everything else
        if (!append && fragmentOf(old, text)) return `error: not written: ${path} is a whole page (${lineCount(old)} lines) and this content is only a piece of one (no <!doctype> or <html>). To add or change part of the page use edit_file; to replace it, write the whole page`;
        if (!append && old === text) return `unchanged: ${path} already has exactly this content. Do not write it again: serve it, or fix what is wrong with edit_file, or finish`;
        await ws.write(path, text);
        const size = `${lineCount(text)} lines, ${kb(new TextEncoder().encode(text).length)}`;
        return (before != null ? `appended to ${path} (now ${size})` : `wrote ${path} (${size})${whole ? "; it started like the file itself, so it replaced the file instead of being appended" : kept(old, text)}`) + await reloadNote(path);
      },
    },
  ];
}
