// Argument repair for tool calls from small models (Qwen3 1.7B writes JSON calls with the right
// intent but loose shapes): the agent runs every call's arguments through fixArgs() before the
// tool sees them, so a slip like {"file": "./index.html", "contents": ["<h1>", "hi"]} or
// {"append": "false"} does the obvious thing instead of failing and sending the model into a
// rewrite loop. Only fills a parameter the call left out, never overrides one it gave; values of
// the wrong type are converted when the conversion is unambiguous, else dropped (the tool's
// default applies). DOM-free, schema driven, the same for every model.

// other names models use for a parameter, most likely first. A key that is itself one of the
// tool's parameters is never taken as an alias of another.
export const ARG_ALIASES = {
  path: ["file", "filename", "file_name", "filepath", "file_path", "fname", "name", "dir", "directory", "folder"],
  content: ["contents", "text", "code", "body", "data", "html", "source", "file_content", "value", "lines"],
  old: ["old_string", "old_text", "old_str", "search", "find", "original", "before", "from"],
  new: ["new_string", "new_text", "new_str", "replace", "replacement", "after", "to", "with", "updated"],
  dir: ["path", "folder", "root", "directory", "dir_path", "root_dir", "base", "base_dir", "project"],
  entry: ["file", "page", "index", "main", "entry_point", "entrypoint", "html", "filename", "start"],
  port: ["port_number", "portnumber", "p"],
  pattern: ["query", "regex", "search", "term", "q", "text", "keyword", "string"],
  start_line: ["start", "from_line", "line", "offset", "first_line", "begin", "line_start"],
  end_line: ["end", "to_line", "last_line", "stop", "line_end"],
  code: ["script", "js", "javascript", "source", "content", "snippet"],
  page: ["entry", "html", "file"],
  since: ["cursor", "from", "after"],
  ignore_case: ["case_insensitive", "insensitive", "i", "nocase"],
  append: ["add", "append_mode"],
};
// parameters that hold a path: trimmed, quotes and a leading "./" or "/" dropped
const PATHY = new Set(["path", "dir", "entry", "page"]);

// "startLine" / "Start-Line" / "START_LINE" -> "start_line"
const snake = (k) => String(k).replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/[\s-]+/g, "_").toLowerCase();

// a path as models write it: ` "./index.html" `, `/index.html`, `file:///index.html`
export function cleanPathArg(v) {
  let s = String(v).trim().replace(/^(['"`])(.*)\1$/s, "$2").trim();
  s = s.replace(/^file:\/\/\/?/i, "").replace(/^\.\/+/, "").replace(/^\/+/, "");
  return s === "." ? "" : s;
}
// "true" / "yes" / "1" / 1 -> true, the opposites -> false, else undefined
export function toBool(v) {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v === 1 ? true : v === 0 ? false : undefined;
  const s = String(v ?? "").trim().toLowerCase();
  return ["true", "yes", "1", "on", "y"].includes(s) ? true : ["false", "no", "0", "off", "n", "none", "null", ""].includes(s) ? false : undefined;
}
// "5173" / ":5173" / "localhost:5173" / 5173.0 -> 5173; else undefined
export function toInt(v) {
  if (typeof v === "number") return Number.isFinite(v) ? Math.round(v) : undefined;
  const m = /-?\d+/.exec(String(v ?? "").replace(/^.*:(?=\d+\D*$)/, ""));
  return m ? Number(m[0]) : undefined;
}
// a string parameter given as something else: lines joined, objects as JSON
export function toText(v) {
  if (typeof v === "string") return v;
  if (v == null) return undefined;
  if (Array.isArray(v)) return v.every((x) => typeof x === "string" || typeof x === "number") ? v.join("\n") + (v.length ? "\n" : "") : JSON.stringify(v, null, 2);
  if (typeof v === "object") return JSON.stringify(v, null, 2);
  return String(v);
}

// fixArgs(args, parameters) -> a plain object of arguments for a tool with that JSON schema
export function fixArgs(args, parameters) {
  const props = parameters?.properties || {}, names = Object.keys(props);
  let a = args;
  if (typeof a === "string") {
    const t = a.trim();
    let parsed;
    if (/^[{[]/.test(t)) { try { parsed = JSON.parse(t); } catch { /* text */ } }
    if (parsed !== undefined) a = parsed;
    else {
      // a bare value: it is the tool's first required (else first) parameter
      const first = parameters?.required?.[0] ?? names[0];
      a = first && t ? { [first]: a } : {};
    }
  }
  // positional: ["index.html", "<h1>hi</h1>", "false"] in the schema's order
  if (Array.isArray(a)) a = Object.fromEntries(a.slice(0, names.length).map((v, i) => [names[i], v]));
  if (!a || typeof a !== "object") return {};
  // one level of wrapping: {"arguments": {...}}, {"write_file": {...}}
  const keys = Object.keys(a);
  if (keys.length === 1 && !names.includes(keys[0]) && a[keys[0]] && typeof a[keys[0]] === "object" && !Array.isArray(a[keys[0]])) a = a[keys[0]];
  const out = {};
  // exact names, then names that differ only in case / style
  for (const [k, v] of Object.entries(a)) {
    if (names.includes(k)) out[k] = v;
    else { const s = snake(k); if (names.includes(s) && !(s in a)) out[s] = v; }
  }
  // aliases for what is still missing
  const used = new Set(Object.keys(out));
  for (const n of names) {
    if (n in out) continue;
    for (const al of ARG_ALIASES[n] || []) {
      const k = Object.keys(a).find((x) => !used.has(x) && !names.includes(x) && !names.includes(snake(x)) && snake(x) === al);
      if (k === undefined) continue;
      out[n] = a[k]; used.add(k);
      break;
    }
  }
  // {"mode": "append"} means append: true
  if ("append" in props && !("append" in out) && /^append/i.test(String(a.mode ?? ""))) out.append = true;
  // types
  for (const n of Object.keys(out)) {
    const ty = props[n]?.type, v = out[n];
    let w = v;
    if (v == null) w = undefined;
    else if (ty === "boolean") w = toBool(v);
    else if (ty === "integer" || ty === "number") w = toInt(v);
    else if (ty === "string") w = toText(v);
    if (w !== undefined && PATHY.has(n) && typeof w === "string") w = cleanPathArg(w);
    if (w === undefined) delete out[n]; else out[n] = w;
  }
  return Object.fromEntries(names.filter((n) => n in out).map((n) => [n, out[n]]));   // in the schema's order
}

// Tool names models reach for, mapped to the tool that does it (only when that tool exists and
// the name itself is not a tool).
export const TOOL_ALIASES = {
  write_file: ["create_file", "write", "save_file", "save", "create", "new_file", "writefile", "write_to_file", "file_write", "make_file", "update_file", "overwrite_file"],
  read_file: ["read", "open_file", "cat", "view_file", "view", "readfile", "get_file", "show_file", "open"],
  list_dir: ["ls", "list", "list_files", "listdir", "list_directory", "dir", "files", "list_folder"],
  edit_file: ["edit", "replace", "str_replace", "replace_in_file", "modify_file", "patch", "update", "change_file", "str_replace_editor"],
  serve: ["preview", "start_server", "run_server", "serve_folder", "open_preview", "start", "run", "server", "host", "launch", "deploy", "open_in_browser", "serve_file"],
  search: ["grep", "find", "search_files", "find_in_files", "rg"],
  preview_logs: ["logs", "console", "get_logs", "read_logs", "console_logs"],
  run_js: ["run_javascript", "exec_js", "eval", "execute", "node", "run_code"],
};
export function fixToolName(name, has) {
  if (typeof name !== "string") return name;
  if (has(name)) return name;
  const s = snake(name.trim());
  if (has(s)) return s;
  for (const [t, al] of Object.entries(TOOL_ALIASES)) if (al.includes(s) && has(t)) return t;
  return name;
}
