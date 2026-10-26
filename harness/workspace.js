// The files the Code mode agent works on. Two implementations with one interface:
//   MemoryWorkspace   - a Map of path -> text (tests; scratch projects use OPFS, projects.js)
//   DirWorkspace      - a folder the user picked with showDirectoryPicker() (File System Access
//                       API): reads and writes go to the real files on their disk. With
//                       { private: true } (a folder on disk) hidden and secret files do not exist
//                       for it: whatever the agent reads can reach every peer, and a write to
//                       .git/config or a hook runs commands on the host's machine
// Paths are relative, "/"-separated, and may not climb out of the root ("..").
//
// interface: list(dir) -> [{ name, dir: bool }], read(path) -> string, write(path, text),
//            readBytes(path) -> Uint8Array, writeBytes(path, u8), remove(path) (file or folder),
//            exists(path) -> bool, walk() -> [path] (every file, for search)
// watch(ws) adds onChange(fn) so the preview server hears about every write made through it.

const enc = new TextEncoder(), dec = new TextDecoder();

export function normPath(p) {
  const parts = [];
  for (const s of String(p || "").replace(/\\/g, "/").split("/")) {
    if (!s || s === ".") continue;
    if (s === "..") throw new Error(`path leaves the workspace: ${p}`);
    parts.push(s);
  }
  return parts.join("/");
}

// Off limits in a private workspace: any dot segment (.git, .env*, .npmrc, .ssh, .vscode, .github...)
// and key files by name.
const SECRET = /^(?:id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|.*\.(?:pem|key|p12|pfx|keystore|jks)|credentials(?:\.json)?|secrets?\.(?:json|ya?ml|toml))$/i;
export const secretPath = (p) => normPath(p).split("/").some((s) => s.startsWith(".") || SECRET.test(s));
// Files whose content can run commands on the host's machine once written (a build script, a
// hook, an npm script): an edit to one always asks, with a warning.
const RISKY_NAME = /^(?:package\.json|makefile|gnumakefile|dockerfile|docker-compose\.ya?ml|justfile|rakefile|gemfile|setup\.py|pyproject\.toml|cargo\.toml|build\.gradle(?:\.kts)?|pom\.xml|.*\.(?:sh|bash|zsh|fish|ps1|bat|cmd|command|mk))$/i;
export const riskyPath = (p) => { const parts = normPath(p).split("/"); return parts.some((s) => s.startsWith(".")) || RISKY_NAME.test(parts[parts.length - 1] || ""); };

// Values are text or Uint8Array (images for the preview); each read converts as needed.
export class MemoryWorkspace {
  constructor(files = {}) { this.files = new Map(Object.entries(files).map(([k, v]) => [normPath(k), v])); }
  _get(p) {
    const k = normPath(p);
    if (!this.files.has(k)) throw new Error(`no such file: ${k}`);
    return this.files.get(k);
  }
  async read(p) { const v = this._get(p); return typeof v === "string" ? v : dec.decode(v); }
  async readBytes(p) { const v = this._get(p); return typeof v === "string" ? enc.encode(v) : v; }
  async write(p, text) { this.files.set(normPath(p), String(text)); }
  async writeBytes(p, u8) { this.files.set(normPath(p), new Uint8Array(u8)); }
  async remove(p) {
    const k = normPath(p), pre = k + "/";
    let n = 0;
    for (const f of [...this.files.keys()]) if (f === k || f.startsWith(pre)) { this.files.delete(f); n++; }
    if (!n) throw new Error(`no such file: ${k}`);
  }
  async exists(p) { const k = normPath(p); return this.files.has(k) || [...this.files.keys()].some((f) => f.startsWith(k + "/")); }
  async list(dir = "") {
    const d = normPath(dir), pre = d ? d + "/" : "", seen = new Map();
    for (const f of this.files.keys()) {
      if (!f.startsWith(pre)) continue;
      const rest = f.slice(pre.length), i = rest.indexOf("/");
      if (i < 0) seen.set(rest, false); else seen.set(rest.slice(0, i), true);
    }
    if (d && !seen.size) throw new Error(`no such directory: ${d}`);
    return [...seen].map(([name, dir]) => ({ name, dir })).sort((a, b) => a.name.localeCompare(b.name));
  }
  async walk() { return [...this.files.keys()].sort(); }
}

// Folders nobody wants an agent to read through.
export const SKIP_DIRS = new Set([".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "target"]);

export class DirWorkspace {
  constructor(handle, { private: priv = false } = {}) { this.root = handle; this.private = priv; }   // a FileSystemDirectoryHandle
  _parts(p) {
    const n = normPath(p);
    if (this.private && n && secretPath(n)) throw new Error(`${n} is off limits: hidden and secret files of a folder on disk stay private`);
    return n ? n.split("/") : [];
  }
  check(p) { this._parts(p); }   // throws for a path this workspace refuses
  _hide(name) { return this.private && (name.startsWith(".") || SECRET.test(name)); }
  async _dir(parts, create = false) {
    let h = this.root;
    for (const s of parts) h = await h.getDirectoryHandle(s, { create });
    return h;
  }
  async _file(p) {
    const parts = this._parts(p);
    return (await (await this._dir(parts.slice(0, -1))).getFileHandle(parts[parts.length - 1])).getFile();
  }
  async read(p) { return (await this._file(p)).text(); }
  async readBytes(p) { return new Uint8Array(await (await this._file(p)).arrayBuffer()); }
  async write(p, text) {
    const parts = this._parts(p);
    const f = await (await this._dir(parts.slice(0, -1), true)).getFileHandle(parts[parts.length - 1], { create: true });
    const w = await f.createWritable(); await w.write(text); await w.close();
  }
  async writeBytes(p, u8) { return this.write(p, u8); }   // createWritable takes a BufferSource too
  async remove(p) {
    const parts = this._parts(p);
    await (await this._dir(parts.slice(0, -1))).removeEntry(parts[parts.length - 1], { recursive: true });
  }
  async exists(p) {
    let parts;
    try { parts = this._parts(p); } catch { return false; }
    try {
      const d = await this._dir(parts.slice(0, -1));
      try { await d.getFileHandle(parts[parts.length - 1]); } catch { await d.getDirectoryHandle(parts[parts.length - 1]); }
      return true;
    } catch { return false; }
  }
  async list(dir = "") {
    const d = await this._dir(this._parts(dir)), out = [];
    for await (const [name, h] of d.entries()) if (!this._hide(name)) out.push({ name, dir: h.kind === "directory" });
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }
  async walk(limit = 5000) {
    const out = [];
    const go = async (h, pre) => {
      for await (const [name, c] of h.entries()) {
        if (out.length >= limit) return;
        if (this._hide(name)) continue;
        if (c.kind === "directory") { if (!SKIP_DIRS.has(name)) await go(c, pre + name + "/"); }
        else out.push(pre + name);
      }
    };
    await go(this.root, "");
    return out.sort();
  }
}

// A view of ws whose writes and removes are reported to onChange listeners after they land
// ({ path, kind: "write" | "remove" }). Reads go straight through. Watching a watched workspace
// returns it as is, so the tools and the preview server can each call watch() safely.
export function watch(ws) {
  if (ws.onChange) return ws;
  const fns = new Set(), w = Object.create(ws);
  const emit = (path, kind) => { for (const f of [...fns]) { try { f({ path, kind }); } catch (e) { console.error(e); } } };
  w.write = async (p, t) => { await ws.write(p, t); emit(normPath(p), "write"); };
  w.writeBytes = async (p, u8) => { await ws.writeBytes(p, u8); emit(normPath(p), "write"); };
  w.remove = async (p) => { await ws.remove(p); emit(normPath(p), "remove"); };
  w.onChange = (fn) => { fns.add(fn); return () => fns.delete(fn); };
  return w;
}
