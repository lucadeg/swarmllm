// Converted-weights disk cache for the Deno/Node tests and benchmarks (never the browser).
//
// Loading a GGUF costs a disk read plus a CPU conversion per tensor (Q4_0/Q8_0 repack, K-quant ->
// Q8 requant, BF16/F16 -> f32). This caches, per tensor, the exact entry convertEntry() returns
// (the bytes the engine uploads to the GPU), so a warm load is one sequential read per tensor and
// no conversion. Attach it with attachWeightCache(G, path): engine/gguf.js ggufEntry consults
// G.entryCache before converting and stores what it converts.
//
// Keying: one directory per (GGUF realpath, size, mtime, LOADER_VERSION). LOADER_VERSION hashes
// engine/gguf.js (all the conversion code lives there) plus LOADER_EPOCH, so any change to the
// conversion starts a fresh directory; entries are never reused across conversion code versions.
// No load option changes the conversion today; if one ever does, pass it in opts.variant.
//
// Safety: entries are written to a temp file and renamed into place (a killed process leaves at
// worst a stray *.tmp-*, never a truncated entry under the real name). On read the header (magic,
// format, tensor type, element count, kind) and the exact file size are checked against the GGUF
// tensor info; any mismatch or I/O error falls back to fresh conversion and rewrites the entry.
// WEIGHT_CACHE_VERIFY=1 also checks a checksum of the payload on every read.
//
// Env: WEIGHT_CACHE unset -> ~/.cache/swarmllm-weights; WEIGHT_CACHE=0 (or off) -> disabled;
// anything else -> that directory. Without write permission the cache is read-only.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import process from "node:process";

export const LOADER_EPOCH = 1;   // bump by hand if conversion semantics change outside engine/gguf.js
const MAGIC = 0x43575753;        // "SWWC"
const FORMAT = 1;
const HDR = 64;
const KIND = { f32: 0, q4: 1, q8: 2 }, KIND_NAME = ["f32", "q4", "q8"];

const ggufSrc = fs.readFileSync(new URL("../engine/gguf.js", import.meta.url));
export const LOADER_VERSION = `e${LOADER_EPOCH}-` + crypto.createHash("sha256").update(ggufSrc).digest("hex").slice(0, 16);

// Deno: ask the permission state first so a missing --allow-env / --allow-write degrades the
// cache (disabled / read-only) instead of stopping the run at an interactive prompt
const granted = (desc) => { try { return !globalThis.Deno || Deno.permissions.querySync(desc).state === "granted"; } catch { return false; } };
const env = (k) => { if (!granted({ name: "env", variable: k })) return undefined; try { return process.env[k]; } catch { return undefined; } };

export function weightCacheRoot() {
  const v = env("WEIGHT_CACHE");
  if (v === "0" || v === "off" || v === "false") return null;
  if (v) return path.resolve(v);
  const home = env("HOME");
  if (!home) return null;
  return path.join(home, ".cache", "swarmllm-weights");
}

// 32-bit multiply-xor hash over the payload's u32 words (tail bytes folded in). Not
// cryptographic; catches bit rot and partial overwrites when WEIGHT_CACHE_VERIFY=1. Takes the
// payload as consecutive parts; every part but the last must be a multiple of 4 bytes.
export function payloadHash(...parts) {
  let h = 0x811C9DC5, total = 0;
  for (const u8 of parts) {
    const n4 = u8.byteLength >>> 2;
    const w = (u8.byteOffset & 3) === 0 ? new Uint32Array(u8.buffer, u8.byteOffset, n4) : new Uint32Array(u8.slice(0, n4 * 4).buffer);
    for (let i = 0; i < n4; i++) h = Math.imul(h ^ w[i], 0x01000193);
    for (let i = n4 * 4; i < u8.byteLength; i++) h = Math.imul(h ^ u8[i], 0x01000193);
    total += u8.byteLength;
  }
  return (h ^ total) >>> 0;
}

// Byte layout of an entry for this tensor/kind: [qs | pad to 4 | scales] or [data]
function layout(info, kind) {
  const n = info.nElems, nb = Math.ceil(n / 32);
  if (kind === "f32") return { a: n * 4, pad: 0, b: 0 };
  const a = kind === "q4" ? n / 2 : nb * 32, pad = (4 - (a & 3)) & 3;
  return { a, pad, b: Math.ceil(nb / 2) * 4 };
}

const u8of = (v) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);

function readFull(fd, u8, pos) {
  let o = 0;
  while (o < u8.byteLength) {
    const n = fs.readSync(fd, u8, o, Math.min(u8.byteLength - o, 1 << 30), pos + o);
    if (n <= 0) throw new Error("short read");
    o += n;
  }
}

class WeightCache {
  constructor(dir, meta) {
    this.dir = dir; this.meta = meta;
    this.verify = env("WEIGHT_CACHE_VERIFY") === "1";
    this.readOnly = false;
    this.stats = { hit: 0, miss: 0, bad: 0, write: 0, hitBytes: 0, writeBytes: 0, readMs: 0, writeMs: 0 };
  }
  file(name) { return path.join(this.dir, name.replace(/[^A-Za-z0-9._-]/g, "_") + ".bin"); }

  // Validate an entry's header and exact size against the tensor info, without reading the
  // payload: { fd, kind, L, hash } (caller closes fd) or null. Bad entries are counted and warned.
  _open(info) {
    const f = this.file(info.name);
    let fd;
    try { fd = fs.openSync(f, "r"); } catch { this.stats.miss++; return null; }
    try {
      const size = fs.fstatSync(fd).size;
      if (size < HDR) throw new Error("short header");
      const h = new Uint8Array(HDR); readFull(fd, h, 0);
      const dv = new DataView(h.buffer);
      if (dv.getUint32(0, true) !== MAGIC || dv.getUint32(4, true) !== FORMAT) throw new Error("bad magic");
      const kind = KIND_NAME[dv.getUint32(8, true)];
      if (!kind || dv.getUint32(12, true) !== info.ggmlType || dv.getFloat64(16, true) !== info.nElems) throw new Error("wrong tensor");
      const L = layout(info, kind);
      if (dv.getFloat64(24, true) !== L.a || dv.getFloat64(32, true) !== L.b) throw new Error("wrong layout");
      if (size !== HDR + L.a + L.pad + L.b) throw new Error(`size ${size} != ${HDR + L.a + L.pad + L.b}`);
      return { fd, kind, L, hash: dv.getUint32(40, true), size };
    } catch (err) {
      fs.closeSync(fd);
      this._bad(f, err);
      return null;
    }
  }
  _bad(f, err) {
    this.stats.bad++;
    if (this.stats.bad <= 3) console.warn(`weight cache: ignoring ${path.basename(f)} (${err.message}), converting fresh`);
  }

  // Entry for this tensor from disk, or null (missing, corrupt, wrong size): caller converts fresh.
  get(info) {
    const t0 = performance.now();
    const o = this._open(info);
    if (!o) return null;
    const { fd, kind, L } = o;
    try {
      const u8 = new Uint8Array(L.a + L.pad + L.b); readFull(fd, u8, HDR);
      if (this.verify && payloadHash(u8) !== o.hash) throw new Error("checksum");
      this.stats.hit++; this.stats.hitBytes += o.size; this.stats.readMs += performance.now() - t0;
      if (kind === "f32") return { kind, data: new Float32Array(u8.buffer, 0, info.nElems) };
      return { kind, qs: new Uint8Array(u8.buffer, 0, L.a), scales: new Uint32Array(u8.buffer, L.a + L.pad, L.b / 4), shape: info.shape };
    } catch (err) {
      this._bad(this.file(info.name), err);
      return null;
    } finally { fs.closeSync(fd); }
  }

  // For serving an entry without reading it into memory (tests/bench/serve.mjs): a validated
  // { file, offset, kind, a, pad, b } for the payload bytes, or null. Header and size are checked;
  // the payload checksum only with verify on (then the payload is read once here).
  locate(info) {
    const o = this._open(info);
    if (!o) return null;
    try {
      if (this.verify) { const u8 = new Uint8Array(o.L.a + o.L.pad + o.L.b); readFull(o.fd, u8, HDR); if (payloadHash(u8) !== o.hash) { this._bad(this.file(info.name), new Error("checksum")); return null; } }
      this.stats.hit++; this.stats.hitBytes += o.size;
      return { file: this.file(info.name), offset: HDR, kind: o.kind, a: o.L.a, pad: o.L.pad, b: o.L.b };
    } finally { fs.closeSync(o.fd); }
  }

  // Store a freshly converted entry (atomic: temp file + rename).
  put(info, e) {
    if (this.readOnly || !e || e.gpu) return;
    const kind = e.kind;
    if (!(kind in KIND)) return;
    const parts = kind === "f32" ? [u8of(e.data)] : [u8of(e.qs), u8of(e.scales)];
    const L = layout(info, kind);
    if (parts[0].byteLength !== L.a || (parts[1]?.byteLength ?? 0) !== L.b) return;   // not a layout we know: don't cache
    const pad = new Uint8Array(L.pad);
    const body = kind === "f32" ? parts : [parts[0], pad, parts[1]];
    const h = payloadHash(...body);   // a + pad is a multiple of 4, so this equals the hash of the file body
    const hdr = new Uint8Array(HDR), dv = new DataView(hdr.buffer);
    dv.setUint32(0, MAGIC, true); dv.setUint32(4, FORMAT, true); dv.setUint32(8, KIND[kind], true);
    dv.setUint32(12, info.ggmlType, true); dv.setFloat64(16, info.nElems, true);
    dv.setFloat64(24, L.a, true); dv.setFloat64(32, L.b, true); dv.setUint32(40, h, true);
    const f = this.file(info.name), tmp = `${f}.tmp-${process.pid}-${Math.random().toString(36).slice(2)}`;
    const t0 = performance.now();
    let fd = null;
    try {
      fd = fs.openSync(tmp, "w");
      for (const b of [hdr, ...body]) { let o = 0; while (o < b.byteLength) o += fs.writeSync(fd, b, o, Math.min(b.byteLength - o, 1 << 30)); }
      fs.closeSync(fd); fd = null;
      fs.renameSync(tmp, f);
      this.stats.write++; this.stats.writeBytes += HDR + L.a + L.pad + L.b; this.stats.writeMs += performance.now() - t0;
    } catch (err) {
      if (fd !== null) try { fs.closeSync(fd); } catch { /* already closed */ }
      try { fs.unlinkSync(tmp); } catch { /* never created */ }
      this.readOnly = true;
      console.warn(`weight cache: cannot write (${err.message}); cache is read-only for this run`);
    }
  }
  flush() {}
  summary() {
    const s = this.stats, gb = (b) => (b / 2 ** 30).toFixed(2);
    return `weight cache ${this.dir}: ${s.hit} hits (${gb(s.hitBytes)} GB, ${(s.readMs / 1000).toFixed(1)} s), ${s.miss} misses, ${s.bad} bad, ${s.write} written (${gb(s.writeBytes)} GB, ${(s.writeMs / 1000).toFixed(1)} s)`;
  }
}

// Open (creating if needed) the cache directory for one GGUF file. null when disabled.
// opts.root overrides the env-derived root; opts.variant keys any future conversion option.
export function openWeightCache(ggufPath, opts = {}) {
  const root = opts.root !== undefined ? opts.root : weightCacheRoot();
  if (!root) return null;
  let real, st;
  try { real = fs.realpathSync(ggufPath); st = fs.statSync(real); } catch { return null; }
  const meta = { gguf: real, size: st.size, mtimeMs: Math.floor(st.mtimeMs),   // Deno reports whole ms, Node fractional
    loader: LOADER_VERSION, variant: opts.variant || "" };
  const key = crypto.createHash("sha256").update(JSON.stringify(meta)).digest("hex").slice(0, 16);
  const dir = path.join(root, `${path.basename(real, ".gguf")}-${key}`);
  const c = new WeightCache(dir, meta);
  if (!granted({ name: "read", path: root })) return null;
  if (!granted({ name: "write", path: root })) {
    if (!fs.existsSync(dir)) return null;
    c.readOnly = true;
    if (!opts.quiet) console.warn("weight cache: no write permission (--allow-write), read-only");
    return c;
  }
  try {
    fs.mkdirSync(dir, { recursive: true });
    const mf = path.join(dir, "meta.json");
    if (!fs.existsSync(mf)) { const tmp = `${mf}.tmp-${process.pid}`; fs.writeFileSync(tmp, JSON.stringify(meta, null, 1)); fs.renameSync(tmp, mf); }
  } catch (err) {
    if (!fs.existsSync(dir)) { if (!opts.quiet) console.warn(`weight cache: disabled (${err.message})`); return null; }
    c.readOnly = true;
  }
  if (!opts.quiet) pruneStale(root, meta, dir);
  return c;
}

// Other directories for the same GGUF path (older loader versions or an older copy of the file)
// are dead weight: report them, and delete them when WEIGHT_CACHE_PRUNE=1.
function pruneStale(root, meta, keep) {
  let names = [];
  try { names = fs.readdirSync(root); } catch { return; }
  for (const n of names) {
    const d = path.join(root, n);
    if (d === keep) continue;
    try {
      const m = JSON.parse(fs.readFileSync(path.join(d, "meta.json"), "utf8"));
      if (m.gguf !== meta.gguf) continue;
      if (env("WEIGHT_CACHE_PRUNE") === "1") { fs.rmSync(d, { recursive: true, force: true }); console.warn(`weight cache: pruned stale ${n}`); }
      else console.warn(`weight cache: stale entry ${d} (loader ${m.loader}); WEIGHT_CACHE_PRUNE=1 deletes it`);
    } catch { /* not ours */ }
  }
}

// Attach to a parsed header: ggufEntry(G, ...) then reads/writes through the cache.
export function attachWeightCache(G, ggufPath, opts = {}) {
  const c = openWeightCache(ggufPath, opts);
  if (c) G.entryCache = c;
  return c;
}
