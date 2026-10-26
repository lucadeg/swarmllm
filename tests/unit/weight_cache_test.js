// Converted-weights cache (tests/weight_cache.js) is bit-identical to fresh conversion, and a
// bad entry (short, corrupt payload, garbage header) falls back to fresh conversion.
// Uses real tensors of every ggml type in the 27B and MoE files; skipped when the models are
// absent or the run has no write permission (the cache lives in a temp dir):
//   deno test --allow-read --allow-write tests/unit/weight_cache_test.js
// Also checks the u16 fast paths of q4Repack / q8Repack / streamEntryToGPU against the
// original per-block copy, including odd-offset inputs and ragged network chunks.
import fs from "node:fs";
import { parseGGUFHeader, ggufEntry, convertEntry, streamEntryToGPU, GGML_Q4_0, GGML_Q8_0, QK8_0, Q8_0_BLOCK_BYTES } from "../../engine/gguf.js";
import { openWeightCache } from "../weight_cache.js";

const root = new URL("../../", import.meta.url).pathname;
const MODELS = [root + "models/q38/model.gguf", root + "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"];
const TYPES = { 0: "F32", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 8: "Q8_0", 13: "Q5_K", 14: "Q6_K", 30: "BF16" };
const have = MODELS.filter((p) => { try { return fs.statSync(p).isFile(); } catch { return false; } });
const canWrite = Deno.permissions.querySync({ name: "write" }).state === "granted";
const ignore = !have.length || !canWrite;
const MAX = 96 * 2 ** 20;   // keep each case small: smallest tensor of each type under this size

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const bytesEq = (a, b) => {
  if (!a || !b) return a === b;
  const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
};
const sameEntry = (a, b) => a.kind === b.kind && (a.kind === "f32" ? bytesEq(a.data, b.data) : bytesEq(a.qs, b.qs) && bytesEq(a.scales, b.scales) && String(a.shape) === String(b.shape));

function open(path) {
  const fd = fs.openSync(path, "r");
  const readAt = (off, len) => { const out = new Uint8Array(len); let o = 0; while (o < len) { const n = fs.readSync(fd, out, o, len - o, off + o); if (n <= 0) break; o += n; } return out; };
  const G = parseGGUFHeader(readAt(0, 64 << 20).buffer, { skipTokenizer: true });
  return { G, readAt, close: () => fs.closeSync(fd) };
}
// the cases: per model, smallest 2D tensor of each type (and 1D F32/BF16), plus one stacked 3D expert tensor
function pickCases(G) {
  const best = {};
  for (const t of Object.values(G.tensors)) {
    if (!(t.ggmlType in TYPES) || t.byteLength > MAX * (t.shape.length === 3 ? 2 : 1)) continue;
    const k = TYPES[t.ggmlType] + "/" + t.shape.length + "D";
    if (!best[k] || t.byteLength < best[k].byteLength) best[k] = t;
  }
  return Object.entries(best).sort();
}
const as2D = (info) => info.shape.length === 3 ? { ...info, shape: [info.shape[0] * info.shape[1], info.shape[2]] } : info;

Deno.test({ name: "weight cache: cached entry is byte-identical to fresh conversion, every ggml type", ignore, async fn() {
  const tmp = await Deno.makeTempDir({ prefix: "wcache-test-" });
  try {
    const seen = new Set();
    for (const path of have) {
      const { G, readAt, close } = open(path);
      const cache = openWeightCache(path, { root: tmp, quiet: true });
      ok(cache, "cache opened");
      const Gc = { ...G, entryCache: cache };
      for (const [k, info] of pickCases(G)) {
        seen.add(k.split("/")[0]);
        const info2 = as2D(info);
        const fresh = convertEntry(info2, readAt(info2.byteOffset, info2.byteLength));
        let reads = 0;
        const bytesOf = (i) => { reads++; return readAt(i.byteOffset, i.byteLength); };
        const e1 = await ggufEntry(Gc, bytesOf, info.name);          // miss: converts, writes
        ok(reads === 1 && sameEntry(e1, fresh), `${k} ${info.name}: first load differs from fresh`);
        const e2 = await ggufEntry(Gc, bytesOf, info.name);          // hit: no GGUF read
        ok(reads === 1, `${k} ${info.name}: cache hit still read the GGUF`);
        ok(sameEntry(e2, fresh), `${k} ${info.name}: cached entry differs from fresh conversion`);
        if (e2.kind !== "f32") ok(e2.scales instanceof Uint32Array && e2.qs instanceof Uint8Array && e2.scales.byteOffset % 4 === 0, "entry views have the engine's types");
        else ok(e2.data instanceof Float32Array, "f32 entry is a Float32Array");
        console.log(`  ${path.split("/").pop().slice(0, 18).padEnd(18)} ${k.padEnd(8)} ${info.name.padEnd(34)} ${(info.byteLength / 2 ** 20).toFixed(1).padStart(6)} MB -> ${e2.kind} identical`);
      }
      ok(cache.stats.bad === 0 && cache.stats.hit === cache.stats.write, JSON.stringify(cache.stats));
      close();
    }
    console.log("  types covered:", [...seen].sort().join(", "));
    for (const t of ["Q4_0", "Q8_0", "F32"]) ok(seen.has(t), "covered " + t);
  } finally { await Deno.remove(tmp, { recursive: true }); }
} });

Deno.test({ name: "weight cache: short, corrupt or garbage entries fall back to fresh conversion", ignore, async fn() {
  const tmp = await Deno.makeTempDir({ prefix: "wcache-test-" });
  try {
    const { G, readAt, close } = open(have[0]);
    const cases = pickCases(G).filter(([k]) => /^(Q4_0|F32|Q5_K|Q8_0)\/[12]D$/.test(k)).map(([, i]) => i);
    for (const info of cases) {
      const fresh = convertEntry(info, readAt(info.byteOffset, info.byteLength));
      const cache = openWeightCache(have[0], { root: tmp, quiet: true });
      cache.verify = true;   // as with WEIGHT_CACHE_VERIFY=1
      const Gc = { ...G, entryCache: cache };
      let reads = 0;
      const bytesOf = (i) => { reads++; return readAt(i.byteOffset, i.byteLength); };
      await ggufEntry(Gc, bytesOf, info.name);
      const f = cache.file(info.name), good = fs.readFileSync(f);
      const damage = {
        short: () => fs.writeFileSync(f, good.subarray(0, good.length - 1)),
        long: () => fs.writeFileSync(f, Buffer.concat([good, Buffer.alloc(4)])),
        empty: () => fs.writeFileSync(f, new Uint8Array(0)),
        header: () => { const b = Buffer.from(good); b.fill(0x5a, 0, 16); fs.writeFileSync(f, b); },
        kind: () => { const b = Buffer.from(good); b[8] ^= 3; fs.writeFileSync(f, b); },
        bitflip: () => { const b = Buffer.from(good); b[b.length - 3] ^= 0x10; fs.writeFileSync(f, b); },
      };
      for (const [what, hurt] of Object.entries(damage)) {
        hurt();
        const before = reads;
        const origWarn = console.warn; console.warn = () => {};
        let e;
        try { e = await ggufEntry(Gc, bytesOf, info.name); } finally { console.warn = origWarn; }
        ok(reads === before + 1, `${info.name} ${what}: bad entry was used instead of converting fresh`);
        ok(sameEntry(e, fresh), `${info.name} ${what}: fallback differs from fresh`);
        ok(bytesEq(fs.readFileSync(f), good), `${info.name} ${what}: entry was not rewritten`);
      }
      // a stray temp file (a process killed mid-write) is never read as an entry
      fs.writeFileSync(f + ".tmp-1-x", good.subarray(0, 10));
      const before = reads;
      ok(sameEntry(await ggufEntry(Gc, bytesOf, info.name), fresh) && reads === before, "hit ignores stray tmp");
      console.log(`  ${info.name}: short/long/empty/header/kind/bitflip all fell back and were rewritten`);
    }
    // a different loader variant / GGUF identity never shares a directory
    const a = openWeightCache(have[0], { root: tmp, quiet: true }), b = openWeightCache(have[0], { root: tmp, quiet: true, variant: "x" });
    ok(a.dir !== b.dir, "variant changes the cache key");
    close();
  } finally { await Deno.remove(tmp, { recursive: true }); }
} });

// ---- fast repack paths vs the original per-block copy ----
function refRepack(info, bytes, q4) {
  const BLK = q4 ? 18 : Q8_0_BLOCK_BYTES, QSB = q4 ? 16 : QK8_0, nb = info.nElems / 32;
  const qs = new Uint8Array(nb * QSB), scales = new Uint32Array(Math.ceil(nb / 2)), sc16 = new Uint16Array(scales.buffer);
  for (let b = 0; b < nb; b++) { const base = b * BLK; sc16[b] = bytes[base] | (bytes[base + 1] << 8); qs.set(bytes.subarray(base + 2, base + BLK), b * QSB); }
  return { qs, scales };
}
function synth(type, rows, cols, seed = 1) {
  const n = rows * cols, BLK = type === GGML_Q4_0 ? 18 : Q8_0_BLOCK_BYTES, b = new Uint8Array((n / 32) * BLK);
  let x = seed; for (let i = 0; i < b.length; i++) { x = Math.imul(x ^ (x >>> 13), 0x5bd1e995) + i; b[i] = x >>> 24; }
  return { info: { name: "synth", ggmlType: type, shape: [rows, cols], nElems: n, byteLength: b.length }, bytes: b };
}

Deno.test("repack fast paths: q4Repack / q8Repack identical to per-block copy, aligned and odd offsets", () => {
  for (const type of [GGML_Q4_0, GGML_Q8_0]) {
    for (const [r, c] of [[1, 32], [3, 64], [7, 96], [64, 512]]) {   // odd block counts too
      const { info, bytes } = synth(type, r, c, r * c);
      const ref = refRepack(info, bytes, type === GGML_Q4_0);
      const odd = new Uint8Array(bytes.length + 1).subarray(1); odd.set(bytes);
      for (const src of [bytes, odd]) {
        const e = convertEntry(info, src);
        ok(bytesEq(e.qs, ref.qs) && bytesEq(e.scales, ref.scales), `type ${type} ${r}x${c} offset ${src.byteOffset}`);
      }
    }
  }
});

Deno.test("streamEntryToGPU: uploaded bytes identical to the repack, ragged and odd-offset chunks", async () => {
  globalThis.GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 0x8, COPY_SRC: 0x4 };
  const device = {
    pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer: ({ size }) => ({ mem: new Uint8Array(size) }),
    queue: { writeBuffer(buf, off, src, srcOff = 0, size) { const s = ArrayBuffer.isView(src) ? new Uint8Array(src.buffer, src.byteOffset, src.byteLength) : new Uint8Array(src); buf.mem.set(s.subarray(srcOff, srcOff + (size ?? s.length - srcOff)), off); } },
  };
  for (const type of [GGML_Q4_0, GGML_Q8_0]) {
    const { info, bytes } = synth(type, 96, 256, 7 + type);
    const ref = refRepack(info, bytes, type === GGML_Q4_0);
    for (const odd of [false, true]) {
      const openRange = async () => {
        let off = 0, i = 0;
        return { ok: true, status: 200, body: { getReader: () => ({ read: async () => {
          if (off >= bytes.length) return { done: true };
          const n = Math.min(bytes.length - off, [1, 17, 4099, 999, 18 * 50, 34 * 3 + 1][i++ % 6]);
          let v = bytes.slice(off, off + n); off += n;
          if (odd) { const w = new Uint8Array(n + 1).subarray(1); w.set(v); v = w; }
          return { done: false, value: v };
        } }) } };
      };
      const e = await streamEntryToGPU(device, info, openRange, { staging: 3000 });
      ok(bytesEq(e.gpu.qs.mem.subarray(0, ref.qs.length), ref.qs), `stream qs type ${type} odd ${odd}`);
      ok(bytesEq(e.gpu.sc.mem.subarray(0, ref.scales.byteLength), new Uint8Array(ref.scales.buffer)), `stream scales type ${type} odd ${odd}`);
    }
  }
});
