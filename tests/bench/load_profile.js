// CPU-only load profile (no GPU device): where does model load time go?
//   deno run --allow-read --allow-env --allow-write tests/bench/load_profile.js [model.gguf] [--cache]
// Times: header parse, raw disk read of every tensor the 27B/MoE loader touches,
// and the per-tensor CPU conversion (repack / requant / dequant) by ggml type.
// The GPU upload is stubbed: converted CPU copies are dropped as soon as they exist,
// the same peak-memory shape as the real loader handing them to the engine.
// --cache routes the conversion through the weight cache (tests/weight_cache.js),
// so running it twice shows the warm-cache load cost.
import { parseGGUFHeader, qwen35Weights } from "../../engine/gguf.js";
import { attachWeightCache } from "../weight_cache.js";

const root = new URL("../..", import.meta.url).pathname;
const args = Deno.args.filter((a) => !a.startsWith("--"));
const useCache = Deno.args.includes("--cache");
const PATH = args[0] || root + "models/q38/model.gguf";
const TYPES = { 0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 8: "Q8_0", 13: "Q5_K", 14: "Q6_K", 30: "BF16" };

const fh = await Deno.open(PATH);
const readAt = async (off, len) => {
  await fh.seek(off, Deno.SeekMode.Start);
  const out = new Uint8Array(len); let got = 0;
  while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; }
  return out;
};
let t = performance.now();
const head = await readAt(0, 64 << 20);
const tHeadRead = performance.now() - t;
t = performance.now();
const G = parseGGUFHeader(head.buffer);
const tParse = performance.now() - t;
t = performance.now();
parseGGUFHeader(head.buffer, { skipTokenizer: true });
const tParseSkip = performance.now() - t;
const arch = G.meta["general.architecture"], nBlk = G.meta[arch + ".block_count"], nextn = G.meta[arch + ".nextn_predict_layers"] || 0;
const L = nBlk - (nextn || (G.tensors[`blk.${nBlk - 1}.nextn.eh_proj.weight`] ? 1 : 0));
console.log(`${PATH.split("/").pop()}: ${arch}, ${Object.keys(G.tensors).length} tensors, trunk layers ${L}`);
console.log(`header: read 64 MB ${tHeadRead.toFixed(0)} ms, parse ${tParse.toFixed(0)} ms (skipTokenizer ${tParseSkip.toFixed(0)} ms)`);

const hist = {};
for (const ti of Object.values(G.tensors)) { const k = TYPES[ti.ggmlType] || ti.ggmlType; (hist[k] ||= { n: 0, bytes: 0 }).n++; hist[k].bytes += ti.byteLength; }
console.log("tensor types:", Object.entries(hist).map(([k, v]) => `${k} x${v.n} ${(v.bytes / 2 ** 30).toFixed(2)} GB`).join(", "));

// per-type accounting through the real loader
const byType = {};
let readMs = 0, readBytes = 0, lastEnd = 0, pendingRead = 0;
const bytesOf = async (info) => {
  const t0 = performance.now(); const b = await readAt(info.byteOffset, info.byteLength);
  const dt = performance.now() - t0; readMs += dt; pendingRead += dt; readBytes += b.length; return b;
};
const cache = useCache ? attachWeightCache(G, PATH) : null;
if (useCache) console.log("weight cache:", cache ? cache.dir : "disabled");
const onEntry = (e, name) => {
  const now = performance.now(), info = G.tensors[name];
  const k = (TYPES[info.ggmlType] || info.ggmlType) + "->" + e.kind;
  const s = (byType[k] ||= { n: 0, ms: 0, read: 0, bytes: 0 });
  s.n++; s.ms += now - lastEnd - pendingRead; s.read += pendingRead; s.bytes += info.byteLength;
  pendingRead = 0;
  e.qs = e.scales = e.data = null;   // stub upload: drop the CPU copy
  lastEnd = performance.now();
};
t = performance.now(); lastEnd = t;
await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }, () => {}, onEntry);
const tAll = performance.now() - t;
console.log(`loader total ${(tAll / 1000).toFixed(1)} s: read ${(readMs / 1000).toFixed(1)} s (${(readBytes / 2 ** 30).toFixed(2)} GB, ${(readBytes / 2 ** 30 / (readMs / 1000)).toFixed(2)} GB/s), conversion+other ${((tAll - readMs) / 1000).toFixed(1)} s`);
for (const [k, s] of Object.entries(byType).sort((a, b) => b[1].ms - a[1].ms))
  console.log(`  ${k.padEnd(12)} x${String(s.n).padEnd(4)} ${(s.bytes / 2 ** 30).toFixed(2).padStart(6)} GB  convert ${(s.ms / 1000).toFixed(2).padStart(6)} s  read ${(s.read / 1000).toFixed(2).padStart(6)} s`);
if (cache) console.log(cache.summary());
