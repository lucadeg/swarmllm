// fuseProj A/B (GPU): merged projection GEMVs ([qkv|z], [beta|alpha], [k|v], MoE [router|shared
// gate]) must give the same bits as the per-tensor GEMVs over the same weights, for decode and for
// batched passes at every width (twins, and the full-width GEMM prefill that keeps the separate ops).
// Also prints dispatches per decode token with and without the merge.
//   deno run --unstable-webgpu --allow-read --allow-env test_fuse_proj.js            (27B, LAYERS=8)
//   MODEL=moe LAYERS=8 deno run --unstable-webgpu --allow-read --allow-env test_fuse_proj.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights } from "../engine/gguf.js";
const moe = Deno.env.get("MODEL") === "moe";
const PATH = moe ? "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" : "../models/q38/model.gguf";
const fh = await Deno.open(PATH);
const readAt = async (off, len) => { await fh.seek(off, 0); const o = new Uint8Array(len); let g = 0; while (g < len) { const n = await fh.read(o.subarray(g)); if (n === null) break; g += n; } return o; };
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.pushErrorScope("validation");
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
const L = +(Deno.env.get("LAYERS") || 8);
const NC = +(Deno.env.get("BCOLS") || 16);
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const mk = (w, fuseProj) => Qwen35Engine.create({ device, meta: G.meta, weights: w, vocab: 248320, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 128, batchCols: NC, coopRowsB: NC === 16 ? 1 : 4, fuseProj });
const eng = await mk(weights, true);
// reference with the old separate buffers (fuseProj: false at create): catches layout bugs that the
// runtime switch cannot (it runs over the merged buffers and their strides)
const ref = await mk(await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }), false);
console.log("fuseStats", JSON.stringify(eng.fuseStats));
let fail = eng.fuseStats.merged === 0 || eng.fuseStats.decodeOps === 0;
if (fail) console.log("nothing was merged");
const ids = [760, 6511, 315, 9109, 3139, 1234, 42, 7, 999, 31337, 2048, 4096, 11, 12, 13, 14, 15, 16, 17, 18];
const bits = (a) => new Uint32Array(Float32Array.from(a).buffer);
const same = (a, b) => { const x = bits(a), y = bits(b); if (x.length !== y.length) return false; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
const run = async (fuse, mode, e = eng) => {
  e.fuseProj = fuse; e.reset(); e.pos = 0;
  if (mode === "seq") { let last; for (const t of ids.slice(0, 12)) last = await e.forwardToken(t); return last; }
  // mode = batch width list: consecutive batched passes of these widths (NC = the GEMM prefill)
  let pos = 0, out = [];
  for (const w of mode) { const hs = await e.embedRunBatch(ids.slice(pos, pos + w), pos); pos += w; out = hs; }
  return out;
};
const count = async (fuse) => {
  let n = 0; const o3 = eng._d3, oxyz = eng._dxyz, od = eng._d;
  eng._d3 = function (...a) { n++; return o3.apply(this, a); }; eng._dxyz = function (...a) { n++; return oxyz.apply(this, a); }; eng._d = function (...a) { n++; return od.apply(this, a); };
  eng.fuseProj = fuse; eng.reset(); eng.pos = 0; await eng.forwardToken(ids[0]); n = 0; await eng.forwardToken(ids[1]);
  eng._d3 = o3; eng._dxyz = oxyz; eng._d = od;
  return n;
};
for (const mode of ["seq", [NC, 4], [3, 1, 8], [2, 5]]) {
  const a = await run(true, mode), b = await run(false, mode), c = await run(false, mode, ref);
  const ok = same(a, b) && same(a, c);
  console.log(`${JSON.stringify(mode).padEnd(10)} merged == separate: ${same(a, b) ? "yes" : "NO"}  == unfused engine: ${same(a, c) ? "yes" : "NO"}`);
  if (!ok) fail = true;
}
console.log(`dispatches per decode token (${L} layers + head): merged ${await count(true)}, separate ${await count(false)}`);
const err = await device.popErrorScope();
if (err) { console.log("validation error:", err.message); fail = true; }
console.log(fail ? "FUSE PROJ FAIL" : "FUSE PROJ PASS ✓");
if (fail) Deno.exit(1);
