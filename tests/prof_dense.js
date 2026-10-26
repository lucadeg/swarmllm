// Per-kernel GPU time for the dense engine (Qwen3 1.7B by default), from timestamp queries: every
// dispatch runs in its own compute pass with begin/end timestamps. Profiles one decode token and one
// batched prefill pass at context position FILL.
//   cd tests && [DENSE=qwen17] [FILL=4096] [OPTS='{...}'] deno run --unstable-webgpu --allow-read --allow-env prof_dense.js
import { DenseEngine, makeTokenizer } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import fs from "node:fs";

const env = (k, d) => Deno.env.get(k) ?? d;
const M = env("DENSE", "qwen17"), FILL = +env("FILL", 4096), OPTS = JSON.parse(env("OPTS", "{}"));
const dir = new URL(`../models/${M}/`, import.meta.url).pathname;
const fd = fs.openSync(dir + "model.gguf", "r");
const readAt = (off, len) => { const o = new Uint8Array(len); let g = 0; while (g < len) { const n = fs.readSync(fd, o, g, len - g, off + g); if (n <= 0) break; g += n; } return o; };
const G = parseGGUFHeader(readAt(0, 32 << 20).buffer, { skipTokenizer: true });
const cfg = JSON.parse(await Deno.readTextFile(dir + "config.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "tokenizer.json")));
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
const L = cfg.num_hidden_layers;
const weights = await ggufWeights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
const eng = await DenseEngine.create({ device, cfg, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: Math.max(8192, FILL + 64), ...OPTS });
let ids = tok.encode(await Deno.readTextFile(new URL("../engine/qwen35.js", import.meta.url)));
while (ids.length < FILL + 16) ids = ids.concat(ids);
await eng.prefillTokens(ids.slice(0, FILL));
let t0 = performance.now(); for (let i = 0; i < 10; i++) await eng.forwardToken(ids[FILL + i]); const wall = (performance.now() - t0) / 10;
const base = eng.pos;
eng.encodeAhead = false; eng._fwdPre = null;   // the hook must see every recorded dispatch

const MAXQ = 4096, qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
const res = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const rd = device.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
let names = [], nq = 0, on = false;
const pname = new Map(Object.entries(eng.pipes).map(([k, v]) => [v, k]));
const origCreate = device.createCommandEncoder.bind(device);
device.createCommandEncoder = (d) => { const enc = origCreate(d); if (!on) return enc; const ob = enc.beginComputePass.bind(enc); const q0 = nq;
  enc.beginComputePass = () => { let pipe = null, name = "?"; const bgs = {};
    return { setPipeline(p) { pipe = p; name = pname.get(p) || "?"; }, setBindGroup(i, b) { bgs[i] = b; },
      dispatchWorkgroups(x, y = 1, z = 1) { const p = ob({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: nq, endOfPassWriteIndex: nq + 1 } }); nq += 2; names.push(name);
        p.setPipeline(pipe); for (const i in bgs) p.setBindGroup(+i, bgs[i]); p.dispatchWorkgroups(x, y, z); p.end(); }, end() {} }; };
  const ofin = enc.finish.bind(enc); enc.finish = () => { if (nq > q0) { enc.resolveQuerySet(qs, q0, nq - q0, res, q0 * 8); enc.copyBufferToBuffer(res, q0 * 8, rd, q0 * 8, (nq - q0) * 8); } return ofin(); };
  return enc; };
const collect = async (fn, runs) => {
  const agg = {};
  for (let r = 0; r < runs; r++) {
    names = []; nq = 0; on = true; await fn(); on = false;
    await device.queue.onSubmittedWorkDone();
    await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0, nq * 8)); rd.unmap();
    names.forEach((n, i) => { const ns = Number(t[2 * i + 1] - t[2 * i]); (agg[n] ||= [0, 0])[0] += ns / 1e6 / runs; agg[n][1] += 1 / runs; });
  }
  return agg;
};
const show = (title, agg) => {
  const gpu = Object.values(agg).reduce((a, b) => a + b[0], 0);
  console.log(`${title}: sum of kernel GPU time ${gpu.toFixed(2)} ms · dispatches ${Math.round(Object.values(agg).reduce((a, b) => a + b[1], 0))}`);
  for (const [k, [ms, n]] of Object.entries(agg).sort((a, b) => b[1][0] - a[1][0])) console.log(`  ${k.padEnd(24)} ${ms.toFixed(3).padStart(8)} ms  ${String(Math.round(n)).padStart(5)}×  ${(ms / n * 1000).toFixed(1).padStart(8)} µs each`);
};
console.log(`${M} at pos ${base}: decode wall ${wall.toFixed(2)} ms/token (${(1000 / wall).toFixed(1)} tok/s), opts ${JSON.stringify(OPTS)}`);
// decode tokens: each forwardToken does two submits (token, readback); only the first has dispatches
show("decode token", await collect(async () => { eng.pos = base; await eng.forwardToken(ids[FILL]); }, 3));
// one batched prefill pass (prefillTokens with exactly one chunk of the engine's batch width)
const W = eng.NC || 4;
show(`prefill pass (${W} cols)`, await collect(async () => { eng.pos = base; await eng.prefillTokens(ids.slice(FILL, FILL + W)); }, 3));
