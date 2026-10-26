// Dense-engine (Qwen3 / Llama) benchmark: prefill speed as the context fills and plain decode speed at
// each fill, plus the greedy tokens (so two engine variants can be compared token for token).
//   cd tests && [DENSE=qwen17] [FILLS=512,4096] [TOKENS=32] [CTX=8192] [OPTS='{"attnFlash":true}'] \
//     deno run --unstable-webgpu --allow-read --allow-env bench_dense.js
// Prefill goes through prefillTokens (the room's solo path); the last prompt token through forwardToken.
import { DenseEngine, makeTokenizer, argmax } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import fs from "node:fs";

const env = (k, d) => Deno.env.get(k) ?? d;
const M = env("DENSE", "qwen17");
const dir = new URL(`../models/${M}/`, import.meta.url).pathname;
const N = +env("TOKENS", 32), MAXSEQ = +env("CTX", 8192);
const FILLS = env("FILLS", "512,4096").split(",").map(Number);
const OPTS = JSON.parse(env("OPTS", "{}"));
const fd = fs.openSync(dir + "model.gguf", "r");
const readAt = (off, len) => { const o = new Uint8Array(len); let g = 0; while (g < len) { const n = fs.readSync(fd, o, g, len - g, off + g); if (n <= 0) break; g += n; } return o; };
const G = parseGGUFHeader(readAt(0, 32 << 20).buffer, { skipTokenizer: true });
const cfg = JSON.parse(await Deno.readTextFile(dir + "config.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "tokenizer.json")));
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErrors++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 300)); });
const L = cfg.num_hidden_layers;
let t0 = performance.now();
const weights = await ggufWeights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
const eng = await DenseEngine.create({ device, cfg, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: MAXSEQ, ...OPTS });
console.log(`${M}: loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s, maxSeq ${MAXSEQ}, opts ${JSON.stringify(OPTS)}`);

// realistic coding context: this repo's source
const need = Math.max(...FILLS) + 8;
let ids = [];
for (const f of ["../engine/qwen35.js", "../room.js", "../engine/gguf.js", "../engine/wgsl/base.js", "../harness/agent.js"]) {
  ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url))));
  if (ids.length >= need) break;
}
while (ids.length < need) ids = ids.concat(ids);
const rows = [];
for (const fill of FILLS) {
  eng.reset();
  // warm-up pass on a short prefix (pipelines, first-use allocations), not timed
  if (!rows.length) { await eng.prefillTokens(ids.slice(0, 64)); await eng.forwardToken(ids[64]); eng.reset(); }
  t0 = performance.now();
  await eng.prefillTokens(ids.slice(0, fill - 1));
  let logits = await eng.forwardToken(ids[fill - 1]);
  const pf = fill / ((performance.now() - t0) / 1000);
  let next = argmax(logits); const gen = [next];
  t0 = performance.now();
  for (let i = 1; i < N; i++) { logits = await eng.forwardToken(next); next = argmax(logits); gen.push(next); }
  const dec = (N - 1) / ((performance.now() - t0) / 1000);
  const r = { fill, prefill: +pf.toFixed(1), decode: +dec.toFixed(2), gen };
  rows.push(r);
  console.log(`fill ${String(fill).padStart(6)}: prefill ${r.prefill} tok/s · decode ${r.decode} tok/s · ${JSON.stringify(tok.decode(gen).slice(0, 60))}`);
}
console.log("RESULT " + JSON.stringify({ model: M, opts: OPTS, rows, gpuErrors }));
if (gpuErrors) Deno.exit(1);
