// Dense engine: engine variants that must be bit-identical give the same logits, bit for bit.
// One engine; each variant is a set of runtime switches (engine fields). For each variant: reset,
// batched prefill of P prompt tokens (prefillTokens), then D greedy decode tokens (forwardToken);
// every logit of every step must equal the reference variant's (compared as u32 bit patterns).
// P > 2048 covers the long-context path. Also checks the one-token prefill path (prefillToken).
//   cd tests && [DENSE=qwen17] [P=2600] [D=6] [VARIANTS='[{},{"attnFast":false}]'] [OPTS='{}'] \
//     deno run --unstable-webgpu --allow-read --allow-env test_dense_exact.js
import { DenseEngine, makeTokenizer, argmax } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import fs from "node:fs";

const env = (k, d) => Deno.env.get(k) ?? d;
const M = env("DENSE", "qwen17"), P = +env("P", 2600), D = +env("D", 6);
const VARIANTS = JSON.parse(env("VARIANTS", '[{"attnFast":false},{}]'));
const OPTS = JSON.parse(env("OPTS", "{}"));
const dir = new URL(`../models/${M}/`, import.meta.url).pathname;
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
const weights = await ggufWeights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
const eng = await DenseEngine.create({ device, cfg, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: Math.max(4096, P + D + 64), ...OPTS });
let ids = tok.encode(await Deno.readTextFile(new URL("../engine/gguf.js", import.meta.url)));   // a file no kopt branch changes, so hashes compare across branches
while (ids.length < P + 16) ids = ids.concat(ids);
const defaults = {};
for (const v of VARIANTS) for (const k of Object.keys(v)) if (!(k in defaults)) defaults[k] = eng[k];
// mode "batched": prefillTokens over P - 1 tokens; mode "one": prefillToken one at a time over
// min(P, 700) - 1 tokens (slow path, shorter)
const run = async (v, mode) => {
  Object.assign(eng, defaults, v);
  eng.reset();
  const p = mode === "one" ? Math.min(P, 700) : P;
  const t0 = performance.now();
  if (mode === "one") { for (let i = 0; i < p - 1; i++) { await eng.prefillToken(ids[i]); if (i % 8 === 7) await device.queue.onSubmittedWorkDone(); } }
  else await eng.prefillTokens(ids.slice(0, p - 1));
  const out = [];
  let lg = await eng.forwardToken(ids[p - 1]);
  out.push(new Uint32Array(lg.buffer.slice(0)));
  for (let i = 1; i < D; i++) { lg = await eng.forwardToken(argmax(lg)); out.push(new Uint32Array(lg.buffer.slice(0))); }
  return { out, ms: performance.now() - t0, p };
};
let fail = 0;
for (const mode of ["batched", "one"]) {
  const runs = [];
  for (const v of VARIANTS) runs.push(await run(v, mode));
  const ref = runs[0];
  runs.forEach((r, k) => {
    let diff = 0, first = -1;
    r.out.forEach((a, s2) => { const b = ref.out[s2]; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { diff++; if (first < 0) first = s2; } });
    let hsh = 0x811c9dc5; for (const a of r.out) for (let i = 0; i < a.length; i++) hsh = Math.imul(hsh ^ a[i], 0x01000193) >>> 0;
    console.log(`${mode} prefill of ${r.p - 1} ${JSON.stringify(VARIANTS[k])}: ${r.out.length} steps, ${diff ? `${diff} logits DIFFER (first at step ${first})` : "bit-identical"} · logits hash ${hsh.toString(16)} · ${(r.ms / 1000).toFixed(2)} s`);
    if (diff) fail++;
  });
}
console.log(fail || gpuErrors ? `DENSE EXACT FAIL (${fail} variants differ, ${gpuErrors} GPU errors)` : "DENSE EXACT PASS");
Deno.exit(fail || gpuErrors ? 1 : 0);
