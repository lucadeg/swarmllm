// GPU sampling (exp/gpu-sample) checks shared by the model tests (GPU_SAMPLE=1 in test_moe.js,
// test_mtp.js): a greedy sampler that reads GPU candidates, and the head check on random hiddens.
import { argmax } from "../engine/engine.js";
import { topkNaive, readCands } from "../engine/topk.js";

export const GPU_SAMPLE = Deno.env.get("GPU_SAMPLE") !== "0";   // on by default (the engine default); GPU_SAMPLE=0: the logits path
export const ARGMAX_WIDE = (Deno.env.get("ARGMAX_WIDE") ?? (GPU_SAMPLE ? "1" : "0")) === "1";
// greedy that reads either logits or the engine's GPU candidates; .gpu makes the engine sample on the GPU
export const gpuGreedy = Object.assign((x) => (x && x.ids instanceof Uint32Array ? x.ids[0] : argmax(x)), { gpu: { kind: "greedy" } });

// headFromHiddenIds greedy == argmax(headFromHidden) on n random hiddens (and the value and bad
// count), top-40 == the sorted logits on every 8th. Returns the number of mismatches.
export async function checkHeadIds(eng, n = 256, log = console.log) {
  let seed = 4242;
  const rnd = () => { let t = (seed = (seed + 0x6d2b79f5) | 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };
  const { dim, vocab } = eng.dims;
  let bad = 0, badTop = 0, t0 = performance.now(), tIds = 0, tLg = 0;
  for (let i = 0; i < n; i++) {
    const h = Float32Array.from({ length: dim }, () => (rnd() * 2 - 1) * 3);
    let t = performance.now();
    const lg = await eng.headFromHidden(h);
    tLg += performance.now() - t; t = performance.now();
    const g = await eng.headFromHiddenIds(h, { kind: "greedy" });
    tIds += performance.now() - t;
    if (g.ids[0] !== argmax(lg) || g.bad !== 0 || g.vals[0] !== lg[g.ids[0]]) bad++;
    if (i % 8 === 0) {
      const tk = await eng.headFromHiddenIds(h, { kind: "topk", k: 40, temp: 0.8 });
      const ref = readCands(topkNaive(lg, { n: vocab, k: 40 }), 0, 40);
      if (tk.ids.join() !== ref.ids.join() || tk.vals.join() !== ref.vals.join()) badTop++;
    }
  }
  log(`GPU sampling head check: ${n} random hiddens, greedy mismatches ${bad}, top-40 mismatches ${badTop}/${Math.ceil(n / 8)}; ` +
    `head + logits readback ${(tLg / n).toFixed(2)} ms, head + GPU argmax ${(tIds / n).toFixed(2)} ms (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  return bad + badTop;
}
