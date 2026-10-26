// 4-token batched pass at 8 batch columns: the 4-column twin kernels must match
// the 8-column kernels (and the sequential path) bit-close.
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { q38Context } from "./load_model.js";

export async function run({ device, model }) {
  const G = model.G;
  const L = +(Deno.env.get("LAYERS") || 8);
  const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
  const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, vocab: 248320, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 64, batchCols: 8, coopRowsB: 2 });
  const ids = [760, 6511, 315, 9109, 3139, 1234, 42, 7, 999, 31337, 2048, 4096];
  const runMode = async (mode) => {   // mode: "seq" | "b8" | "b4"
    eng.reset(); eng.pos = 0; eng.b4 = mode === "b4";
    let last;
    if (mode === "seq") { for (const t of ids) last = await eng.forwardToken(t); }
    else { await eng.embedRunBatch(ids.slice(0, 8), 0); const hs = await eng.embedRunBatch(ids.slice(8, 12), 8); last = (await eng.headBatch(hs, 4))[3]; }
    return Float32Array.from(last);
  };
  const seq = await runMode("seq"), b8 = await runMode("b8"), b4 = await runMode("b4");
  const rel = (a, b) => { let num = 0, den = 0; for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += a[i] ** 2; } return Math.sqrt(num / den); };
  console.log(`argmax seq=${argmax(seq)} b8=${argmax(b8)} b4=${argmax(b4)}  relDiff b8=${rel(seq, b8).toExponential(2)} b4=${rel(seq, b4).toExponential(2)}`);
  const ok = argmax(seq) === argmax(b4) && rel(seq, b4) < 1e-3;
  console.log(ok ? "B4 TWIN PASS ✓" : "B4 TWIN FAIL");
  return ok;
}

if (import.meta.main) Deno.exit((await run(await q38Context({ skipTokenizer: true }))) ? 0 : 1);
