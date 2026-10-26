// The narrow batched twins (8- and 4-column kernels) must produce output
// bit-identical to the full-width kernel on the live columns, at every column
// count the engine dispatches (verify passes are 2..8 wide, prefill tails 4/8).
// A slot-layout bug in the reduction once made twins differ by O(1) while the
// sequential-vs-batched test still passed; this is the check for that class.
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { Qwen35Engine } from "../engine/qwen35.js";
import { q38Context } from "./load_model.js";

export async function run({ device, model }) {
  const G = model.G;
  const L = +(Deno.env.get("LAYERS") || 8), NC = +(Deno.env.get("BCOLS") || 16);
  const w = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
  const eng = await Qwen35Engine.create({ device, meta: G.meta, weights: w, vocab: 248320, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 64,
    batchCols: NC, coopRowsB: +(Deno.env.get("ROWSB") || (NC === 16 ? 1 : NC === 8 ? 2 : 4)) });
  const rel = (a, b) => { let n = 0, d = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; } return Math.sqrt(n / d); };
  let fail = 0;
  for (const n of [2, 4, 6, 8, NC].filter((x, i, arr) => x <= NC && arr.indexOf(x) === i)) {
    const ids = Array.from({ length: n }, (_, i) => 1000 + i * 37);
    const go = async (b4) => { eng.reset(); eng.pos = 0; eng.b4 = b4; eng.gemm = false; return Float32Array.from(await eng.embedRunBatch(ids, 0)); };
    const twin = await go(true), full = await go(false), r = rel(twin, full);
    // NaN-aware: a non-finite output fails even when both sides agree
    const finite = twin.every(Number.isFinite) && full.every(Number.isFinite);
    if (r !== 0 || !finite) fail++;
    console.log(`n=${String(n).padEnd(2)} twin vs full-width: relDiff ${r.toExponential(2)} ${!finite ? "NON-FINITE" : r === 0 ? "identical" : "MISMATCH"}`);
  }
  console.log(fail ? "TWINS FAIL" : "TWINS PASS ✓ (bit-identical at every width)");
  return !fail;
}

if (import.meta.main) Deno.exit((await run(await q38Context({ skipTokenizer: true }))) ? 0 : 1);
