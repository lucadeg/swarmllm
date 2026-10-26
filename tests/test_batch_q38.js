// Qwen3.8 batched-prefill equivalence on the first 6 layers (covers both
// DeltaNet and full-attention layer types + a batch tail).
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { q38Context } from "./load_model.js";

export async function run({ device, model }) {
  const G = model.G, tok = model.tokenizer();
  const L = 6;
  const mk = async () => {
    const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
    return Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 64,
    batchCols: +(Deno.env.get("BCOLS") || 4), coopRowsB: +(Deno.env.get("ROWSB") || 4) });
  };
  const ids = tok.encode("The capital of France is Paris, and the capital of Germany is Berlin. The quick brown");
  console.log("prompt tokens:", ids.length);
  const e1 = await mk();
  let ref = null;
  for (const id of ids) ref = await e1.forwardToken(id);
  const e2 = await mk();
  const t0 = performance.now();
  await e2.prefillTokens(ids.slice(0, -1));
  const t1 = performance.now();
  e2.pos = ids.length - 1;
  const got = await e2.forwardToken(ids[ids.length - 1]);
  let md = 0, sc = 1e-6;
  for (let i = 0; i < ref.length; i++) { md = Math.max(md, Math.abs(got[i] - ref[i])); sc = Math.max(sc, Math.abs(ref[i])); }
  const rel = md / sc;
  console.log(`argmax seq=${argmax(ref)} batch=${argmax(got)}  relDiff=${rel.toExponential(2)}  batchedPrefill=${((ids.length - 1) / ((t1 - t0) / 1000)).toFixed(1)} tok/s`);
  // NaN-aware: all-NaN or all-equal logits give argmax 0 == 0 and relDiff 0 (that passed on Metal with every f32
  // weight zeroed by the one-process upload); demand finite, non-constant logits on both paths
  const sane = (a) => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (!Number.isFinite(v)) return false; lo = Math.min(lo, v); hi = Math.max(hi, v); } return hi > lo; };
  const finite = sane(ref) && sane(got);
  if (!finite) console.log("non-finite or constant logits");
  const ok = finite && argmax(ref) === argmax(got) && rel < 2e-3;
  console.log(ok ? "Q38 BATCH PREFILL PASS ✓" : "Q38 BATCH PREFILL FAIL");
  return ok;
}

if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1);
