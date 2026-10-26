// MTP speculative decoding: greedy output must be IDENTICAL to plain greedy
// decoding (verification uses the trunk; the draft can only affect speed).
// Prints acceptance rate and tok/s for both paths. Full 64-layer 27B.
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { q38Context } from "./load_model.js";
import { GPU_SAMPLE, ARGMAX_WIDE, gpuGreedy, checkHeadIds } from "./gpusample_check.js";

export async function run({ device, model }) {
  const N = +(Deno.env.get("TOKENS") || 40);
  const K = +(Deno.env.get("K") || 3);
  const G = model.G, tok = model.tokenizer();
  const L = +(Deno.env.get("LAYERS") || (G.meta["qwen35.block_count"] - 1));
  const t0 = performance.now();
  const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
  const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512,
    batchCols: +(Deno.env.get("BCOLS") || 4), coopRowsB: +(Deno.env.get("ROWSB") || 4),
    // DRAFTCHAIN=0 / SPECFUSE=0: per-submit drafts / separate verify submits (A/B; same output)
    draftChain: Deno.env.get("DRAFTCHAIN") !== "0", specFuse: Deno.env.get("SPECFUSE") !== "0",
    // GPU_SAMPLE=0: the logits path (GPU sampling and the two-stage draft argmax are on by default; ARGMAX_WIDE=0|1 alone: only the latter)
    gpuSample: GPU_SAMPLE, argmaxWide: ARGMAX_WIDE });
  console.log(`draftChain ${!!eng.draftChain}, specFuse ${eng.specFuse}, gpuSample ${eng.gpuSample}, argmaxWide ${eng.argmaxWide}`);
  console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s; mtp=${!!eng.mtp}`);
  const V = tok.vocab;
  const prompt = [V["<|im_start|>"], ...tok.encode("user\nWrite the Python code for two sum. Code only."), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  // ---- plain greedy ----
  eng.reset(); eng.mtpFill = false;
  await eng.prefillTokens(prompt.slice(0, -1));
  let logits = await eng.forwardToken(prompt[prompt.length - 1]);
  let next = argmax(logits); const plain = [next];
  const tp0 = performance.now();
  for (let i = 1; i < N; i++) { logits = GPU_SAMPLE ? await eng.forwardTokenIds(next) : await eng.forwardToken(next); next = gpuGreedy(logits); plain.push(next); }
  const plainTs = (N - 1) / ((performance.now() - tp0) / 1000);
  // ---- speculative greedy ----
  eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 };
  await eng.prefillTokens(prompt.slice(0, -1));
  logits = await eng.forwardToken(prompt[prompt.length - 1]);
  next = argmax(logits); const spec = [next];
  const ts0 = performance.now();
  while (spec.length < N) {
    const got = await eng.specStep(next, GPU_SAMPLE ? gpuGreedy : argmax, K);
    for (const t of got) spec.push(t);
    next = spec[spec.length - 1];
  }
  const specTs = (spec.length - 1) / ((performance.now() - ts0) / 1000);
  const same = plain.every((t, i) => spec[i] === t);
  const st = eng.mtp.stats;
  console.log(`K=${K} plain: ${plainTs.toFixed(2)} tok/s   spec: ${specTs.toFixed(2)} tok/s   acceptance ${st.accepted}/${st.drafts} = ${(100 * st.accepted / Math.max(1, st.drafts)).toFixed(0)}%`);
  console.log("plain:", JSON.stringify(tok.decode(plain).slice(0, 120)));
  console.log("spec: ", JSON.stringify(tok.decode(spec.slice(0, N)).slice(0, 120)));
  console.log(same ? "MTP SPEC PASS ✓ (identical output)" : "MTP SPEC FAIL (output differs)");
  if (GPU_SAMPLE && await checkHeadIds(eng)) { console.log("GPU SAMPLING FAIL (head check)"); return false; }
  return same;
}

if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1);
