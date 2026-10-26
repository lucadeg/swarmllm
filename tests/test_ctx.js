// Context beyond 512 positions on the 27B: (1) a short prompt gives bit-identical logits with
// maxSeq 512 and 2048 (the cache size is only a stride), (2) a ~640-token prompt prefills and
// generates past position 512 with no NaN, no GPU error and coherent text.
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { argmax } from "../engine/engine.js";
import { Qwen35Engine } from "../engine/qwen35.js";
import { q38Context, wideOpts } from "./load_model.js";

export async function run({ device, model, errors }) {
  const G = model.G;
  const errors0 = errors.count, gpuErrors = () => errors.count - errors0;   // errors raised during this check
  const L = 64;
  console.log("loading all", L, "layers…");
  const tok = model.tokenizer();
  // an engine takes ownership of its weight buffers (CPU copies are freed after upload), so each
  // engine loads its own copy (in the one-process runner both share the preuploaded set instead)
  const mk = async (maxSeq) => {
    const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true });
    return Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1, ...wideOpts() });   // PREFILL_UBATCH: the ~640-token prompt through the wide path
  };

  // (1) short prompt, two cache sizes, same logits bit for bit
  const short = tok.encode("The capital of France is");
  const fwd = async (eng, ids) => { let lg = null; for (const id of ids) lg = await eng.forwardToken(id); return lg; };
  const e512 = await mk(512); const a = Float32Array.from(await fwd(e512, short));
  try { e512.destroy?.(); } catch {}
  const e2048 = await mk(2048); const b = await fwd(e2048, short);
  let diff = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
  console.log("short prompt, maxSeq 512 vs 2048: differing logits =", diff);
  if (diff) { console.log("CTX FAIL: cache size changed the logits"); return false; }

  // (2) long prompt past the old limit: batched prefill, then greedy generation across position 512
  const para = "I am planning a two week trip through Japan in late October with my partner. We land in Tokyo, want three days there, then a day trip to Nikko, then the bullet train to Kyoto for four days with a side trip to Nara, then two nights in Osaka, and we fly home from Osaka. We like food markets, old temples, hiking, and small neighborhood bars, and we want to avoid the most crowded tourist spots where we can. Our budget is moderate, around two hundred dollars a day for the two of us not counting hotels. ";
  const V = tok.vocab;
  const ids = [V["<|im_start|>"], ...tok.encode("user\n" + para.repeat(4) + "Please give me a day by day itinerary."), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  console.log("long prompt tokens:", ids.length);
  e2048.reset?.(); e2048.pos = 0;
  let t0 = performance.now();
  await e2048.prefillTokens(ids.slice(0, -1));
  let logits = await e2048.forwardToken(ids[ids.length - 1]);
  console.log("prefill", ((performance.now() - t0) / 1000).toFixed(1), "s, pos now", e2048.pos);
  const gen = []; let nan = false;
  for (let i = 0; i < 48; i++) {
    if (!Number.isFinite(logits[0]) || !Number.isFinite(logits[logits.length - 1])) { nan = true; break; }
    const next = argmax(logits); gen.push(next);
    if (next === V["<|im_end|>"]) break;
    logits = await e2048.forwardToken(next);
  }
  const text = tok.decode(gen);
  const uniq = new Set(gen).size / Math.max(1, gen.length);
  console.log("generated past pos 512:", JSON.stringify(text));
  console.log(`positions ${ids.length}..${e2048.pos}, NaN: ${nan}, GPU errors: ${gpuErrors()}, unique-token ratio ${uniq.toFixed(2)}`);
  const ok = !nan && gpuErrors() === 0 && gen.length >= 8 && uniq > 0.5 && !/\(\s*\(\s*\(/.test(text);
  console.log(ok ? "\nCTX PASS ✓ (prefill and decode past the 512-position boundary are clean)" : "\nCTX FAIL");
  return ok;
}

if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1);
