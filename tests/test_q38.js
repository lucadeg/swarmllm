// Qwen3.8 GPU engine vs validated CPU reference (layers 0..4, 2 tokens).
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { Qwen35Engine } from "../engine/qwen35.js";
import { q38Context } from "./load_model.js";

export async function run({ adapter, device, model }) {
  const G = model.G;
  if (adapter) console.log("maxStorageBinding:", (adapter.limits.maxStorageBufferBindingSize / 2 ** 30).toFixed(2), "GB");
  const LO = 0, HI = 5;
  console.log("loading shard layers", LO, "..", HI - 1);
  const t0 = performance.now();
  const weights = await model.weights({ lo: LO, hi: HI, hasEmbed: true, hasHead: false });
  console.log("weights in", ((performance.now() - t0) / 1000).toFixed(1), "s");
  const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [LO, HI], hasEmbed: true, hasHead: false });
  console.log("engine ready");

  const tok = model.tokenizer();
  const ids = tok.encode("The capital");
  console.log("ids:", JSON.stringify(ids));

  // reference layer_out-4 values from ref_q38.mjs (validated vs llama.cpp):
  const REF = [
    [-0.04135, -0.00732],  // token 0: first two dims
    [0.12867, -0.13125],   // token 1
  ];
  let fail = 0;
  for (let ti = 0; ti < ids.length; ti++) {
    const h = await eng.embedRun(ids[ti], ti);
    const ok = Math.abs(h[0] - REF[ti][0]) < 0.02 && Math.abs(h[1] - REF[ti][1]) < 0.02;
    if (!ok) fail++;
    console.log(`token ${ti}: engine [${h[0].toFixed(5)}, ${h[1].toFixed(5)}] ref [${REF[ti]}] ${ok ? "MATCH" : "MISMATCH"}`);
  }
  return fail === 0;
}

if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1);
