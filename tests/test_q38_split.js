// Qwen3.8 split across two shards (the 2-device scenario).
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { argmax } from "../engine/engine.js";
import { Qwen35Engine } from "../engine/qwen35.js";
import { q38Context } from "./load_model.js";

export async function run({ device, model }) {
  const G = model.G;
  const L = 64, mid = 33;
  console.log(`host [0,${mid}) + worker [${mid},${L})`);
  const wHost = await model.weights({ lo: 0, hi: mid, hasEmbed: true, hasHead: true });
  const host = await Qwen35Engine.create({ device, meta: G.meta, weights: wHost, layerRange: [0, mid], hasEmbed: true, hasHead: true });
  const wWork = await model.weights({ lo: mid, hi: L, hasEmbed: false, hasHead: false });
  const work = await Qwen35Engine.create({ device, meta: G.meta, weights: wWork, layerRange: [mid, L], hasEmbed: false, hasHead: false });
  console.log("shards ready");

  const tok = model.tokenizer();
  const ids = tok.encode("The capital of France is");
  let pos = 0, logits = null;
  const pipe = async (id) => {
    const h1 = await host.embedRun(id, pos);
    const h2 = await work.runHidden(h1, pos);
    pos++;
    return await host.headFromHidden(h2);
  };
  for (const id of ids) logits = await pipe(id);
  const gen = [];
  const t0 = performance.now();
  for (let i = 0; i < 12; i++) { const n = argmax(logits); gen.push(n); logits = await pipe(n); }
  const text = tok.decode(gen);
  console.log("split :", JSON.stringify(text));
  console.log("speed:", (12 / ((performance.now() - t0) / 1000)).toFixed(2), "tok/s");
  const ok = text === " Paris.\nThe capital of Germany is Berlin.\nThe";
  console.log(ok ? "QWEN3.8 SPLIT PASS ✓" : "SPLIT MISMATCH");
  return ok;
}

if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1);
