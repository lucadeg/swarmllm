// Speculative decoding across a device chain: host (layers 0..S, embed, head,
// MTP) + worker (layers S..64) with the room's protocol simulated locally:
// batched verify with snapshots on both, rollback message on rejection.
// Output must be identical to plain split decoding.
// Standalone, or as a check of tests/run_q38_once.js (export run(ctx), see tests/load_model.js).
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { q38Context } from "./load_model.js";

export async function run({ device, model }) {
  const N = +(Deno.env.get("TOKENS") || 32), K = +(Deno.env.get("K") || 3);
  const G = model.G, tok = model.tokenizer();
  const L = G.meta["qwen35.block_count"] - 1, S = +(Deno.env.get("SPLIT") || 24);
  const host = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, S], hasEmbed: true, hasHead: true, maxSeq: 512,
    weights: await model.weights({ lo: 0, hi: S, hasEmbed: true, hasHead: true, mtp: true }) });
  const worker = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [S, L], hasEmbed: false, hasHead: false, maxSeq: 512,
    vocab: G.tensors["token_embd.weight"].shape[0], weights: await model.weights({ lo: S, hi: L, hasEmbed: false, hasHead: false }) });
  console.log(`host layers 0-${S - 1} (+mtp), worker ${S}-${L - 1}`);
  const V = tok.vocab;
  const prompt = [V["<|im_start|>"], ...tok.encode("user\nWrite the Python code for two sum. Code only."), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  // chain primitives (what the room does over WebRTC)
  const pipe = async (id, pos) => { const h1 = await host.embedRun(id, pos); const h2 = await worker.runHidden(h1, pos); return { h: h2, logits: await host.headFromHidden(h2) }; };
  async function prefill(eng2) { // split prefill: 4-token batches through both
    let pos = 0;
    for (; prompt.length - 1 - pos >= 4; pos += 4) { const hb = await host.embedRunBatch(prompt.slice(pos, pos + 4), pos); await worker.runHiddenBatch(hb, pos); }
    let out = null;
    for (; pos < prompt.length; pos++) out = await pipe(prompt[pos], pos);
    return out;
  }
  // ---- plain ----
  host.reset(); worker.reset(); host.mtpFill = false;
  let r = await prefill(); let next = argmax(r.logits); const plain = [next]; let pos = prompt.length;
  const tp0 = performance.now();
  for (let i = 1; i < N; i++) { r = await pipe(next, pos++); next = argmax(r.logits); plain.push(next); }
  const plainTs = (N - 1) / ((performance.now() - tp0) / 1000);
  // ---- speculative over the chain ----
  host.reset(); worker.reset(); host.mtpFill = true; host.mtp.stats = { drafts: 0, accepted: 0 };
  // prefill must also fill the draft cache: use the same batched path but with mtp fills (host side)
  {
    let p2 = 0;
    for (; prompt.length - 1 - p2 >= 4; p2 += 4) {
      const hb = await host.embedRunBatch(prompt.slice(p2, p2 + 4), p2);
      await worker.runHiddenBatch(hb, p2);
    }
    for (; p2 < prompt.length; p2++) r = await pipe(prompt[p2], p2);
  }
  host.setHidden(r.h); host.pos = prompt.length;
  next = argmax(r.logits); const spec = [next];
  let rollbacks = 0;
  const opts = {
    runTrunk: async (tokens, p) => { const hb = await host.embedRunBatch(tokens, p, true); return worker.runHiddenBatch(hb, p, true); },
    onReject: async (k) => { rollbacks++; worker.restoreDN(k); },
  };
  const ts0 = performance.now();
  while (spec.length < N) { const got = await host.specStep(next, argmax, K, opts); for (const t of got) spec.push(t); next = spec[spec.length - 1]; }
  const specTs = (spec.length - 1) / ((performance.now() - ts0) / 1000);
  const same = plain.every((t, i) => spec[i] === t);
  const st = host.mtp.stats;
  console.log(`chain K=${K}: plain ${plainTs.toFixed(2)} tok/s  spec ${specTs.toFixed(2)} tok/s  acceptance ${st.accepted}/${st.drafts}  rollbacks ${rollbacks}`);
  console.log("plain:", JSON.stringify(tok.decode(plain).slice(0, 100)));
  console.log("spec: ", JSON.stringify(tok.decode(spec.slice(0, N)).slice(0, 100)));
  console.log(same ? "MTP CHAIN PASS ✓ (identical output)" : "MTP CHAIN FAIL (output differs)");
  return same;
}

if (import.meta.main) Deno.exit((await run(await q38Context())) ? 0 : 1);
