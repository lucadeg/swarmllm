// Expert-grouped MoE prefill (engine option moeGroupPrefill, engine/wgsl/moe_group.js) on the real
// Qwen3.6-35B-A3B: the grouped path must give the SAME BITS as the per-pass path, not merely close ones.
// For each prompt (lengths chosen to hit whole ubatches plus a tail through the 16 / 8 / 4 passes and
// single tokens): prefill with engine.moeGroup off and on, compare the next-token logits element by element
// (===), then N greedy tokens, then speculative decoding with the draft cache filled during prefill
// (identical tokens and identical draft acceptance, which also checks the MTP fill). Prints the prefill
// speed of both paths and what the sort saw (pairs, distinct routed experts, chunks).
//   U=64 (ubatch, a multiple of BCOLS) UC=8 BCOLS=16 N=16 LENS=150,301,700 [PERF=2048]
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_moe_group.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, prefillTol } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const U = +env("U", 64), UC = +env("UC", 8), NC = +env("BCOLS", 16), N = +env("N", 16), K = +env("K", 3);
const TILED = env("TILED", "0") === "1";   // tiled kernels: not bit-identical; judged by relDiff < the MoE prefill tolerance (2e-2, load_model.js prefillTol) and argmax
const LENS = env("LENS", "150,301,700").split(",").map(Number), PERF = +env("PERF", 2048);
const { device } = await gpuDevice();
watchGpuErrors(device);
const model = openGGUF(env("MOE", MOE_PATH));
const G = model.G, arch = G.meta["general.architecture"], nBlk = G.meta[arch + ".block_count"];
const hasMtp = Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`));
const L = trunkLayers(G), tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const maxSeq = Math.ceil((Math.max(...LENS, PERF) + 2 * N + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq,
  batchCols: NC, coopRowsB: NC >= 16 ? 1 : 4, moeGroupPrefill: U, moeGroupUC: UC, moeGroupTiled: TILED });
if (!eng.moeGrpU) { console.log("MOE GROUP FAIL: the engine did not enable moeGroupPrefill (see the warning above)"); Deno.exit(1); }
console.log(`${arch}: ${L} layers, mtp ${!!eng.mtp}, batchCols ${eng.NC}, moeGroupPrefill ${eng.moeGrpU} UC ${eng.moeGrpUC}${eng.moeGrpTiled ? " TILED" : ""}, maxSeq ${maxSeq}`);

// realistic text: this repo's own source
let ids = [];
for (const f of ["../engine/qwen35.js", "../engine/wgsl/moe.js", "../room.js"]) {
  ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url))));
  if (ids.length > Math.max(...LENS, PERF) + 8) break;
}

// prefill ids[0 .. n-2], next-token logits from ids[n-1], then N greedy tokens; optionally speculative decoding
async function runOnce(grouped, n, spec) {
  eng.reset(); eng.moeGroup = grouped;
  if (eng.mtp) { eng.mtpFill = spec; eng.mtp.stats = { drafts: 0, accepted: 0 }; }
  const p = ids.slice(0, n);
  const t0 = performance.now();
  await eng.prefillTokens(p.slice(0, -1));
  let logits = await eng.forwardToken(p[p.length - 1]);
  const pf = (performance.now() - t0) / 1000, first = Float32Array.from(logits);
  let next = argmax(logits); const gen = [next];
  if (spec) { while (gen.length < N) { for (const t of await eng.specStep(next, argmax, K)) gen.push(t); next = gen[gen.length - 1]; } gen.length = N; }
  else for (let i = 1; i < N; i++) { logits = await eng.forwardToken(next); next = argmax(logits); gen.push(next); }
  return { first, gen, pf, stats: eng.mtp ? { ...eng.mtp.stats } : null, grp: grouped ? await eng.moeGroupStats() : null };
}

let fail = 0;
for (const n of LENS) {
  const a = await runOnce(false, n, false), b = await runOnce(true, n, false);
  let diff = 0, maxRel = 0;
  for (let i = 0; i < a.first.length; i++) if (!Object.is(a.first[i], b.first[i])) { diff++; maxRel = Math.max(maxRel, Math.abs(a.first[i] - b.first[i]) / (Math.abs(a.first[i]) + 1e-6)); }
  const same = a.gen.every((t, i) => t === b.gen[i]);
  let md = 0, sc = 1e-6;
  for (let i = 0; i < a.first.length; i++) { md = Math.max(md, Math.abs(a.first[i] - b.first[i])); sc = Math.max(sc, Math.abs(a.first[i])); }
  const relDiff = md / sc, am = argmax(a.first) === argmax(b.first);
  let g = 0; while (n - 1 - g >= 2 * NC) g += Math.min(U, Math.floor((n - 1 - g) / NC) * NC);   // prefillTokens' split
  const ub = g;
  console.log(`prompt ${n} tok (${g} through grouped ubatches, ${n - 1 - g} through the ordinary passes): logits ${diff ? `DIFFER in ${diff} (max rel ${maxRel.toExponential(2)})` : "bit-identical"}, ` +
    `greedy ${same ? "identical" : "DIFFERS"} · prefill ${a.pf.toFixed(2)}s -> ${b.pf.toFixed(2)}s · last sort ${JSON.stringify(b.grp)}`);
  if (TILED) console.log(`  tiled: relDiff ${relDiff.toExponential(2)} (gate ${prefillTol(true)}), argmax ${am ? "same" : "DIFFERS"}, greedy ${N} ${same ? "same" : "differs (informational)"}`);
  if ((TILED ? !(am && relDiff < prefillTol(true)) : diff || !same) || !ub) fail++;
  if (eng.mtp) {
    const sa = await runOnce(false, n, true), sb = await runOnce(true, n, true);
    // tiled: speculative decoding must equal plain greedy within each path (spec == plain), not across paths
    const ok = TILED ? sa.gen.every((t, i) => t === a.gen[i]) && sb.gen.every((t, i) => t === b.gen[i])
      : sa.gen.every((t, i) => t === sb.gen[i]) && sa.gen.every((t, i) => t === a.gen[i]) && sa.stats.accepted === sb.stats.accepted && sa.stats.drafts === sb.stats.drafts;
    console.log(`  spec K=${K} with the draft cache from prefill: ${ok ? "identical" : "DIFFERS"} (accepted ${sa.stats.accepted}/${sa.stats.drafts} vs ${sb.stats.accepted}/${sb.stats.drafts})`);
    if (!ok) fail++;
  }
}
if (PERF > 0) {
  const n = Math.floor(PERF / U) * U + 1;   // whole ubatches: prefill n - 1 tokens
  for (const g of [false, true, false, true]) {
    const r = await runOnce(g, n, false);
    console.log(`perf ${n - 1} tok, ${g ? "grouped " : "per-pass"}: ${((n - 1) / r.pf).toFixed(1)} tok/s (${r.pf.toFixed(2)}s)`);
  }
}
eng.moeGroup = true;
console.log(fail ? "MOE GROUP FAIL" : "MOE GROUP PASS ✓"); if (fail) Deno.exit(1);
