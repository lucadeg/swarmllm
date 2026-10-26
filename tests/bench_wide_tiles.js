// Tile sweep for the wide prefill GEMM (prefillUbatch): one model load, one engine per tile, prefill tok/s
// of N tokens (best of R) vs the default 16-column path. MODEL=27b|moe N=1024 R=2 U=256
//   TILES='[{"BM":64,"BN":64,"TM":4,"TN":4},...]'
import { Qwen35Engine } from "../engine/qwen35.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH } from "./load_model.js";
const env = (k, d) => Deno.env.get(k) ?? d;
Deno.env.set("PREFILL_UBATCH", env("U", "256"));   // gpuDevice asks for the adapter's workgroup memory
const MODEL = env("MODEL", "27b"), N = +env("N", 1024), R = +env("R", 2), U = +env("U", 256);
const TILES = JSON.parse(env("TILES", JSON.stringify([
  null, { BM: 64, BN: 64, TM: 4, TN: 4 }, { BM: 64, BN: 64, TM: 4, TN: 4, KB: 1 }, { BM: 128, BN: 64, TM: 8, TN: 4 },
  { BM: 64, BN: 128, TM: 4, TN: 8 }, { BM: 128, BN: 128, TM: 8, TN: 8 }, { BM: 128, BN: 32, TM: 4, TN: 4 }, { BM: 32, BN: 128, TM: 4, TN: 4 },
])));
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: L < nBlk });
const ids = Array.from({ length: N }, (_, i) => 1000 + ((i * 7919) % 50000));
for (const t of TILES) {
  let eng;
  try { eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: N + 256, batchCols: 16, coopRowsB: 1, ...(t ? { prefillUbatch: U, prefillTile: t } : {}) }); }
  catch (e) { console.log(`${JSON.stringify(t)}: ${e.message}`); continue; }
  if (eng.mtp) eng.mtpFill = true;
  let best = 0;
  for (let r = 0; r < R + 1; r++) {
    eng.reset();
    const t0 = performance.now(); await eng.prefillTokens(ids); await device.queue.onSubmittedWorkDone();
    const tps = N / ((performance.now() - t0) / 1000); if (r) best = Math.max(best, tps);
  }
  console.log(`${t ? JSON.stringify(eng.wideCfg) : "default 16-col"}: ${best.toFixed(1)} tok/s  errors ${errors.count}`);
  eng.destroy?.();
}
Deno.exit(0);
