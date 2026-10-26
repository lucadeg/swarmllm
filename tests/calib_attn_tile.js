// Calibration for test_attn_tile.js: how far each prefill path lands from the token-by-token decode
// path (forwardToken for every prompt token, attn_flash per column). relDiff = max|diff| / max|ref|.
//   MODEL=27b|moe LENS=700 CTX=4096 LAYERS=all|N
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, Q38_PATH, MOE_PATH } from "./load_model.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), CTX = +env("CTX", 4096);
const LENS = env("LENS", "700").split(",").map(Number);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "27b" ? Q38_PATH : MOE_PATH);
const G = model.G, tok = model.tokenizer();
const L = env("LAYERS", "all") === "all" ? trunkLayers(G) : +env("LAYERS");
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: false });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true,
  maxSeq: CTX, batchCols: 16, coopRowsB: 1, attnPrefillTile: true });
console.log(`${MODEL} layers ${L}: tile ${JSON.stringify(eng.attnPTCfg)}`);
let ids = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js"]) ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url))));
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(a[i])); } return md / sc; };
const batch = async (tile, p, t2 = true) => { eng.attnPrefillTile = tile; eng.attnTile = t2 && eng.attnTileOn; eng.reset(); await eng.prefillTokens(p.slice(0, -1)); return (await eng.forwardToken(p[p.length - 1])).slice(); };
const seq = async (p) => { eng.reset(); let lg; for (const t of p) lg = await eng.forwardToken(t); return lg.slice(); };
for (const len of LENS) {
  const p = ids.slice(0, len);
  const S = await seq(p), A = await batch(false, p), B = await batch(true, p), C = await batch(false, p, false);
  console.log(`${len}: rel(old batch, seq) ${rel(S, A).toExponential(2)}  rel(tile batch, seq) ${rel(S, B).toExponential(2)}  rel(old, tile) ${rel(A, B).toExponential(2)}  rel(old no-t2, seq) ${rel(S, C).toExponential(2)}  argmax seq/old/tile ${argmax(S)}/${argmax(A)}/${argmax(B)}`);
}
console.log("gpu errors", errors.count);
Deno.exit(0);
