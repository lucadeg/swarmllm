// CPU checks for the opt-in prefill GEMM operand precisions (prefillMath "f16" / "sgmatrix"):
//  - the tensor-core kernel's JavaScript twin (engine/wgsl/gemm_sgm.js sgmGemmJS: same plan, same
//    index functions as the WGSL generator, subgroup-matrix builtins emulated as specified) writes
//    every partial exactly once and matches a float64 GEMM on f16-rounded operands, for Q4_0 and
//    Q8_0, several MMA shapes and subgroup sizes, and every split-K factor;
//  - the size of the f16 operand rounding against the f32 GEMM (what prefillMath f16 / sgmatrix
//    change), on weights and activations shaped like the real ones;
//  - the WGSL generators: every entry point the engine asks for, no unfilled template values, and
//    the default f32 module unchanged when R16 is off;
//  - pickSgmConfig and the engine's per-op routing (_dop) and state signature, on a stub engine.
// No GPU.   deno test --allow-read tests/unit/gemm_sgm_test.js
import { sgmPlan, sgmGemmJS, gemmSgmWGSL, pickSgmConfig, SGM_TM } from "../../engine/wgsl/gemm_sgm.js";
import { gemmWGSL, GEMM_S, GEMM_TILE } from "../../engine/wgsl/gemm.js";
import { Qwen35Engine, PREFILL_MATH, prefillMathFeatures } from "../../engine/qwen35.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const assert = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const h = Math.f16round ?? ((v) => f16ToF32(f32ToF16(v)));
let seed = 4242;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

// random weights in the engine layout; deq(r, k) = f32 dequantized value (exact: q * f16 scale)
function quant(q8, rows, dIn) {
  const nb = dIn / 32, W = q8 ? 8 : 4;
  const qs = Uint32Array.from({ length: rows * nb * W }, () => (rnd() * 4294967296) >>> 0);
  const sch = Uint16Array.from({ length: rows * nb }, () => f32ToF16((0.004 + rnd() * 0.03) * (rnd() < 0.5 ? -1 : 1)));
  const deq = (r, k) => {
    const b = Math.floor(k / 32), kk = k % 32, s = f16ToF32(sch[r * nb + b]);
    if (!q8) { const w = qs[(r * nb + b) * 4 + ((kk % 16) >> 2)]; return Math.fround((((w >>> (8 * (kk % 4) + (kk >= 16 ? 4 : 0))) & 15) - 8) * s); }
    const w = qs[(r * nb + b) * 8 + (kk >> 2)]; return Math.fround(((w << (24 - 8 * (kk % 4))) >> 24) * s);
  };
  return { qs, sc: new Uint32Array(sch.buffer), deq };
}

function checkTwin({ q8, dIn, S, dOut, N = 16, cfg, KB = 1, TM = SGM_TM, PAD = 8 }) {
  const P = sgmPlan({ q8, dIn, S, N, KB, TM, PAD, ...cfg });
  const Wt = quant(q8, dOut, dIn);
  const xT = Float32Array.from({ length: dIn * N }, () => Math.fround(gauss()));
  const y = new Float64Array(S * N * dOut + 64).fill(NaN);
  sgmGemmJS(P, { qs: Wt.qs, sc: Wt.sc, xT, y, dOut, f16round: h });
  for (let i = 0; i < S * N * dOut; i++) assert(Number.isFinite(y[i]), `partial ${i} never written`);
  for (let i = S * N * dOut; i < y.length; i++) assert(Number.isNaN(y[i]), `write past the partials at ${i}`);
  let worst = 0;
  for (let r = 0; r < dOut; r++) {
    const w = Array.from({ length: dIn }, (_, k) => h(Wt.deq(r, k)));
    for (let c = 0; c < N; c++) {
      let ref = 0, mag = 0; for (let k = 0; k < dIn; k++) { const t = w[k] * h(xT[k * N + c]); ref += t; mag += Math.abs(t); }
      let got = 0; for (let s = 0; s < S; s++) got += y[(s * N + c) * dOut + r];
      worst = Math.max(worst, Math.abs(got - ref) / Math.max(mag, 1e-30));
    }
  }
  assert(worst < 1e-12, `twin vs float64 (f16 operands): ${worst}`);
  return worst;
}

const CFGS = [
  { M: 16, Nm: 16, Kc: 16, SG: 32 },   // NVIDIA / AMD cooperative-matrix f16 (Vulkan)
  { M: 8, Nm: 8, Kc: 8, SG: 32 },      // Apple simdgroup_matrix
  { M: 16, Nm: 8, Kc: 16, SG: 32 },    // non-square
  { M: 16, Nm: 16, Kc: 16, SG: 64 },   // wave64
];

Deno.test("sgm twin: Q4_0 and Q8_0, every MMA shape, split-K 1..8", () => {
  for (const cfg of CFGS) for (const q8 of [false, true]) for (const [dIn, S] of [[256, 1], [512, 2], [1024, 4], [2048, 8]]) {
    checkTwin({ q8, dIn, S, dOut: 2 * SGM_TM, cfg });
  }
});

Deno.test("sgm twin: tuning knobs (KB = 2, TM = 64 / 256, PAD 0 / 16)", () => {
  for (const q8 of [false, true]) {
    checkTwin({ q8, dIn: 1024, S: 2, dOut: SGM_TM, cfg: CFGS[0], KB: 2 });
    checkTwin({ q8, dIn: 1024, S: 2, dOut: 192, cfg: CFGS[0], KB: 2, TM: 64, PAD: 0 });
    checkTwin({ q8, dIn: 512, S: 1, dOut: 512, cfg: CFGS[1], TM: 256, PAD: 16 });
  }
});

Deno.test("sgm plan: rejects shapes it cannot tile", () => {
  const bad = [{ dIn: 96, S: 1 }, { dIn: 512, S: 3 }, { dIn: 512, S: 2, M: 24 }, { dIn: 512, S: 2, SG: 48 }, { dIn: 512, S: 2, PAD: 4 }];
  for (const b of bad) {
    let threw = false;
    try { sgmPlan({ q8: false, N: 16, ...CFGS[0], ...b }); } catch { threw = true; }
    assert(threw, "should reject " + JSON.stringify(b));
  }
});

// What f16 operands cost in accuracy: one GEMM output vs the f32-operand result, activations from
// a unit Gaussian plus a few large outlier channels (like FFN down inputs).
Deno.test("f16 operands: size of the change vs f32 operands (report)", () => {
  const out = [];
  for (const q8 of [false, true]) for (const outl of [0, 50]) {
    const dIn = 2048, rows = 64, N = 4, Wt = quant(q8, rows, dIn);
    const x = Array.from({ length: N }, () => Float32Array.from({ length: dIn }, (_, k) => Math.fround(gauss() * (outl && k % 211 === 0 ? outl : 1))));
    let num = 0, den = 0, maxRel = 0;
    for (let r = 0; r < rows; r++) for (let c = 0; c < N; c++) {
      let a = 0, b = 0; for (let k = 0; k < dIn; k++) { const w = Wt.deq(r, k); a += w * x[c][k]; b += h(w) * h(x[c][k]); }
      num += (a - b) ** 2; den += a * a; maxRel = Math.max(maxRel, Math.abs(a - b) / Math.max(Math.abs(a), 1e-3));
    }
    const rel = Math.sqrt(num / den);
    out.push(`${q8 ? "q8" : "q4"} outliers ${outl}: rel L2 ${rel.toExponential(2)}`);
    assert(rel > 1e-5 && rel < 2e-3, `f16 operand rounding error out of the expected range: ${rel}`);
  }
  console.log("  " + out.join("\n  "));
});

Deno.test("WGSL: sgm module has every entry point, no unfilled values", () => {
  const pairs = [[5120, 4], [5120, 8], [5120, 2], [5120, 16], [17408, 2], [6144, 8]], pairs8 = [[17408, 2], [6144, 8], [5120, 4]];
  for (const cfg of CFGS) {
    const code = gemmSgmWGSL({ N: 16, cfg, pairs, pairs8 });
    assert(code.startsWith("enable f16;\nenable chromium_experimental_subgroup_matrix;"), "enable directives first");
    assert(/subgroupMatrixLoad<subgroup_matrix_left<f16, \d+, \d+>, row_major>\(&sg_W, \d+u, \d+u\)/.test(code), "template-majorness load");
    assert(/subgroupMatrixStore<col_major>\(&sg_y, .*, c0_0, dOut\)/.test(code), "template-majorness store");
    const old = gemmSgmWGSL({ N: 16, cfg, pairs, pairs8, syntax: "bool" });
    assert(/subgroupMatrixLoad<subgroup_matrix_right<f16, \d+, \d+>>\(&sg_X, \d+u, false, 16u\)/.test(old) && /subgroupMatrixStore\(&sg_y, .*, c0_0, true, dOut\)/.test(old), "bool-majorness spelling");
    for (const [dIn, S] of pairs) assert(code.includes(`fn gemm_sgm_q4_${dIn}_s${S}(`), `q4 ${dIn} s${S}`);
    for (const [dIn, S] of pairs8) assert(code.includes(`fn gemm_sgm_q8_${dIn}_s${S}(`), `q8 ${dIn} s${S}`);
    assert(!/undefined|NaN|\[object/.test(code), "unfilled template value");
    assert((code.match(/{/g) || []).length === (code.match(/}/g) || []).length, "braces balance");
    assert(code.includes(`@workgroup_size(${cfg.SG})`), "one subgroup per workgroup");
  }
});

Deno.test("WGSL: R16 twins are extra entry points; the default module is unchanged by them", () => {
  const o = { N: 16, pairs: [[5120, 4], [17408, 2]], pairs8: [[17408, 2]], UNPACK: true };
  const a = gemmWGSL(o), b = gemmWGSL({ ...o, R16: true });
  assert(!a.includes("r16"), "default module must not contain the f16 twins");
  for (const n of ["gemm_q4_5120_s4_r16", "gemm_q4_17408_s2_r16", "gemm_q8_17408_s2_r16", "gemm_xpose_r16", "fn gm_r16"]) assert(b.includes(n), n);
  // stripping the twins gives back the default module's entry points, in order
  const ep = (s) => [...s.matchAll(/fn (gemm_\w+)\(/g)].map((m) => m[1]).filter((n) => !n.endsWith("_r16"));
  assert(JSON.stringify(ep(a)) === JSON.stringify(ep(b)), "f32 entry points unchanged");
  assert(SGM_TM === GEMM_TILE, "sgm grid must equal the f32 GEMM grid");
  assert(GEMM_S["17408x5120"] === 4, "GEMM_S untouched");
});

Deno.test("pickSgmConfig", () => {
  const c = (componentType, resultComponentType, M, N, K) => ({ componentType, resultComponentType, M, N, K });
  assert(pickSgmConfig({}).cfg === null, "no configs");
  assert(pickSgmConfig({ subgroupMatrixConfigs: [c("f16", "f16", 16, 16, 16)] }).cfg === null, "f16 result only");
  const nv = pickSgmConfig({ subgroupMinSize: 32, subgroupMaxSize: 32, subgroupMatrixConfigs: [c("f16", "f16", 16, 16, 16), c("f16", "f32", 16, 16, 16), c("f32", "f32", 8, 8, 8)] });
  assert(nv.cfg && nv.cfg.M === 16 && nv.cfg.Nm === 16 && nv.cfg.Kc === 16 && nv.cfg.SG === 32 && !nv.why, JSON.stringify(nv));
  const ap = pickSgmConfig({ subgroupMinSize: 32, subgroupMaxSize: 32, subgroupMatrixConfigs: [c("f16", "f32", 8, 8, 8)] });
  assert(ap.cfg && ap.cfg.M === 8, "apple 8x8x8");
  const amd = pickSgmConfig({ subgroupMinSize: 32, subgroupMaxSize: 64, subgroupMatrixConfigs: [c("f16", "f32", 16, 16, 16)] });
  assert(amd.cfg && amd.cfg.SG === 64 && amd.why, "variable subgroup size: workgroup = max, with a note");
  assert(pickSgmConfig({ subgroupMaxSize: 32, subgroupMatrixConfigs: [c("f16", "f32", 16, 32, 16)] }).cfg === null, "N=32 does not tile 16 columns");
});

Deno.test("engine: prefillMath routing in _dop and the state signature (stub engine)", () => {
  const e = Object.create(Qwen35Engine.prototype);
  Object.assign(e, { NC: 16, gemm: true, gemm8: true, _gz: 0, skip: null, _pmAvail: { f32: true, f16: true, sgmatrix: true },
    lo: 0, hi: 2, mtpLayer: null, flash: true, kvQ8: false, dims: { dim: 1, kvDim: 1, nVH: 1, convDim: 1 } });
  const seen = [];
  e._d3 = (pass, pipe, bg, wgs) => seen.push([pipe, bg, wgs]);
  const op = { pipe: "matvec_q4_coop_b", bg: "gemvBG", wgs: 7, pipe8: "g8", bg8: "g8BG", pipe4: "g4", bg4: "g4BG",
    gemm: { pipe: "gemm_q4_5120_s4", bg: ["A0", "A1"], wgs: 40, red: "gemm_red_s4", redBg: ["R0", "R1"], redWgs: 5,
      sgm: { pipe: "gemm_sgm_q4_5120_s4", bg: ["S0", "S1"] }, r16: { pipe: "gemm_q4_5120_s4_r16", bg: ["H0", "H1"] } } };
  const xp = { pipe: "gemm_xpose", bg: "X", wgs: 3, r16: { pipe: "gemm_xpose_r16", bg: "XR", wgs: 3 } };
  const run = (pm, n) => { e.prefillMath = pm; seen.length = 0; e._dop({}, xp); e._dop({}, op, n); return seen.map((s) => s[0]); };
  assert(JSON.stringify(run("f32", 16)) === JSON.stringify(["gemm_xpose", "gemm_q4_5120_s4", "gemm_red_s4"]), "f32 default");
  assert(JSON.stringify(run("f16", 16)) === JSON.stringify(["gemm_xpose_r16", "gemm_q4_5120_s4_r16", "gemm_red_s4"]), "f16");
  assert(JSON.stringify(run("sgmatrix", 16)) === JSON.stringify(["gemm_xpose", "gemm_sgm_q4_5120_s4", "gemm_red_s4"]), "sgmatrix");
  assert(seen[1][2] === 40, "same grid as the f32 GEMM");
  for (const pm of PREFILL_MATH) assert(JSON.stringify(run(pm, 4).slice(1)) === JSON.stringify(["g4"]), `narrow passes stay on the GEMV (${pm})`);
  // an op without the variant (shape not tiled) falls back to the f32 GEMM
  const op2 = { ...op, gemm: { ...op.gemm, sgm: undefined } };
  e.prefillMath = "sgmatrix"; seen.length = 0; e._dop({}, op2, 16);
  assert(seen[0][0] === "gemm_q4_5120_s4", "fallback per op");
  e.prefillMath = "f32";
  assert(!("pm" in e.stateSignature()), "default signature unchanged");
  e.prefillMath = "f16";
  assert(e.stateSignature().pm === "f16", "f16 states are marked");
  e._pmAvail.f16 = false;
  assert(!("pm" in e.stateSignature()), "a mode that was not built runs f32 and is not marked");
});

Deno.test("prefillMathFeatures", () => {
  const ad = { features: new Set(["shader-f16", "timestamp-query"]) };
  assert(JSON.stringify(prefillMathFeatures(ad, "sgmatrix")) === JSON.stringify(["shader-f16"]), "only what the adapter has");
  assert(prefillMathFeatures(ad, "f16").length === 0 && prefillMathFeatures(ad, "f32").length === 0, "ALU modes need nothing");
});
