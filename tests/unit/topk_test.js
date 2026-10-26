// GPU sampling (exp/gpu-sample): the two-stage argmax / top-k index math of topk_a / topk_b
// (engine/wgsl/qwen35.js), modelled on the CPU by engine/topk.js thread by thread, against a plain
// sort and against the host samplers (room/sampling.js). Ties, NaN, +-Inf, n not a multiple of
// 4096, the 65536-row draft head, column strides, the result layout and aiSampleTop == aiSample.
import { topkTwoStage, topkNaive, readCands, topkK, NONE, TOPK_MAX } from "../../engine/topk.js";
import { aiSample, aiSampleTop, greedy, pickSampler } from "../../room/sampling.js";
import { argmax } from "../../engine/sampling.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
let seed = 12345;
const rnd = () => {   // mulberry32: distinct values, so only the tie tests have ties
  let t = (seed = (seed + 0x6d2b79f5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
};
const randn = (n, scale = 4) => Float32Array.from({ length: n }, () => (rnd() * 2 - 1) * scale);
const same = (x, o, m) => {
  const a = topkTwoStage(x, o), b = topkNaive(x, o);
  eq([...a], [...b], `${m} (n ${o.n}, k ${o.k}, col ${o.col || 0})`);
  return a;
};

Deno.test("two-stage == sort: random logits at the real vocab sizes", () => {
  for (const n of [248320, 151936, 65536]) for (const k of [1, 20, 40, 64]) same(randn(n), { n, k }, "random");
});

Deno.test("two-stage == sort: n not a multiple of 4096, tiny n, n < k", () => {
  for (const n of [1, 3, 63, 255, 256, 4095, 4096, 4097, 5000, 8191, 12289, 65537]) for (const k of [1, 7, 64]) same(randn(n), { n, k }, "odd n");
});

Deno.test("ties: lowest index first, value descending, across workgroup boundaries", () => {
  // few distinct values: every pick is a tie broken by index, including ties that span slices
  for (const n of [9000, 248320]) for (const k of [1, 5, 40, 64]) {
    const x = Float32Array.from({ length: n }, () => Math.floor(rnd() * 6));
    const r = same(x, { n, k }, "ties");
    if (k === 1) { eq(r[0], greedy(x), "greedy tie rule"); eq(r[0], argmax(x), "engine argmax tie rule"); }
  }
  // the max repeated only in the last slice and the first
  const x = new Float32Array(20000).fill(1); x[19999] = 9; x[4096 * 3 + 5] = 9; x[7] = 9;
  eq(topkTwoStage(x, { n: 20000, k: 3 })[0], 7);
  eq([...topkTwoStage(x, { n: 20000, k: 3 })].filter((_, i) => i % 2 === 0 && i < 6), [7, 4096 * 3 + 5, 19999]);
});

Deno.test("NaN / -Inf never picked, +Inf picked, all counted as bad", () => {
  const n = 10000, x = randn(n);
  x[3] = NaN; x[4097] = NaN; x[9999] = -Infinity; x[500] = -Infinity; x[8000] = Infinity;
  for (const k of [1, 64]) {
    const r = same(x, { n, k }, "non-finite");
    eq(r[2 * k], 5, "bad count");
    eq(r[0], 8000, "+Inf is the max");
    const ids = [...r].filter((_, i) => i % 2 === 0 && i < 2 * k);
    ok(!ids.includes(3) && !ids.includes(4097) && !ids.includes(9999) && !ids.includes(500), "NaN / -Inf picked");
  }
  // NaN in slot 0 does not stop the scan (the host greedy's strict > skips it too)
  const y = Float32Array.from([NaN, 1, 2, 2, NaN]);
  eq(topkTwoStage(y, { n: 5, k: 1 })[0], 2);
  eq(greedy(y), 2);
});

Deno.test("columns with no finite value: argmax falls back to 0, top-k entries are missing", () => {
  const x = new Float32Array(5000).fill(-Infinity); x[10] = NaN;
  const r = same(x, { n: 5000, k: 4 }, "all -Inf");
  eq(r[0], 0); eq(r[2], NONE); eq(r[8], 5000, "bad count (every value non-finite)");
  eq(greedy(x), 0, "host greedy on all -Inf");
  // fewer finite values than k
  const z = new Float32Array(300).fill(-Infinity); z[7] = 1; z[299] = 2;
  const c = readCands(topkTwoStage(z, { n: 300, k: 8 }), 0, 8);
  eq([...c.ids], [299, 7]); eq([...c.vals], [2, 1]); eq(c.bad, 298);
});

Deno.test("column stride: B.logits layout (columns 256-byte aligned)", () => {
  const n = 248320, stride = Math.ceil(n * 4 / 256) * 256 / 4, cols = 3;
  const X = new Float32Array(stride * cols).fill(NaN);   // padding is never read
  for (let c = 0; c < cols; c++) X.set(randn(n), c * stride);
  for (let c = 0; c < cols; c++) {
    const r = same(X, { n, stride, col: c, k: 40 }, "strided");
    eq(r[80], 0, "padding read as a logit");
    eq(r[0], greedy(X.subarray(c * stride, c * stride + n)));
  }
});

Deno.test("draft head: argmax over the first 65536 rows only", () => {
  const n = 248320, x = randn(n); x[200000] = 100;   // above draftVocab: must not be seen
  const r = topkTwoStage(x, { n: 65536, k: 1 });
  eq(r[0], greedy(x.subarray(0, 65536)));
  ok(r[0] !== 200000);
});

Deno.test("readCands and topkK", () => {
  const u = new Uint32Array([5, 0x40000000, 9, 0x3f800000, NONE, 0, 3, 0]);   // k = 3: 2.0, 1.0, missing; bad 3
  const c = readCands(u, 0, 3);
  eq([...c.ids], [5, 9]); eq([...c.vals], [2, 1]); eq(c.bad, 3);
  eq(topkK({ kind: "greedy" }), 1); eq(topkK({ kind: "topk", k: 40 }), 40); eq(topkK({ kind: "topk", k: 500 }), TOPK_MAX); eq(topkK(null), 1);
});

Deno.test("aiSampleTop over the GPU's top-k == aiSample over the logits (same Math.random)", () => {
  const real = Math.random;
  try {
    for (const [temp, k] of [[0.8, 40], [0.4, 20], [1.5, 64]]) for (let trial = 0; trial < 6; trial++) {
      const x = randn(248320, 6);
      const c = readCands(topkTwoStage(x, { n: x.length, k }), 0, k);
      for (const r of [0, 0.01, 0.3, 0.5, 0.77, 0.999]) {
        Math.random = () => r;
        eq(aiSampleTop(c, temp), aiSample(x, temp, k), `temp ${temp} k ${k} r ${r}`);
      }
    }
  } finally { Math.random = real; }
});

Deno.test("pickSampler: .gpu descriptor, reads logits or candidates", () => {
  const x = randn(5000);
  const g = pickSampler("exact");
  eq(g.gpu, { kind: "greedy" });
  const c = readCands(topkTwoStage(x, { n: 5000, k: 1 }), 0, 1);
  eq(g(x), g(c)); eq(g(c), greedy(x));
  const s = pickSampler("creative");
  eq(s.gpu, { kind: "topk", k: 40, temp: 0.8 });
  const real = Math.random;
  try {
    Math.random = () => 0.42;
    eq(s(x), s(readCands(topkTwoStage(x, { n: 5000, k: 40 }), 0, 40)));
  } finally { Math.random = real; }
  // a masking wrapper is a new function: no .gpu, so the engine hands it full logits
  const wrapped = (lg) => g(lg);
  ok(!wrapped.gpu);
});
