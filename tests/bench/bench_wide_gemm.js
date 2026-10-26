// Prefill-profile experiment (docs/research/prefill-profile-2026-09.md, candidate B): how fast can a
// classic shared-memory tiled Q4_0 GEMM go on this GPU when the prefill ubatch is wide (N = 32..512
// token columns) instead of 16? Standalone, random weights, checked against a naive kernel.
// Weight layout = the engine's repacked Q4_0 (per row: nb blocks of vec4<u32> nibbles, f16 scales
// in pairs), activations column-major xT[k][N], output y[col][row].
//   deno run --unstable-webgpu --allow-read --allow-env tests/bench/bench_wide_gemm.js
const ad = await navigator.gpu.requestAdapter();
const dev = await ad.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: ad.limits.maxComputeWorkgroupStorageSize } });
const rng = (n) => Array.from({ length: n }, (_, i) => i);

// BM x BN tile, 256 threads, each TM x TN outputs; K step = 32 (one Q4_0 block) x KB blocks
function tiled({ BM, BN, TM, TN, KB = 2 }) {
  const T = 256; if ((BM / TM) * (BN / TN) !== T) throw new Error("tile/thread mismatch");
  const KS = 32 * KB, TX = BN / TN;
  const units = BM * KB;               // (row quad, block, word): 8 k values x 4 rows, stored as vec4 over rows
  const wPer = Math.ceil(units / T);
  const xPer = KS * BN / 4 / T;        // vec4 of X per thread
  if (xPer % 1) throw new Error("stage split");
  return `
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sc: array<u32>;
@group(0) @binding(2) var<storage, read> xT: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { M: u32, K: u32, N: u32, p: u32 };
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> Ws: array<vec4<f32>, ${KS * BM / 4}>;   // [k][BM/4]
var<workgroup> Xs: array<vec4<f32>, ${KS * BN / 4}>;   // [k][BN/4]
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let tx = t % ${TX}u; let ty = t / ${TX}u;
  let row0 = wg.x * ${BM}u; let col0 = wg.y * ${BN}u; let nb = u.K / 32u;
  ${rng(TM).map((r) => `var a${r} = array<f32, ${TN}>(${rng(TN).map(() => "0.0").join(",")});`).join(" ")}
  for (var k0: u32 = 0u; k0 < u.K; k0 += ${KS}u) {
    workgroupBarrier();
    ${rng(wPer).map((j) => `if (t + ${j * T}u < ${units}u) {
      let li = t + ${j * T}u; let rq = li % ${BM / 4}u; let rest = li / ${BM / 4}u; let jj = rest % 4u; let bb = rest / 4u;
      let blk = k0 / 32u + bb;
      ${rng(4).map((r) => `let g${r} = (row0 + rq * 4u + ${r}u) * nb + blk; let w${r} = qs[g${r}][jj]; let s${r} = unpack2x16float(sc[g${r} >> 1u])[g${r} & 1u];
      let lo${r} = (vec4<f32>(unpack4xU8(w${r} & 0x0F0F0F0Fu)) - 8.0) * s${r}; let hi${r} = (vec4<f32>(unpack4xU8((w${r} >> 4u) & 0x0F0F0F0Fu)) - 8.0) * s${r};`).join("\n      ")}
      ${rng(4).map((i) => `Ws[(bb * 32u + 4u * jj + ${i}u) * ${BM / 4}u + rq] = vec4<f32>(lo0[${i}], lo1[${i}], lo2[${i}], lo3[${i}]); Ws[(bb * 32u + 16u + 4u * jj + ${i}u) * ${BM / 4}u + rq] = vec4<f32>(hi0[${i}], hi1[${i}], hi2[${i}], hi3[${i}]);`).join("\n      ")}
    }`).join("\n    ")}
    ${rng(xPer).map((j) => `{ let li = t + ${j * T}u; let kk = li / ${BN / 4}u; let cc = li % ${BN / 4}u; Xs[li] = xT[(k0 + kk) * (u.N / 4u) + col0 / 4u + cc]; }`).join("\n    ")}
    workgroupBarrier();
    let wb = ty * ${TM / 4}u; let xb = tx * ${TN / 4}u;
    ${rng(KS).map((k) => `{
      ${rng(TM / 4).map((r4) => `let w${r4} = Ws[${k * BM / 4}u + wb + ${r4}u];`).join(" ")}
      ${rng(TN / 4).map((c4) => `let x${c4} = Xs[${k * BN / 4}u + xb + ${c4}u];`).join(" ")}
      ${rng(TM).map((r) => rng(TN).map((c) => `a${r}[${c}] += w${r >> 2}[${r & 3}] * x${c >> 2}[${c & 3}];`).join(" ")).join(" ")}
    }`).join("\n    ")}
  }
  ${rng(TM).map((r) => rng(TN).map((c) => `y[(col0 + tx * ${TN}u + ${c}u) * u.M + row0 + ty * ${TM}u + ${r}u] = a${r}[${c}];`).join(" ")).join("\n  ")}
}`;
}
// naive reference: one thread per output
const naive = `
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sc: array<u32>;
@group(0) @binding(2) var<storage, read> xT: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { M: u32, K: u32, N: u32, p: u32 };
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let row = g.x; let col = g.y; if (row >= u.M) { return; }
  let nb = u.K / 32u; var acc = 0.0;
  for (var b: u32 = 0u; b < nb; b++) {
    let gi = row * nb + b; let w = qs[gi]; let s = unpack2x16float(sc[gi >> 1u])[gi & 1u];
    for (var j: u32 = 0u; j < 4u; j++) { for (var i: u32 = 0u; i < 4u; i++) {
      let byte = (w[j] >> (8u * i)) & 0xFFu;
      let k = b * 32u + 4u * j + i;
      acc += (f32(byte & 0xFu) - 8.0) * s * xT[(k * u.N + col) / 4u][(k * u.N + col) % 4u];
      acc += (f32(byte >> 4u) - 8.0) * s * xT[((k + 16u) * u.N + col) / 4u][((k + 16u) * u.N + col) % 4u];
    } }
  }
  y[col * u.M + row] = acc;
}`;

dev.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message?.slice(0, 400)));
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
async function run(M, K, N, cfg, reps = 10) {
  const nb = K / 32;
  const qsA = new Uint32Array(M * nb * 4); for (let i = 0; i < qsA.length; i++) qsA[i] = (Math.random() * 2 ** 32) >>> 0;
  const scA = new Uint16Array(M * nb); for (let i = 0; i < scA.length; i++) scA[i] = 0x2000 + ((Math.random() * 512) | 0);   // ~0.008..0.016
  const xA = new Float32Array(K * N); for (let i = 0; i < xA.length; i++) xA[i] = Math.random() - 0.5;
  const mk = (a) => { const b = dev.createBuffer({ size: Math.ceil(a.byteLength / 4) * 4, usage: S }); dev.queue.writeBuffer(b, 0, a); return b; };
  const qs = mk(qsA), sc = mk(scA), xT = mk(xA);
  const y = dev.createBuffer({ size: M * N * 4, usage: S }), yr = dev.createBuffer({ size: M * N * 4, usage: S });
  const ub = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(ub, 0, new Uint32Array([M, K, N, 0]));
  const pipe = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
  const bg = (p, out) => dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [qs, sc, xT, out, ub].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const pt = pipe(tiled(cfg)), pn = pipe(naive);
  const qset = dev.createQuerySet({ type: "timestamp", count: 2 * reps });
  const res = dev.createBuffer({ size: 16 * reps, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const rd = dev.createBuffer({ size: 16 * reps, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  { const p = enc.beginComputePass(); p.setPipeline(pn); p.setBindGroup(0, bg(pn, yr)); p.dispatchWorkgroups(Math.ceil(M / 64), N); p.end(); }
  const bgt = bg(pt, y);
  for (let r = 0; r < reps; r++) {
    const p = enc.beginComputePass({ timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 } });
    p.setPipeline(pt); p.setBindGroup(0, bgt); p.dispatchWorkgroups(M / cfg.BM, N / cfg.BN); p.end();
  }
  enc.resolveQuerySet(qset, 0, 2 * reps, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, 16 * reps);
  const ry = dev.createBuffer({ size: M * N * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyBufferToBuffer(y, 0, ry, 0, M * N * 4); enc.copyBufferToBuffer(yr, 0, ry, M * N * 4, M * N * 4);
  dev.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
  await ry.mapAsync(GPUMapMode.READ); const o = new Float32Array(ry.getMappedRange().slice(0)); ry.unmap();
  let md = 0, mx = 0; for (let i = 0; i < M * N; i++) { md = Math.max(md, Math.abs(o[i] - o[M * N + i])); mx = Math.max(mx, Math.abs(o[M * N + i])); }
  const ts = rng(reps).map((r) => Number(t[2 * r + 1] - t[2 * r]) / 1e6).sort((a, b) => a - b);
  const ms = ts[Math.floor(reps / 2)];
  for (const b of [qs, sc, xT, y, yr, ub, res, rd, ry]) b.destroy(); qset.destroy();
  return { ms, tflops: 2 * M * K * N / ms / 1e9, gbs: M * K * 18 / 32 / ms / 1e6, relDiff: md / mx };
}
const shapes = (Deno.env.get("SHAPES") || "17408x5120,5120x17408,10240x5120,8192x2048,2048x4096").split(",").map((s) => s.split("x").map(Number));
const Ns = (Deno.env.get("NS") || "64,128,256,512").split(",").map(Number);
const cfgs = JSON.parse(Deno.env.get("CFGS") || JSON.stringify([{ BM: 64, BN: 64, TM: 4, TN: 4 }, { BM: 128, BN: 64, TM: 8, TN: 4 }, { BM: 64, BN: 128, TM: 4, TN: 8 }, { BM: 128, BN: 128, TM: 8, TN: 8, KB: 1 }]));
for (const [M, K] of shapes) for (const N of Ns) for (const cfg of cfgs) {
  if (N % cfg.BN || M % cfg.BM) continue;
  try {
    const r = await run(M, K, N, cfg);
    console.log(`${M}x${K} N=${String(N).padStart(3)} tile ${cfg.BM}x${cfg.BN} (${cfg.TM}x${cfg.TN}/thr): ${r.ms.toFixed(3)} ms  ${r.tflops.toFixed(2)} TFLOPS  ${(r.ms / N * 1000).toFixed(1)} µs/token  weights ${r.gbs.toFixed(0)} GB/s  relDiff ${r.relDiff.toExponential(1)}`);
  } catch (e) { console.log(`${M}x${K} N=${N} ${JSON.stringify(cfg)}: ${e.message}`); }
}
