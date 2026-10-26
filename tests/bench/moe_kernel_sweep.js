// Expert GEMV layout sweep at the Qwen3.6-35B-A3B shape: synthetic Q4_0 / Q8_0 experts (256 x [512][2048]
// gate and up, 256 x [2048][512] down), 8 random distinct experts per launch so the reads come from DRAM,
// one launch per compute pass with timestamp queries. Prints the median µs per launch and the effective
// GB/s of every layout, fastest first, and checks each layout's output against the legacy layout.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench/moe_kernel_sweep.js
//   env: KIND=gu|dn|both (both), GRID=small|full (small), ITERS (200), DNFMT=q4|q8 (q4), ONLY='<moeKernel JSON>'
import { moeWGSL, moeKernelConfig } from "../../engine/wgsl/moe.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const dim = 2048, inter = 512, nExp = 256, K = 8, ITERS = +env("ITERS", 200), KIND = env("KIND", "both"), DNFMT = env("DNFMT", "q4");
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"],
  requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const U = GPUBufferUsage;
let seed = 99;
const fill = (bytes, scale) => {   // random words, or a constant f16 pair (scales) so no value is NaN / denormal
  const b = device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const CH = 1 << 24;
  for (let off = 0; off < bytes; off += CH) {
    const n = Math.min(CH, bytes - off) / 4, a = new Uint32Array(n);
    if (scale) a.fill(0x2c002c00); else for (let i = 0; i < n; i++) a[i] = seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
    device.queue.writeBuffer(b, off, a);
  }
  return b;
};
const qBytes = (fmt, rows, dIn) => rows * (dIn / 32) * (fmt === "q4" ? 16 : 32), sBytes = (rows, dIn) => rows * (dIn / 32) * 2;
const W = {
  gq: fill(qBytes("q4", nExp * inter, dim)), gs: fill(sBytes(nExp * inter, dim), true),
  uq: fill(qBytes("q4", nExp * inter, dim)), us: fill(sBytes(nExp * inter, dim), true),
  dq: fill(qBytes(DNFMT, nExp * dim, inter)), ds: fill(sBytes(nExp * dim, inter), true),
};
const x = device.createBuffer({ size: dim * 4, usage: U.STORAGE | U.COPY_DST });
device.queue.writeBuffer(x, 0, Float32Array.from({ length: dim }, (_, i) => Math.sin(i) * 0.5));
const h = device.createBuffer({ size: K * inter * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
device.queue.writeBuffer(h, 0, Float32Array.from({ length: K * inter }, (_, i) => Math.cos(i) * 0.5));
const y = device.createBuffer({ size: K * dim * 4, usage: U.STORAGE | U.COPY_SRC });
const hOut = device.createBuffer({ size: K * inter * 4, usage: U.STORAGE | U.COPY_SRC });
// NSEL routing rows of K distinct experts, each in its own 256-byte slot (storage offset alignment)
const NSEL = 64, sel = device.createBuffer({ size: NSEL * 256, usage: U.STORAGE | U.COPY_DST });
{ const a = new Uint32Array(NSEL * 64); for (let s = 0; s < NSEL; s++) { const pick = new Set(); while (pick.size < K) pick.add((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) % nExp); [...pick].forEach((e, k) => a[s * 64 + k] = e); } device.queue.writeBuffer(sel, 0, a); }
const uni = (a) => { const b = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(a)); return b; };
const uGu = uni([inter, dim, K, nExp, dim, inter, 0, 0]), uDn = uni([dim, inter, K, nExp, inter, dim, 0, 0]);
const MAXQ = 2 * ITERS, qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
const res = device.createBuffer({ size: MAXQ * 8, usage: U.QUERY_RESOLVE | U.COPY_SRC });
const rd = device.createBuffer({ size: MAXQ * 8, usage: U.COPY_DST | U.MAP_READ });
const readF = async (b, n) => {
  const st = device.createBuffer({ size: n * 4, usage: U.MAP_READ | U.COPY_DST });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, n * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const r = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return r;
};

async function time(kind, opt) {
  const cfg = moeKernelConfig(opt, { dim, inter }), c = cfg[kind];
  const fmt = kind === "gu" ? "q4" : DNFMT, name = `moe_${kind}_${fmt}`;
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: moeWGSL(cfg) });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: name } });
  const err = await device.popErrorScope();
  if (err) throw new Error(`${name} ${JSON.stringify(c)}: ${err.message}`);
  const bgs = Array.from({ length: NSEL }, (_, s) => {
    const sb = { buffer: sel, offset: s * 256, size: K * 4 };
    const bufs = kind === "gu" ? [W.gq, W.gs, W.uq, W.us, x, hOut, sb, uGu] : [W.dq, W.ds, h, y, sb, uDn];
    return device.createBindGroup({ layout: pipe.getBindGroupLayout(1), entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })) });
  });
  const gx = Math.ceil((kind === "gu" ? inter : dim) / c.rows);
  const run = (n, ts) => {
    const enc = device.createCommandEncoder();
    for (let i = 0; i < n; i++) {
      const p = enc.beginComputePass(ts ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } } : undefined);
      p.setPipeline(pipe); p.setBindGroup(1, bgs[i % NSEL]); p.dispatchWorkgroups(gx, K); p.end();
    }
    if (ts) { enc.resolveQuerySet(qs, 0, 2 * n, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, 2 * n * 8); }
    device.queue.submit([enc.finish()]);
  };
  run(20, false);   // warm the clocks
  run(ITERS, true);
  await rd.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(rd.getMappedRange().slice(0, 2 * ITERS * 8)); rd.unmap();
  const us = Array.from({ length: ITERS }, (_, i) => Number(t[2 * i + 1] - t[2 * i]) / 1e3).sort((a, b) => a - b);
  // one launch on routing row 0 for the output check
  { const enc = device.createCommandEncoder(), p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(1, bgs[0]); p.dispatchWorkgroups(gx, K); p.end(); device.queue.submit([enc.finish()]); }
  const out = await readF(kind === "gu" ? hOut : y, kind === "gu" ? K * inter : K * dim);
  const bytes = kind === "gu" ? K * 2 * qBytes("q4", inter, dim) * 18 / 16 : K * qBytes(fmt, dim, inter) * (fmt === "q4" ? 18 / 16 : 34 / 32);
  return { kind, c, med: us[ITERS >> 1], p10: us[Math.floor(ITERS / 10)], gbs: bytes / (us[ITERS >> 1] * 1e3), out };
}

const grid = (kind) => {
  const out = [], full = env("GRID", "small") === "full";
  for (const WG of full ? [64, 128, 256] : [128, 256]) for (const TPR of kind === "gu" ? (full ? [4, 8, 16, 32, 64] : [8, 16, 32]) : (full ? [1, 2, 4, 8, 16] : [2, 4, 8]))
    for (const R of full ? [1, 2, 4] : [1, 2, 4]) for (const U of kind === "gu" ? (full ? [1, 2, 4] : [1, 2, 4]) : (full ? [1, 2, 4, 8] : [1, 2, 4]))
      for (const xsh of full ? [true, false] : [true]) {
        if (TPR > WG || R > TPR) continue;
        const LANES = TPR, nb = (kind === "gu" ? dim : inter) / 32;
        if (LANES * U > nb * 2) continue;   // all of it would be tail
        out.push({ [kind]: { WG, TPR, R, U, wide: true, xsh } });
      }
  return out;
};
const ONLY = env("ONLY", "");
for (const kind of KIND === "both" ? ["gu", "dn"] : [KIND]) {
  const base = await time(kind, "legacy");
  const rows = [{ name: "legacy", ...base }, { name: "default", ...(await time(kind, undefined)) }];
  for (const opt of ONLY ? [JSON.parse(ONLY)] : grid(kind)) {
    try { rows.push({ name: JSON.stringify(opt[kind] || opt), ...(await time(kind, opt)) }); } catch (e) { console.log("skip", e.message.slice(0, 200)); }
  }
  let refMax = 0; for (const v of base.out) refMax = Math.max(refMax, Math.abs(v));
  console.log(`\n${kind === "gu" ? "moe_gu_q4" : "moe_dn_" + DNFMT}: ${ITERS} launches each, K=${K}, median µs (p10), effective GB/s, max |diff| vs legacy / max |legacy|`);
  for (const r of rows.sort((a, b) => a.med - b.med)) {
    let d = 0; for (let i = 0; i < r.out.length; i++) d = Math.max(d, Math.abs(r.out[i] - base.out[i]));
    const bad = !(d / refMax < 1e-4);
    console.log(`  ${r.med.toFixed(1).padStart(7)} (${r.p10.toFixed(1).padStart(6)})  ${r.gbs.toFixed(0).padStart(4)} GB/s  ${(d / refMax).toExponential(1)}${bad ? " MISMATCH" : ""}  ${r.name}  rows/WG ${r.c.rows}`);
  }
}
