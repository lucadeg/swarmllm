// GPU sampling kernels (topk_a / topk_b, engine/wgsl/qwen35.js) against their CPU model
// (engine/topk.js, itself checked against a plain sort in tests/unit/topk_test.js), bit for bit:
// random logits at the real vocab sizes, ties, NaN / +-Inf, n not a multiple of 4096, strided
// columns, k = 1 / 40 / 64; and k = 1 against the old single-workgroup argmax kernel.
// Runs on any WebGPU adapter; on a CPU Vulkan driver it needs no GPU at all:
//   CPU_ONLY=1 WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json deno run --allow-read --allow-env --unstable-webgpu tests/e2e/topk_kernel.mjs
// (QUICK=1: fewer cases, for slow software drivers)
import { WGSL } from "../../engine/wgsl/base.js";
import { WGSL2 } from "../../engine/wgsl/qwen35.js";
import { topkTwoStage } from "../../engine/topk.js";

const QUICK = Deno.env.get("QUICK") === "1";
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
const ai = adapter.info || {};
console.log("adapter:", ai.vendor, ai.architecture, ai.device, ai.description);
// CPU_ONLY=1: refuse to run on a real GPU (so it never competes with a GPU measurement)
if (Deno.env.get("CPU_ONLY") === "1" && !/llvmpipe|lavapipe|swiftshader|cpu/i.test([ai.vendor, ai.architecture, ai.device, ai.description].join(" "))) { console.log("not a CPU adapter; exiting"); Deno.exit(2); }
device.pushErrorScope("validation");
const mod = device.createShaderModule({ code: WGSL + WGSL2 });
const info = await mod.getCompilationInfo?.();
for (const m of info?.messages || []) if (m.type === "error") console.log(`WGSL ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
const mk = (entryPoint) => device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint } });
const pA = mk("topk_a"), pB = mk("topk_b"), pOld = mk("argmax");
const e0 = await device.popErrorScope();
if (e0) { console.log("pipeline error:", e0.message); Deno.exit(1); }

const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const buf = (size, usage = S) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
const uni = (a) => { const b = buf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(b, 0, new Uint32Array(a)); return b; };
async function read(b, bytes) {
  const st = buf(bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, Math.ceil(bytes / 4) * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const out = new Uint32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return out;
}
const empty = (pipe) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [] });
// the engine's dispatch (Qwen35Engine._dTopk): a over (nw, cols), b over (1, cols)
async function gpuTopk(X, n, stride, cols, k) {
  const nw = Math.ceil(n / 4096), R = 2 * k + 2;
  const x = buf(X.byteLength); device.queue.writeBuffer(x, 0, X);
  const part = buf(cols * nw * R * 4), out = buf(cols * R * 4);
  // group 0 is unused by the kernels ("auto" layout: an empty group), they bind at group 1
  const e2 = device.createCommandEncoder(), q = e2.beginComputePass();
  q.setPipeline(pA);
  q.setBindGroup(0, empty(pA));
  q.setBindGroup(1, device.createBindGroup({ layout: pA.getBindGroupLayout(1), entries: [x, part, uni([n, stride, k, nw])].map((b, i) => ({ binding: i, resource: { buffer: b } })) }));
  q.dispatchWorkgroups(nw, cols, 1);
  q.setPipeline(pB);
  q.setBindGroup(0, empty(pB));
  q.setBindGroup(1, device.createBindGroup({ layout: pB.getBindGroupLayout(1), entries: [part, out, uni([nw, k, 0, 0])].map((b, i) => ({ binding: i, resource: { buffer: b } })) }));
  q.dispatchWorkgroups(1, cols, 1);
  q.end();
  device.queue.submit([e2.finish()]);
  const r = await read(out, cols * R * 4);
  x.destroy(); part.destroy(); out.destroy();
  return r;
}
async function gpuOldArgmax(X, n) {
  const x = buf(X.byteLength); device.queue.writeBuffer(x, 0, X);
  const out = buf(16);
  const e = device.createCommandEncoder(), q = e.beginComputePass();
  q.setPipeline(pOld);
  q.setBindGroup(0, empty(pOld));
  q.setBindGroup(1, device.createBindGroup({ layout: pOld.getBindGroupLayout(1), entries: [x, out, uni([n, 0, 0, 0])].map((b, i) => ({ binding: i, resource: { buffer: b } })) }));
  q.dispatchWorkgroups(1); q.end(); device.queue.submit([e.finish()]);
  const r = await read(out, 16); x.destroy(); out.destroy(); return r;
}

let seed = 777;
const rnd = () => { let t = (seed = (seed + 0x6d2b79f5) | 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };
const gen = (n, kind) => Float32Array.from({ length: n }, () => kind === "ties" ? Math.floor(rnd() * 5) : (rnd() * 2 - 1) * 8);
let fail = 0, cases = 0;
const cmp = (a, b, what) => { cases++; const same = a.length === b.length && a.every((v, i) => v === b[i]); if (!same) { fail++; console.log(`MISMATCH ${what}\n  gpu ${[...a.slice(0, 12)]}\n  cpu ${[...b.slice(0, 12)]}`); } };

const sizes = QUICK ? [5000, 65536] : [248320, 65536, 12289, 4097, 300];
const ks = QUICK ? [1, 40] : [1, 20, 40, 64];
for (const n of sizes) for (const kind of ["rand", "ties", "nonfinite"]) for (const cols of [1, 3]) {
  const stride = cols > 1 ? Math.ceil(n * 4 / 256) * 64 : 0;
  const X = new Float32Array(cols > 1 ? stride * cols : n);
  for (let c = 0; c < cols; c++) {
    const v = gen(n, kind);
    if (kind === "nonfinite") for (let j = 0; j < 6; j++) v[Math.floor(rnd() * n)] = [NaN, -Infinity, Infinity][j % 3];
    X.set(v, c * stride);
  }
  for (const k of ks) {
    const g = await gpuTopk(X, n, stride, cols, k), R = 2 * k + 2;
    for (let c = 0; c < cols; c++) cmp(g.subarray(c * R, (c + 1) * R), topkTwoStage(X, { n, stride, col: c, k }), `n ${n} ${kind} cols ${cols} col ${c} k ${k}`);
  }
  if (cols === 1 && kind !== "nonfinite") {   // the old kernel has no NaN rule to compare
    const o = await gpuOldArgmax(X, n), w = topkTwoStage(X, { n, k: 1 });
    cmp(o.subarray(0, 2), w.subarray(0, 2), `old argmax n ${n} ${kind}`);
  }
}
// a column with no finite value: argmax 0 (the host greedy), not the old kernel's 0xffffffff
{ const X = new Float32Array(9000).fill(-Infinity); cmp(await gpuTopk(X, 9000, 0, 1, 1), topkTwoStage(X, { n: 9000, k: 1 }), "all -Inf"); }
console.log(fail ? `TOPK KERNEL FAIL (${fail}/${cases})` : `TOPK KERNEL PASS ✓ (${cases} cases)`);
if (fail) Deno.exit(1);
