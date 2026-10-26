// The 27B GPU suite (tests/run.sh q38) in one process: the model is read, converted and uploaded
// to the GPU once, and every check builds its engines (whole model, a few layers, or split
// host/worker shards) over views of that one uploaded set (tests/load_model.js sharedQ38Context).
// Each check is the test file's own exported run(): same assertions, same goldens, same env knobs.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write run_q38_once.js
//   ONLY=test_mtp,test_ctx ...   runs a subset (names as in run.sh)
// The per-file tests still run standalone (tests/run.sh q38) and remain the reference.
import { sharedQ38Context } from "./load_model.js";

const SUITE = ["test_q38", "test_batch_q38", "test_mtp", "test_b4", "test_twins", "test_gemm", "test_q38_split", "test_mtp_split", "test_ctx"];
const only = (Deno.env.get("ONLY") || "").split(",").map((s) => s.trim().replace(/\.js$/, "")).filter(Boolean);
const list = only.length ? SUITE.filter((n) => only.includes(n)) : SUITE;
if (only.length && list.length !== only.length) { console.error("unknown in ONLY:", only.filter((n) => !SUITE.includes(n)).join(",")); Deno.exit(2); }

const T0 = performance.now();
const ctx = await sharedQ38Context();
const tLoad = performance.now() - T0;
const results = [];
for (const name of list) {
  console.log(`=== ${name}`);
  const t0 = performance.now(), e0 = ctx.errors.count;
  let ok = false;
  try {
    const { run } = await import(`./${name}.js`);
    ok = await run(ctx);
  } catch (err) {
    console.error(`${name} threw:`, err?.stack || err);
  }
  const newErrors = ctx.errors.count - e0;
  if (newErrors) console.log(`(${newErrors} uncaptured GPU errors during ${name})`);
  results.push({ name, ok: ok === true, s: (performance.now() - t0) / 1000 });
}
console.log("\n=== summary (one process, shared weights)");
console.log(`model load + upload: ${(tLoad / 1000).toFixed(1)} s`);
for (const r of results) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name.padEnd(16)} ${r.s.toFixed(1).padStart(6)} s`);
console.log(`total ${((performance.now() - T0) / 1000).toFixed(1)} s`);
const failed = results.filter((r) => !r.ok).length;
console.log(failed ? `Q38 ONCE: ${failed} FAILED` : "Q38 ONCE PASS ✓");
Deno.exit(failed ? 1 : 0);
