// The load-time GPU checks every room device runs before it loads a model (engine/selftest.js).
// They poke at DenseEngine internals, so an engine change that moves a buffer must keep them passing.
import { gpuSelfTest, kernelMicroTests } from "../engine/selftest.js";

const dev = async () => (await navigator.gpu.requestAdapter()).requestDevice();
const st = await gpuSelfTest(await dev());
console.log("self-test:", st.detail);
const mt = await kernelMicroTests(await dev());
console.log("kernel micro-tests:", mt.detail);
const ok = st.ok && mt.ok;
console.log(ok ? "SELFTEST PASS ✓" : "SELFTEST FAIL: " + (mt.firstFail || st.detail));
Deno.exit(ok ? 0 : 1);
