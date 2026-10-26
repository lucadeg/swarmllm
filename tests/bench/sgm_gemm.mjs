// Run tests/bench/sgm_gemm.html in Chrome with the real GPU: the prefill GEMM kernels alone (no model),
// f32 vs f16-operand twin vs tensor-core subgroup-matrix GEMM, ms / TFLOPS / relDiff per 27B shape.
//   node tests/bench/sgm_gemm.mjs ['reps=20&q8=1&sgm={"TM":128,"KB":1,"PAD":8}']
// CHROME_BIN=<path>: a Chrome/Chromium build other than Playwright's bundled one (newer builds expose the extension).
// Uses the GPU: never run it while another job owns the GPU.
import { chromium } from "playwright"; import { spawn } from "node:child_process";
const root = new URL("../..", import.meta.url).pathname, EXTRA = process.argv[2] ? "?" + process.argv[2] : "";
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, "8792"], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
// --enable-unsafe-webgpu exposes Chrome's experimental WebGPU features (chromium-experimental-subgroup-matrix among them)
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
const b = await chromium.launch({ headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) }), p = await (await b.newContext()).newPage();
p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
await p.goto(`http://127.0.0.1:8792/tests/bench/sgm_gemm.html${EXTRA}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 10 * 60e3, polling: 1000 }).catch((e) => console.log("timeout", e.message));
await b.close(); srv.kill();
