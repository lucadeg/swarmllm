// A small Code mode eval for the small JSON-style model (Qwen3 1.7B) on one local WebGPU engine,
// in Deno: the real agent loop, the real tools and PreviewServer over an in-memory project, the
// room's Code mode settings (JSON tool calls, greedy, 8k context, the room's token budget).
// No browser: a stand-in preview frame reports each served rev as loaded, with a syntax error
// (as the page would log it) for any script that does not parse. Each task's check reads the
// final files. pass = reason "done" + a serve that succeeded + the check.
//
//   deno run --unstable-webgpu -A tests/eval/small_model.js [--model models/qwen17] [--tasks a,b]
//        [--repo <dir>]   (run another checkout's harness with this driver, e.g. a baseline)
//        [--ctx 8192] [--steps 30] [--verbose]
const argv = Deno.args;
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const HERE = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const R = arg("repo", HERE).replace(/\/$/, "");
const M = arg("model", HERE + "/models/qwen17").replace(/\/$/, "") + "/";
const CTX = +arg("ctx", 8192), STEPS = +arg("steps", 30), VERBOSE = argv.includes("--verbose");
const TASK_MS = +arg("task-ms", 420000);

const { makeTokenizer, DenseEngine } = await import(R + "/engine/engine.js");
const { parseGGUFHeader, ggufWeights } = await import(R + "/engine/gguf.js");
const { engineModel } = await import(R + "/harness/engine-model.js");
const { Agent } = await import(R + "/harness/agent.js");
const { codingTools } = await import(R + "/harness/codetools.js");
const { MemoryWorkspace } = await import(R + "/harness/workspace.js");
const { CODE_SYSTEM } = await import(R + "/harness/code-prompt.js");
const { previewTools } = await import(R + "/harness/preview-tools.js");
const { PreviewServer } = await import(R + "/harness/preview.js");
const { detectStyle } = await import(R + "/harness/tools.js");
const { runJsTool } = await import(R + "/harness/run-js.js");

// ---- tasks: { id, prompt, files?, check(files) -> "" (ok) | why it failed }
const html = (f) => f.get("index.html") || "";
const allText = (f) => [...f.values()].join("\n");
const need = (cond, why) => (cond ? "" : why);
const COUNTER = `<!doctype html>
<html>
<head><title>Counter</title></head>
<body>
  <h1>Clicks: <span id="n">0</span></h1>
  <button id="b">Click me</button>
  <script src="app.js"></script>
</body>
</html>
`;
const TASKS = [
  { id: "hello", prompt: "make a website that says hello",
    check: (f) => need(/hello/i.test(html(f)), "index.html does not say hello") },
  { id: "basic", prompt: "build a website that says hello verry basic please",
    check: (f) => need(/hello/i.test(html(f)), "index.html does not say hello") },
  { id: "counter", prompt: "make a page with a button that counts how many times it was clicked",
    check: (f) => need(/<button/i.test(allText(f)), "no button") || need(/click/i.test(allText(f)) && /(\+\+|\+=\s*1|\+\s*1)/.test(allText(f)), "no click counting") },
  { id: "todo", prompt: "make a simple todo list app",
    check: (f) => need(/<input/i.test(allText(f)), "no input") || need(/(appendChild|append\(|insertAdjacent|innerHTML\s*\+?=|createElement)/.test(allText(f)), "never adds items") },
  { id: "add-feature", prompt: "add a button to my page that shows an alert saying hi when clicked",
    files: { "index.html": "<!doctype html>\n<html>\n<head><title>My page</title></head>\n<body>\n  <h1>Welcome</h1>\n  <p>This is my page.</p>\n</body>\n</html>\n" },
    check: (f) => need(/<button/i.test(allText(f)), "no button") || need(/alert\(\s*['"`]hi/i.test(allText(f)), "no alert('hi')") || need(/Welcome/.test(html(f)), "the old heading is gone") },
  { id: "fix-bug", prompt: "the counter goes down instead of up when I click the button, fix it",
    files: { "index.html": COUNTER, "app.js": "let count = 0;\nconst n = document.getElementById(\"n\");\ndocument.getElementById(\"b\").addEventListener(\"click\", () => {\n  count--;\n  n.textContent = count;\n});\n" },
    check: (f) => need(!/count--|count\s*-=\s*1/.test(allText(f)), "still decrements") || need(/count\+\+|count\s*\+=\s*1|count\s*=\s*count\s*\+\s*1/.test(allText(f)), "does not increment") },
  { id: "snake", prompt: "make a simple snake game",
    check: (f) => need(/<canvas|getContext|class=|<div/i.test(allText(f)), "nothing to draw on") || need(/keydown|onkey|addEventListener\(\s*['"]key/i.test(allText(f)), "no keyboard input") },
  { id: "colors", prompt: "change the background color to dark blue and the text to white",
    files: { "index.html": "<!doctype html>\n<html>\n<head>\n  <title>Colors</title>\n  <link rel=\"stylesheet\" href=\"style.css\">\n</head>\n<body>\n  <h1>Hello</h1>\n  <p>Some text.</p>\n</body>\n</html>\n",
      "style.css": "body {\n  background: #ffffff;\n  color: #222222;\n  font-family: sans-serif;\n}\n" },
    check: (f) => need(/(navy|darkblue|dark blue|midnightblue|#0{2}[0-9a-f]{2}[4-9a-f][0-9a-f]|#00008b|#003|#002|#001|#1[0-9a-f]{2}[3-9a-f][0-9a-f]{2}|rgb\(\s*0\s*,\s*0\s*,\s*1\d\d)/i.test(allText(f)), "no dark blue") || need(/color\s*:\s*(white|#fff\b|#ffffff)/i.test(allText(f)), "text is not white") },
];

// ---- a stand-in preview frame: each rev "loads"; scripts that do not parse log a SyntaxError
function scriptsOf(files, entry) {
  const out = [];
  const h = files.get(entry);
  if (typeof h === "string") {
    const re = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
    let m;
    while ((m = re.exec(h))) {
      const src = /\bsrc\s*=\s*["']([^"']+)["']/.exec(m[1]);
      if (src) { const p = src[1].replace(/^\.?\//, ""); if (typeof files.get(p) === "string") out.push([p, files.get(p)]); }
      else if (m[2].trim()) out.push([entry, m[2]]);
    }
  }
  return out;
}
function syntaxError(code) {
  const body = code.replace(/^\s*import\s[^;\n]*;?/gm, "").replace(/^\s*export\s+(default\s+)?/gm, "");
  try { new Function(body); return null; } catch (e) { return e instanceof SyntaxError ? e.message : null; }
}
function fakeFrame(server) {
  server.onUpdate(({ port, rev, stopped }) => {
    if (stopped) return;
    const st = server.served.get(port);
    if (!st) return;
    if (!st.frames) st.frames = 1;
    setTimeout(() => {
      if (st.snap.rev !== rev) return;
      const files = new Map([...st.snap.files].map(([p, v]) => [p, typeof v === "string" ? v : v?.text ?? (v?.bytes ? new TextDecoder().decode(v.bytes) : v)]));
      for (const [src, code] of scriptsOf(files, st.snap.entry)) {
        const e = syntaxError(code);
        if (e) server.pushLog(port, { level: "error", text: `Uncaught SyntaxError: ${e}`, src, line: 1, ms: 20, rev });
      }
      server.frameEvent(port, { t: "ready", rev, ms: 30 });
      server.frameEvent(port, { t: "idle", rev, ms: 530 });
    }, 50);
  });
}

// ---- the model
const cfg = JSON.parse(await Deno.readTextFile(M + "config.json"));
const tj = JSON.parse(await Deno.readTextFile(M + "tokenizer.json"));
const tok = makeTokenizer(tj);
const style = detectStyle(tj.chat_template || "");
const buf = (await Deno.readFile(M + "model.gguf")).buffer;
const G = parseGGUFHeader(buf);
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const weights = await ggufWeights(G, (i) => new Uint8Array(buf, i.byteOffset, i.byteLength), { lo: 0, hi: cfg.num_hidden_layers, hasEmbed: true, hasHead: true });
const engine = await DenseEngine.create({ device, cfg, weights, maxSeq: CTX });
console.log(`model ${M} · style ${style} · context ${CTX} · harness ${R}`);

// the room's Code mode budget (harness/room-model.js): maxSeq - min(maxNew, maxSeq / 4) - 64
const budget = CTX - (Math.min(8192, Math.floor(CTX / 4)) + 64);
const count = (t) => tok.encode(t).length;
const ids = arg("tasks", "") ? arg("tasks").split(",") : TASKS.map((t) => t.id);
const results = [];
for (const id of ids) {
  const task = TASKS.find((t) => t.id === id);
  if (!task) throw new Error("unknown task " + id);
  const ws = new MemoryWorkspace(task.files || {});
  const server = new PreviewServer(ws);
  fakeFrame(server);
  // run_js as the room has it (its prompt text matters); no sandbox here, so a run reports "ok"
  const runner = async () => ({ logs: [], done: { ok: true, ms: 5 } });
  const tools = [...codingTools(ws, { server }), ...previewTools(server), runJsTool(server, { runner })];
  const model = engineModel(engine, tok, { tools, style, maxNew: 8192, spec: false });
  const log = [];
  let served = false, lastServe = "", writes = {};
  const A = new Agent({ generate: model.generate, tools, style, system: CODE_SYSTEM, maxSteps: STEPS, approve: async () => true, budget, count,
    usage: () => model.stats.last,
    onEvent: (e) => {
      if (e.type === "tool") {
        const line = `  [${e.step}] ${e.call.name || "?"}(${JSON.stringify(e.call.arguments || {}).slice(0, 120)}) -> ${String(e.result).slice(0, 200).replace(/\n/g, "⏎")}`;
        log.push(line); if (VERBOSE) console.log(line);
        if (e.call.name === "serve") { lastServe = String(e.result); if (/^serving /.test(lastServe)) served = true; }
        if (e.call.name === "write_file" && e.call.arguments?.path) { const p = String(e.call.arguments.path); writes[p] = (writes[p] || 0) + 1; }
      }
      if (e.type === "text" && VERBOSE) Deno.stdout.writeSync(new TextEncoder().encode(e.text));
    } });
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TASK_MS);
  const t0 = Date.now();
  let r;
  try { r = await A.run(task.prompt, { signal: ctrl.signal }); }
  catch (e) { r = { reason: "error", text: String(e?.message || e), steps: 0 }; }
  clearTimeout(timer);
  await new Promise((res) => setTimeout(res, 150));
  const files = new Map();
  for (const p of await ws.walk()) files.set(p, await ws.read(p));
  // the preview's final state: no error in the last rev
  const port = server.ports()[0]?.port;
  const errs = port ? server.logs(port, 0).lines.filter((e) => e.level === "error" && e.rev === server.snapshot(port).rev) : [];
  let why = r.reason !== "done" ? `ended: ${r.reason}` : !served ? "never served" : errs.length ? "page errors: " + errs[0].text.slice(0, 80) : task.check(files);
  const rewrites = Object.entries(writes).filter(([, n]) => n >= 3).map(([p, n]) => `${p}×${n}`);
  if (!why && rewrites.length) why = "rewrite loop: " + rewrites.join(" ");
  const ok = !why;
  results.push({ id, ok, why, reason: r.reason, steps: r.steps, s: ((Date.now() - t0) / 1000).toFixed(0) });
  console.log(`${id.padEnd(12)} ${ok ? "PASS" : "FAIL"} ${String(r.reason).padEnd(8)} steps ${String(r.steps).padEnd(3)} ${results.at(-1).s}s ${why}`);
  if (!ok || VERBOSE) { for (const l of log) console.log(l); console.log("  final:", (r.text || "").slice(0, 160).replace(/\n/g, " "), "· files:", [...files.keys()].join(", ")); }
  if (Deno.env.get("SAVE_DIR")) await Deno.writeTextFile(`${Deno.env.get("SAVE_DIR")}/${id}.json`, JSON.stringify({ task: id, result: r, turns: A.turns, files: Object.fromEntries(files) }, null, 1));
}
const pass = results.filter((r) => r.ok).length;
console.log(`PASS ${pass}/${results.length} (${Math.round(100 * pass / results.length)} %)`);
Deno.exit(0);
