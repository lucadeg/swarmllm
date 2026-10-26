// tests/eval/: the task table is well formed (each golden run only calls real tools, with valid
// arguments), and harness/run-js.js turns a run into a pass/fail status (the document is run_js_test.js). The checks
// themselves run in a browser: node tests/eval/run.mjs --model mock (golden runs pass, bad files fail).
import { TASKS } from "../eval/tasks/index.js";
import { runProbe, formatProbe, runJsTool } from "../../harness/run-js.js";
import { ToolCallParser } from "../../harness/tools.js";
import { codingTools } from "../../harness/codetools.js";
import { previewTools } from "../../harness/preview-tools.js";
import { MemoryWorkspace } from "../../harness/workspace.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const TOOLS = [...codingTools(new MemoryWorkspace()), ...previewTools({}), runJsTool({})];
const byName = new Map(TOOLS.map((t) => [t.name, t]));

Deno.test("eval tasks: 12, unique ids, every field a runner needs", () => {
  ok(TASKS.length === 12, `${TASKS.length} tasks`);
  ok(new Set(TASKS.map((t) => t.id)).size === 12, "duplicate ids");
  for (const t of TASKS) {
    ok(typeof t.prompt === "string" && t.prompt.length > 20, `${t.id}: prompt`);
    ok(typeof t.check === "string" && /ok\(/.test(t.check), `${t.id}: check asserts something`);
    ok(Array.isArray(t.mock) && t.mock.length >= 2, `${t.id}: mock run`);
    ok(t.bad && typeof t.bad === "object", `${t.id}: bad files for the self-test`);
    ok(!t.check.includes("${"), `${t.id}: check has an unexpanded template`);
  }
});

Deno.test("eval tasks: the golden runs call only known tools with their required arguments, and end with an answer", async () => {
  for (const t of TASKS) {
    const replies = t.mock.map((r) => (typeof r === "function" ? null : r));
    ok(typeof replies[replies.length - 1] === "string" && !replies[replies.length - 1].includes("<tool_call>"), `${t.id}: last reply is the answer`);
    for (const r of replies.slice(0, -1)) {
      if (r == null) continue;   // a function reply (looks at the conversation)
      const P = new ToolCallParser({ schemaFor: (n) => byName.get(n)?.parameters });
      const calls = [...P.feed(r).calls, ...P.end().calls];
      ok(calls.length === 1, `${t.id}: one call per reply (${calls.length})`);
      const c = calls[0], tool = byName.get(c.name);
      ok(tool, `${t.id}: unknown tool ${c.name}`);
      for (const k of tool.parameters.required || []) ok(k in c.arguments, `${t.id}: ${c.name} without ${k}`);
    }
  }
});

Deno.test("runProbe: status from the runner's result; the page falls back to a blank one; formatProbe is run_js's format", async () => {
  const enc = new TextEncoder(), f = (t) => ({ type: "text/html", bytes: enc.encode(t), hash: "h" });
  const snap = { files: new Map([["index.html", f("<body><p>hi</p></body>")]]) };
  let seen = null;
  const fake = (res) => async (s, o) => { seen = { s, o }; return res; };
  let r = await runProbe(snap, { code: "1", page: "index.html", timeout: 8000, runner: fake({ logs: [], done: { ok: true, ms: 12 } }) });
  ok(r.status === "ok" && r.ms === 12 && seen.s.entry === "index.html" && seen.o.timeout === 8000, JSON.stringify(r));
  ok(formatProbe(r) === "ok in 12 ms (no output; await async work and print with console.log)", formatProbe(r));
  r = await runProbe(snap, { code: "1", page: "nope.html", runner: fake({ logs: [{ level: "error", text: "boom", src: "a.js", line: 3, col: 1, ms: 0 }], done: { ok: true, ms: 5 } }) });
  ok(r.status === "error" && seen.s.entry === "__run.html", "a logged error fails; a missing page is a blank one");
  ok(formatProbe(r) === "error in 5 ms\n[0.0s] error a.js:3:1 boom", formatProbe(r));
  r = await runProbe(snap, { code: "1", timeout: 8000, runner: fake({ logs: [], done: null, hung: false }) });
  ok(r.status === "timeout" && formatProbe(r).startsWith("timed out after 8 s"), formatProbe(r));
});
