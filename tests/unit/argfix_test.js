// Small-model argument repair (harness/argfix.js), serve's argument repair
// (harness/preview-tools.js serveArgs), the loose JSON call parser, and the agent's paths for a
// JSON-style model: repaired calls, bare JSON calls, a cut JSON write, and ending cleanly after
// a clean serve instead of rewriting files.
import { fixArgs, fixToolName, cleanPathArg, toBool, toInt, toText } from "../../harness/argfix.js";
import { serveArgs, previewTools } from "../../harness/preview-tools.js";
import { parseLooseJSON, parseCallBody } from "../../harness/tools.js";
import { Agent, bareJsonCalls, salvageWrite, SERVED_OK } from "../../harness/agent.js";
import { codingTools } from "../../harness/codetools.js";
import { MemoryWorkspace } from "../../harness/workspace.js";
import { PreviewServer } from "../../harness/preview.js";
import { hint } from "../../harness/cards.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const WRITE = { type: "object", properties: { path: { type: "string" }, content: { type: "string" }, append: { type: "boolean" } }, required: ["path", "content"] };
const READ = { type: "object", properties: { path: { type: "string" }, start_line: { type: "integer" }, end_line: { type: "integer" } }, required: ["path"] };
const EDIT = { type: "object", properties: { path: { type: "string" }, old: { type: "string" }, new: { type: "string" } }, required: ["path", "old", "new"] };
const SERVE = { type: "object", properties: { dir: { type: "string" }, port: { type: "integer" }, entry: { type: "string" } } };

Deno.test("argfix: value helpers", () => {
  eq(cleanPathArg(" ./index.html "), "index.html");
  eq(cleanPathArg("/src/app.js"), "src/app.js");
  eq(cleanPathArg('"index.html"'), "index.html");
  eq(cleanPathArg("file:///a/b.css"), "a/b.css");
  eq(cleanPathArg("."), "");
  eq([toBool("false"), toBool("True"), toBool(1), toBool("yes"), toBool("maybe")], [false, true, true, true, undefined]);
  eq([toInt("5173"), toInt(":8080"), toInt("http://localhost:5173/"), toInt(12.4), toInt("abc")], [5173, 8080, 5173, 12, undefined]);
  eq(toText(["<h1>", "hi", "</h1>"]), "<h1>\nhi\n</h1>\n");
  eq(toText({ a: 1 }), '{\n  "a": 1\n}');
  eq(toText(42), "42");
});

Deno.test("argfix: other names for parameters, case and style", () => {
  eq(fixArgs({ file: "./index.html", contents: "<h1>hi</h1>" }, WRITE), { path: "index.html", content: "<h1>hi</h1>" });
  eq(fixArgs({ filename: "/a.js", text: "x" }, WRITE), { path: "a.js", content: "x" });
  eq(fixArgs({ Path: "a.js", Content: "x" }, WRITE), { path: "a.js", content: "x" });
  eq(fixArgs({ path: "a.js", startLine: "3", endLine: 9 }, READ), { path: "a.js", start_line: 3, end_line: 9 });
  eq(fixArgs({ file_path: "a.js", old_string: "a", new_string: "b" }, EDIT), { path: "a.js", old: "a", new: "b" });
  eq(fixArgs({ path: "a.js", search: "a", replace: "" }, EDIT), { path: "a.js", old: "a", new: "" });
  // a given parameter is never overridden by an alias
  eq(fixArgs({ path: "a.js", file: "b.js", content: "x" }, WRITE), { path: "a.js", content: "x" });
  // serve: "path" / "file" mean dir / entry
  eq(fixArgs({ path: "index.html", port: "5173" }, SERVE), { dir: "index.html", port: 5173 });
  eq(fixArgs({ root: ".", file: "./index.html" }, SERVE), { dir: "", entry: "index.html" });
});

Deno.test("argfix: types, positional arguments, wrapping, bare values", () => {
  eq(fixArgs({ path: "a", content: ["line 1", "line 2"], append: "false" }, WRITE), { path: "a", content: "line 1\nline 2\n", append: false });
  eq(fixArgs({ path: "a", content: "x", append: "true" }, WRITE), { path: "a", content: "x", append: true });
  eq(fixArgs({ path: "a", content: "x", append: "sure" }, WRITE), { path: "a", content: "x" }, "an unclear boolean falls back to the default");
  eq(fixArgs({ path: "a", content: "x", mode: "append" }, WRITE), { path: "a", content: "x", append: true });
  eq(fixArgs({ path: "package.json", content: { name: "x" } }, WRITE), { path: "package.json", content: '{\n  "name": "x"\n}' });
  eq(fixArgs(["index.html", "<p>hi</p>", "false"], WRITE), { path: "index.html", content: "<p>hi</p>", append: false });
  eq(fixArgs({ arguments: { path: "a", content: "b" } }, WRITE), { path: "a", content: "b" });
  eq(fixArgs({ write_file: { file: "a", content: "b" } }, WRITE), { path: "a", content: "b" });
  eq(fixArgs('{"path": "a"}', READ), { path: "a" });
  eq(fixArgs("index.html", READ), { path: "index.html" });
  eq(fixArgs(null, READ), {});
  eq(fixArgs({ port: "abc", dir: null }, SERVE), {}, "values that cannot be converted are dropped");
});

Deno.test("argfix: tool names", () => {
  const has = (n) => ["write_file", "read_file", "list_dir", "edit_file", "serve", "search"].includes(n);
  eq(["create_file", "writeFile", "ls", "str_replace", "start_server", "grep", "cat", "serve", "nope"].map((n) => fixToolName(n, has)),
    ["write_file", "write_file", "list_dir", "edit_file", "serve", "search", "read_file", "serve", "nope"]);
  eq(fixToolName("run_js", has), "run_js", "an alias target that is not a tool is not used");
});

Deno.test("serveArgs: dir given as the entry file (the Qwen3 1.7B failure)", () => {
  const files = ["index.html"];
  const a = serveArgs({ dir: "index.html", port: 5173, entry: "index.html" }, files);
  eq([a.dir, a.entry, a.port], ["", "index.html", 5173]);
  ok(/is a file/.test(a.notes[0]), a.notes.join());
  const b = serveArgs({ dir: "site/index.html" }, ["site/index.html", "site/app.js"]);
  eq([b.dir, b.entry], ["site", "index.html"]);
  const c = serveArgs({ dir: "site", entry: "site/index.html" }, ["site/index.html"]);
  eq([c.dir, c.entry, c.notes], ["site", "index.html", []]);
  // a dir that only looks like a file (not written yet) is treated the same
  const d = serveArgs({ dir: "game.html" }, ["index.html"]);
  eq([d.dir, d.entry], ["", "index.html"]);
});

Deno.test("serveArgs: defaults, missing folders and pages, bad ports", () => {
  eq(serveArgs({}, ["index.html"]), { dir: "", entry: "index.html", port: 5173, notes: [] });
  const a = serveArgs({ dir: "my-website" }, ["index.html", "style.css"]);
  eq([a.dir, a.entry], ["", "index.html"]);
  ok(/no folder my-website/.test(a.notes[0]));
  const b = serveArgs({ dir: "app" }, ["web/index.html"]);
  eq([b.dir, b.entry], ["web", "index.html"]);
  const c = serveArgs({}, ["game.html", "game.js"]);
  eq(c.entry, "game.html", "the one page there is");
  eq(serveArgs({ entry: "home.html" }, ["index.html"]).entry, "index.html");
  for (const port of [80, 0, 99999, 3.5, "x"]) eq(serveArgs({ port }, ["index.html"]).port, 5173, `port ${port}`);
  eq(serveArgs({ port: 8080 }, ["index.html"]).port, 8080);
  ok(/Write index.html with write_file first/.test(serveArgs({}, []).error));
  ok(/no HTML page.*app\.js/.test(serveArgs({}, ["app.js"]).error));
  const e = serveArgs({}, ["a.html", "b.html"]);
  ok(/"entry": "a.html"/.test(e.error), e.error);
  eq(serveArgs({ dir: "../x" }, ["index.html"]).dir, "", "a dir outside the project is the root");
});

Deno.test("serve tool: runs the repaired arguments and says what it changed", async () => {
  const ws = new MemoryWorkspace({ "index.html": "<h1>hi</h1>" });
  const server = new PreviewServer(ws, { frameWait: 10 });
  const serve = previewTools(server).find((t) => t.name === "serve");
  const r = await serve.run({ dir: "index.html", port: 5173, entry: "index.html" });
  ok(/^serving \. on :5173 \(index\.html/.test(r), r);
  ok(/dir index\.html is a file/.test(r), r);
  const e = await previewTools(new PreviewServer(new MemoryWorkspace({}))).find((t) => t.name === "serve").run({});
  ok(/^error: the project is empty\. Write index\.html/.test(e), e);
  server.close?.();
});

Deno.test("loose JSON: raw newlines, trailing commas, arrays, a cut string", () => {
  eq(parseLooseJSON('{"path": "a", "content": "<h1>\n\thi</h1>",}'), { path: "a", content: "<h1>\n\thi</h1>" });
  eq(parseLooseJSON('{"a": [1, 2, {"b": 3'), { a: [1, 2, { b: 3 }] });
  eq(parseLooseJSON('{"name": "w", "arguments": {"content": "l1\\nl', { open: true }), { name: "w", arguments: { content: "l1\nl" } });
  let threw = false;
  try { parseLooseJSON('{"a": "open'); } catch { threw = true; }
  ok(threw, "a cut string is an error unless open");
});

Deno.test("parseCallBody: other JSON call shapes", () => {
  eq(parseCallBody('{"name": "serve", "dir": "."}'), { name: "serve", arguments: { dir: "." } });
  eq(parseCallBody('{"type": "function", "function": {"name": "serve", "arguments": "{\\"port\\": 5173}"}}'), { name: "serve", arguments: { port: 5173 } });
  eq(parseCallBody('[{"name": "list_dir", "arguments": {}}]'), { name: "list_dir", arguments: {} });
  eq(parseCallBody('{"tool": "read_file", "args": {"path": "a"}}'), { name: "read_file", arguments: { path: "a" } });
  eq(parseCallBody('{"name": "write_file", "arguments": {"path": "a", "content": "x\ny"}}'), { name: "write_file", arguments: { path: "a", content: "x\ny" } });
});

Deno.test("bareJsonCalls: a JSON call outside <tool_call>, not the objects inside it", () => {
  const byName = new Map([["write_file", {}], ["serve", {}]]);
  const text = 'Here:\n```json\n{"name": "create_file", "arguments": {"path": "a.html", "content": "{\\"name\\": 1}"}}\n```\nthen {"name": "serve", "arguments": {}}';
  eq(bareJsonCalls(text, byName), [{ name: "write_file", arguments: { path: "a.html", content: '{"name": 1}' } }, { name: "serve", arguments: {} }]);
  eq(bareJsonCalls('the object {"name": "Bob", "arguments": {}} is data', byName), []);
});

Deno.test("salvageWrite: a JSON write cut mid-content keeps its complete lines", () => {
  const raw = '\n{"name": "write_file", "arguments": {"path": "a.js", "content": "line 1\\nline 2\\nli';
  const c = salvageWrite(raw, () => WRITE);
  eq(c.arguments, { path: "a.js", content: "line 1\nline 2\n", append: false });
  eq(c.salvage, { lines: 2, last: "line 2" });
});

Deno.test("write_file: the same content again is 'unchanged'; no content is an error", async () => {
  const ws = new MemoryWorkspace({ "index.html": "<h1>hi</h1>\n" });
  const w = codingTools(ws).find((t) => t.name === "write_file");
  ok(/^unchanged: index\.html already has exactly this content/.test(await w.run({ path: "index.html", content: "<h1>hi</h1>\n" })));
  ok(/^error: write_file needs content/.test(await w.run({ path: "index.html" })));
  eq(await ws.read("index.html"), "<h1>hi</h1>\n", "not emptied");
  ok(/^error: write_file needs a path/.test(await w.run({ content: "x" })));
  // an append of the whole file again replaces it
  const page = "<!doctype html>\n<body>\n<h1>Hi</h1>\n<p>more</p>\n</body>\n";
  await w.run({ path: "p.html", content: page });
  ok(/replaced the file instead of being appended/.test(await w.run({ path: "p.html", content: page + "<p>new</p>\n", append: true })));
  eq(await ws.read("p.html"), page + "<p>new</p>\n");
  ok(/^appended to p\.html/.test(await w.run({ path: "p.html", content: "<p>tail</p>\n", append: true })));
  // a page replaced by a piece of one is refused; a whole new page is fine
  ok(/^error: not written: p\.html is a whole page/.test(await w.run({ path: "p.html", content: "<button>hi</button>\n" })));
  // (the 1.7B "adding" a button: a bare <body> with only the new part)
  ok(/^error: not written: p\.html is a whole page/.test(await w.run({ path: "p.html", content: "<body>\n  <button onclick=\"alert('hi')\">Click me</button>\n</body>" })));
  ok(/^wrote p\.html/.test(await w.run({ path: "p.html", content: "<!doctype html>\n<body><button>hi</button></body>\n" })));
  const l = codingTools(ws).find((t) => t.name === "list_dir");
  eq(await l.run({ path: "index.html" }), "index.html is a file, not a folder; read_file shows its lines");
  ok(/^error: there is no folder src \(the project has: index\.html/.test(await l.run({ path: "src" })));
  const r = codingTools(ws).find((t) => t.name === "read_file");
  ok(/does not exist yet \(the project has: index\.html, p\.html\); create it with write_file/.test(await r.run({ path: "app.js" })));
});

// a scripted JSON-style model: one reply per step
function scripted(replies) {
  let i = 0;
  return async function* () { yield replies[i++] ?? "done"; };
}
const jcall = (name, args) => `<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`;
function project(files = {}) {
  const ws = new MemoryWorkspace(files);
  const server = new PreviewServer(ws, { frameWait: 10 });
  // a preview frame that loads every rev cleanly
  server.onUpdate(({ port, rev, stopped }) => {
    if (stopped) return;
    const st = server.served.get(port);
    st.frames = 1;
    setTimeout(() => { server.frameEvent(port, { t: "ready", rev, ms: 5 }); server.frameEvent(port, { t: "idle", rev, ms: 505 }); }, 5);
  });
  return { ws, server, tools: [...codingTools(ws, { server }), ...previewTools(server)] };
}

Deno.test("agent (json): the 1.7B's serve slip runs, and a rewrite after a clean serve ends the task", async () => {
  const P = project();
  const page = "<!doctype html>\n<h1>Hello</h1>\n";
  const results = [];
  const A = new Agent({ style: "json", tools: P.tools, onEvent: (e) => { if (e.type === "tool") results.push([e.call.name, e.result]); },
    generate: scripted([
      "Sure! I'll create a simple website.\n" + jcall("write_file", { path: "index.html", content: page }),
      jcall("serve", { dir: "index.html", port: 5173, entry: "index.html" }),
      jcall("write_file", { path: "./index.html", content: page }),
      "never reached",
    ]) });
  const r = await A.run("build a website that says hello verry basic please");
  eq(r.reason, "done");
  eq(r.steps, 3);
  ok(/^serving \. on :5173/.test(results[1][1]), results[1][1]);
  ok(results[1][1].includes(SERVED_OK.trim()), "the coach note after a clean serve");
  ok(/^unchanged:/.test(results[2][1]), results[2][1]);
  eq(await P.ws.read("index.html"), page);
});

Deno.test("agent (json): after a clean serve the model's one-line answer ends it; repeats of serve end it too", async () => {
  const P = project({ "index.html": "<h1>x</h1>" });
  const A = new Agent({ style: "json", tools: P.tools, generate: scripted([jcall("serve", {}), "Built a page that says x."]) });
  const r = await A.run("serve it");
  eq([r.reason, r.steps, r.text], ["done", 2, "Built a page that says x."]);
  const B = new Agent({ style: "json", tools: project({ "index.html": "<h1>x</h1>" }).tools, generate: scripted([jcall("serve", {}), jcall("serve", {}), jcall("serve", {}), jcall("serve", {})]) });
  const s = await B.run("serve it");
  eq([s.reason, s.steps], ["done", 2], "a repeated clean serve is the end, not 'stuck'");
});

Deno.test("agent (json): an edit after a clean serve keeps going; aliases and bare JSON calls run", async () => {
  const P = project({ "index.html": "<h1>x</h1>" });
  const seen = [];
  const A = new Agent({ style: "json", tools: P.tools, onEvent: (e) => { if (e.type === "tool") seen.push(e.call.name + " " + e.result.split("\n")[0]); },
    generate: scripted([
      jcall("start_server", { path: "." }),
      '```json\n{"name": "str_replace", "arguments": {"file": "index.html", "old_string": "x", "new_string": "y"}}\n```',
      jcall("serve", {}),
      "Changed x to y.",
    ]) });
  const r = await A.run("change x to y");
  eq(r.reason, "done");
  eq(r.text, "Changed x to y.");
  ok(/^serve serving \. on :5173/.test(seen[0]), seen[0]);
  ok(/^edit_file edited index\.html/.test(seen[1]), seen[1]);
  eq(await P.ws.read("index.html"), "<h1>y</h1>");
});

Deno.test("agent: xml models get no coach note and keep the XML format card; json models get a JSON one", async () => {
  const P = project({ "index.html": "<h1>x</h1>" });
  const out = [];
  const call = "<tool_call>\n<function=serve>\n</function>\n</tool_call>";
  const A = new Agent({ style: "xml", tools: P.tools, onEvent: (e) => { if (e.type === "tool") out.push(e.result); }, generate: scripted([call, "ok"]) });
  await A.run("serve");
  ok(!out[0].includes(SERVED_OK.trim()), out[0]);
  ok(hint("format", "json").includes('{"name": "write_file"'));
  ok(hint("format").includes("<function=read_file>"));
  const J = new Agent({ style: "json", tools: P.tools, onEvent: (e) => { if (e.type === "tool") out.push(e.result); }, generate: scripted(["<tool_call>\nnot json\n</tool_call>", "ok"]) });
  await J.run("x");
  const res = J.turns.find((t) => t.text.startsWith("<tool_response>")).text;
  ok(res.includes('{"name": "write_file"') && !res.includes("<function="), res);
});

Deno.test("agent (json): polling preview_logs with nothing new after a clean serve ends the task", async () => {
  const P = project({ "index.html": "<h1>x</h1>" });
  const A = new Agent({ style: "json", tools: P.tools,
    generate: scripted([jcall("serve", {}), jcall("preview_logs", { port: 5173, since: 0 }), jcall("preview_logs", { port: 5173, since: 0 })]) });
  const r = await A.run("serve it");
  eq([r.reason, r.steps], ["done", 2]);
});

Deno.test("edit_file: old copied with read_file's line numbers still matches", async () => {
  const ws = new MemoryWorkspace({ "index.html": "<html>\n<body>\n  <h1>Snake</h1>\n</body>\n</html>\n" });
  const e = codingTools(ws).find((t) => t.name === "edit_file");
  const r = await e.run({ path: "index.html", old: "4|</body>", new: '<script src="app.js"></script>\n</body>' });
  ok(/^edited index\.html lines 4-5 .*matched ignoring the line numbers/.test(r), r);
  eq(await ws.read("index.html"), '<html>\n<body>\n  <h1>Snake</h1>\n<script src="app.js"></script>\n</body>\n</html>\n');
  // the numbers on both sides, several lines; wrong numbers do not matter
  const r2 = await e.run({ path: "index.html", old: "9|<body>\n10|  <h1>Snake</h1>", new: "9|<body>\n10|  <h1>Snake!</h1>" });
  ok(/^edited index\.html/.test(r2), r2);
  ok((await ws.read("index.html")).includes("<body>\n  <h1>Snake!</h1>\n"));
  // text that really has "N|" in it is matched as it is
  const ws2 = new MemoryWorkspace({ "t.md": "1|a\n2|b\n" });
  const e2 = codingTools(ws2).find((t) => t.name === "edit_file");
  await e2.run({ path: "t.md", old: "1|a", new: "1|z" });
  eq(await ws2.read("t.md"), "1|z\n2|b\n");
});

Deno.test("agent: one failing call repeated with reads in between gets the loop card, then stops as stuck", async () => {
  const P = project({ "index.html": "<h1>x</h1>\n" });
  const bad = jcall("edit_file", { path: "index.html", old: "nope", new: "y" }), read = jcall("read_file", { path: "index.html" });
  const cards = [];
  const A = new Agent({ style: "json", tools: P.tools, onEvent: (e) => { if (e.type === "card") cards.push(e.id); },
    generate: scripted([bad, read, bad, read, bad, read, bad, read, bad]) });
  const r = await A.run("edit");
  eq(r.reason, "stuck");
  eq(r.steps, 5);
  ok(cards.includes("loop"), cards.join());
});
