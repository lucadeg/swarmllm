// harness/workspace.js DirWorkspace + codetools against a real FileSystemDirectoryHandle (an OPFS
// folder: the same API as a folder the user picks with showDirectoryPicker). No GPU needed.
//   NODE_PATH=... node tests/e2e/workspace_browser.mjs
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";
const PORT = 18985;
async function pageMain() {
  const { DirWorkspace } = await import("/harness/workspace.js");
  const { codingTools } = await import("/harness/codetools.js");
  const out = [], res = [];
  const check = (n, ok, d = "") => { res.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${n}${d ? "  " + d : ""}`); };
  const root = await navigator.storage.getDirectory();
  await root.removeEntry("ws-test", { recursive: true }).catch(() => {});
  const dir = await root.getDirectoryHandle("ws-test", { create: true });
  const ws = new DirWorkspace(dir);
  await ws.write("src/add.js", "export const add = (a, b) => a - b;\n");
  await ws.write("node_modules/x/index.js", "return 1\n");
  await ws.write("README.md", "# t\n");
  const T = Object.fromEntries(codingTools(ws).map((t) => [t.name, t]));
  check("list_dir", (await T.list_dir.run({})) === "node_modules/\nREADME.md\nsrc/", JSON.stringify(await T.list_dir.run({})));
  check("read_file", (await T.read_file.run({ path: "src/add.js" })).includes("1|export const add"));
  check("search skips node_modules", (await T.search.run({ pattern: "return|a - b" })) === "src/add.js:1: export const add = (a, b) => a - b;");
  check("edit_file", (await T.edit_file.run({ path: "src/add.js", old: "a - b", new: "a + b" })).startsWith("edited") && (await ws.read("src/add.js")).includes("a + b"));
  check("write_file creates folders", (await T.write_file.run({ path: "test/add.test.js", content: "ok\n" })).startsWith("wrote") && await ws.exists("test/add.test.js"));
  await ws.writeBytes("img/a.bin", new Uint8Array([0, 255, 7]));
  check("bytes round-trip", (await ws.readBytes("img/a.bin")).join() === "0,255,7");
  await ws.remove("img");
  check("remove a folder", !(await ws.exists("img/a.bin")) && !(await ws.exists("img")));
  let threw = false; try { await ws.read("../secret"); } catch { threw = true; }
  check("paths cannot leave the folder", threw);
  await root.removeEntry("ws-test", { recursive: true });

  // harness/projects.js: OPFS scratch projects with metadata and sessions in IndexedDB
  const P = await import("/harness/projects.js");
  for (const p of await P.listProjects()) await P.deleteProject(p.id);
  const a = await P.createProject("Tetris!"), b = await P.createProject("tetris");
  check("create: slug from the name, then -2", a.id === "opfs:tetris" && b.id === "opfs:tetris-2" && a.name === "Tetris!", `${a.id} ${b.id}`);
  await a.ws.write("index.html", "<p>hi</p>");
  const seen = [];
  a.ws.onChange((e) => seen.push(e.path));
  await a.ws.write("game.js", "1");
  check("project ws is watched", seen.join() === "game.js");
  const onDisk = await (await (await (await root.getDirectoryHandle(P.PROJECTS_DIR)).getDirectoryHandle("tetris")).getFileHandle("index.html")).getFile();
  check("files live in OPFS pooled-projects/<slug>/", (await onDisk.text()) === "<p>hi</p>");
  const list = await P.listProjects();
  check("list", list.length === 2 && list.every((p) => p.kind === "opfs" && p.updated > 0), JSON.stringify(list));
  await P.saveSession(a.id, { v: 1, turns: [{ role: "user", text: "build tetris" }] });
  const re = await P.openProject(a.id);
  check("reopen: files and session come back", (await re.ws.read("game.js")) === "1" && (await P.loadSession(a.id))?.turns?.[0]?.text === "build tetris");
  await P.deleteProject(a.id); await P.deleteProject(b.id);
  let gone = false; try { await (await root.getDirectoryHandle(P.PROJECTS_DIR)).getDirectoryHandle("tetris"); } catch { gone = true; }
  check("delete removes files, metadata and session", gone && !(await P.listProjects()).length && (await P.loadSession(a.id)) === null);
  check("slugify", P.slugify("  Snake Game (v2) ") === "snake-game-v2" && P.slugify("🙂") === "project");
  return { out, ok: res.every(Boolean) };
}
const srv = serveRepo(PORT, {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: ["--no-sandbox"] });   // no WebGPU needed
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const r = await page.evaluate(pageMain);
for (const l of r.out) console.log(l);
console.log(r.ok ? "WORKSPACE PASS" : "WORKSPACE FAIL");
await browser.close(); srv.close();
process.exit(r.ok ? 0 : 1);
