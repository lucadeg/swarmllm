import { xmlCall } from "./_util.js";

const HELPERS = `function formatTime(sec) {
  const m = Math.floor(sec / 60), s = sec % 60;
  return m + ":" + String(s).padStart(2, "0");
}
const slugify = (t) => t.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
`;
const REST = `const tracks = [["Morning Light", 245], ["Night Drive", 312], ["Tide & Stone", 98]];
const ul = document.getElementById("tracks");
for (const [name, sec] of tracks) {
  const li = document.createElement("li");
  li.id = slugify(name);
  li.textContent = name + " (" + formatTime(sec) + ")";
  ul.append(li);
}
`;
const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>Playlist</title></head>
<body><h1>Playlist</h1><ul id="tracks"></ul><script type="module" src="app.js"></script></body></html>
`;

export default {
  id: "refactor",
  kind: "multi-file",
  prompt: "Move the helper functions formatTime and slugify out of app.js into a new module utils.js, export them there and import them in app.js. The page must stay the same.",
  files: { "index.html": html, "app.js": HELPERS + REST },
  maxSteps: 12,
  check: `
import { formatTime, slugify } from "./utils.js";
ok(typeof formatTime === "function" && typeof slugify === "function", "utils.js does not export formatTime and slugify");
ok(formatTime(98) === "1:38" && slugify("Tide & Stone") === "tide-stone", "the helpers changed behaviour");
const li = $("#tide-stone");
ok(li && text(li) === "Tide & Stone (1:38)" && $$("#tracks li").length === 3, "the page no longer renders the same list");
`,
  // file-state part of the check (suite.js runs it against the workspace)
  checkFiles: async (read) => {
    const app = await read("app.js");
    if (app == null) return "app.js is gone";
    if (!/from\s*["']\.\/utils\.js["']/.test(app)) return "app.js does not import from ./utils.js";
    if (/function\s+(formatTime|slugify)\b|(const|let|var)\s+(formatTime|slugify)\s*=/.test(app)) return "app.js still defines a helper";
    return null;
  },
  mock: [
    xmlCall("read_file", { path: "app.js" }),
    xmlCall("write_file", { path: "utils.js", content: HELPERS.replace("function formatTime", "export function formatTime").replace("const slugify", "export const slugify") }),
    xmlCall("write_file", { path: "app.js", content: `import { formatTime, slugify } from "./utils.js";\n` + REST }),
    xmlCall("serve", { port: 5173 }),
    "Moved formatTime and slugify into utils.js; app.js imports them.",
  ],
  bad: { "utils.js": HELPERS.replace("function formatTime", "export function formatTime").replace("const slugify", "export const slugify") },   // app.js still defines them
};
