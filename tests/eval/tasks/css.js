import { xmlCall } from "./_util.js";

const html = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Notes</title><link rel="stylesheet" href="style.css"></head>
<body>
  <header><strong>Notes</strong> <button>New note</button></header>
  <main>
    <p>Meeting on Tuesday: bring the sketches.</p>
    <p>Call the printer about the proofs.</p>
    <p>Order more A5 paper.</p>
    <button>Load more</button>
  </main>
</body></html>
`;
const css = `body { margin: 0; font: 16px system-ui, sans-serif; }
header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: #243447; color: #fff; }
main { padding: 16px; min-height: 150vh; }
button { padding: 6px 12px; border: 1px solid #889; background: #f4f4f8; }
`;
const EXPECTED = "Notes New note Meeting on Tuesday: bring the sketches. Call the printer about the proofs. Order more A5 paper. Load more";

export default {
  id: "css",
  kind: "style",
  prompt: "Make the header stick to the top of the window when the page scrolls, and give every button 8px rounded corners. Change nothing else.",
  files: { "index.html": html, "style.css": css },
  maxSteps: 10,
  check: `
const h = $("header");
ok(h, "the header is gone");
ok(getComputedStyle(h).position === "sticky" || getComputedStyle(h).position === "fixed", "header position is " + getComputedStyle(h).position);
for (const b of $$("button")) ok(getComputedStyle(b).borderTopLeftRadius === "8px" && getComputedStyle(b).borderBottomRightRadius === "8px", "button '" + text(b) + "' radius is " + getComputedStyle(b).borderRadius);
const body = document.body.cloneNode(true);
body.querySelectorAll("script, style").forEach((s) => s.remove());
ok(body.textContent.replace(/\\s+/g, " ").trim() === ${JSON.stringify(EXPECTED)}, "the page's text changed");
`,
  mock: [
    xmlCall("read_file", { path: "style.css" }),
    xmlCall("edit_file", { path: "style.css", old: "header { display: flex;", new: "header { position: sticky; top: 0; display: flex;" }),
    xmlCall("edit_file", { path: "style.css", old: "border: 1px solid #889;", new: "border: 1px solid #889; border-radius: 8px;" }),
    "The header is sticky and every button has 8px rounded corners.",
  ],
  bad: { "style.css": css.replace("header {", "header { position: relative;") },
};
