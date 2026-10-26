import { build } from "./_util.js";

const html = `<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Calculator</title>
<link rel="stylesheet" href="style.css"></head>
<body>
  <div class="calc">
    <div id="display">0</div>
    <div class="keys">
      <button>7</button><button>8</button><button>9</button><button>/</button>
      <button>4</button><button>5</button><button>6</button><button>*</button>
      <button>1</button><button>2</button><button>3</button><button>-</button>
      <button>0</button><button>C</button><button>=</button><button>+</button>
    </div>
  </div>
  <script type="module" src="calc.js"></script>
</body>
</html>
`;
const css = `body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #eef; font: 20px system-ui, sans-serif; }
.calc { width: 240px; padding: 12px; border-radius: 12px; background: #223; }
#display { background: #dfe; padding: 10px; text-align: right; border-radius: 6px; margin-bottom: 10px; min-height: 1.2em; }
.keys { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
button { padding: 12px 0; font: inherit; border: 0; border-radius: 6px; }
`;
const js = `const display = document.getElementById("display");
let expr = "";
function show() { display.textContent = expr || "0"; }
function evaluate() {
  try {
    const v = Function('"use strict"; return (' + (expr || "0") + ")")();
    expr = Number.isFinite(v) ? String(+v.toFixed(10)) : "Error";
  } catch { expr = "Error"; }
}
document.querySelector(".keys").addEventListener("click", (e) => {
  const k = e.target.closest("button")?.textContent;
  if (!k) return;
  if (k === "C") expr = "";
  else if (k === "=") evaluate();
  else expr = (expr === "Error" ? "" : expr) + k;
  show();
});
`;
const files = { "index.html": html, "style.css": css, "calc.js": js };

export default {
  id: "calculator",
  kind: "build, DOM",
  prompt: "Build a calculator: buttons labelled 0-9, + - * /, = and C, and a display with id 'display' that shows the input and the result.",
  maxSteps: 16,
  check: `
const d = $("#display");
ok(d, "no element with id display");
const ALIAS = { "*": ["*", "×", "x"], "/": ["/", "÷"], "-": ["-", "−"], "C": ["C", "AC", "Clear"] };
const B = (l) => { const b = (ALIAS[l] || [l]).map(button).find(Boolean); ok(b, "no button " + l); b.click(); };
for (const l of ["1", "2", "+", "7", "="]) B(l);
await sleep(30);
ok(text(d) === "19", "1 2 + 7 = shows '" + text(d) + "', not 19");
B("C"); await sleep(30);
ok(text(d) === "" || text(d) === "0", "C does not clear (shows '" + text(d) + "')");
for (const l of ["8", "/", "0", "="]) B(l);
B("C");
for (const l of ["9", "*", "3", "-", "4", "="]) B(l);
await sleep(30);
ok(text(d) === "23", "9 * 3 - 4 = shows '" + text(d) + "', not 23");
`,
  mock: build(files, "Built a calculator (calc.js), served on :5173."),
  bad: { "calc.js": js.replace("String(+v.toFixed(10))", "String(Math.round(v) + 1)") },
};
