import { build } from "./_util.js";

const html = `<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Stopwatch</title>
<style>body { font: 18px system-ui, sans-serif; text-align: center; margin-top: 80px; } #time { font: 56px ui-monospace, monospace; margin-bottom: 16px; } button { font: inherit; margin: 0 4px; }</style></head>
<body>
  <div id="time">00:00</div>
  <button id="start">Start</button><button id="stop">Stop</button><button id="reset">Reset</button>
  <script type="module" src="stopwatch.js"></script>
</body>
</html>
`;
const js = `const time = document.getElementById("time");
let started = 0, acc = 0, timer = 0;
const pad = (n) => String(n).padStart(2, "0");
function elapsed() { return acc + (started ? Date.now() - started : 0); }
function show() { const s = Math.floor(elapsed() / 1000); time.textContent = pad(Math.floor(s / 60)) + ":" + pad(s % 60); }
document.getElementById("start").addEventListener("click", () => {
  if (started) return;
  started = Date.now();
  timer = setInterval(show, 100);
});
document.getElementById("stop").addEventListener("click", () => {
  if (!started) return;
  acc = elapsed(); started = 0;
  clearInterval(timer); show();
});
document.getElementById("reset").addEventListener("click", () => { acc = 0; if (started) started = Date.now(); show(); });
`;
const files = { "index.html": html, "stopwatch.js": js };

export default {
  id: "stopwatch",
  kind: "build, timers",
  prompt: "Build a stopwatch: a display with id 'time' showing mm:ss (00:00 at the start) and buttons Start, Stop and Reset.",
  maxSteps: 14,
  check: `
const t = $("#time");
ok(t, "no element with id time");
const secs = () => { const m = /(\\d+):(\\d+)/.exec(text(t)); ok(m, "#time shows '" + text(t) + "', not mm:ss"); return +m[1] * 60 + +m[2]; };
ok(secs() === 0, "does not start at 00:00");
const B = (l) => { const b = button(l); ok(b, "no button " + l); b.click(); };
B("Start"); await sleep(1600);
ok(secs() >= 1 && secs() <= 2, "1.6 s after Start it shows " + text(t));
B("Stop"); const at = text(t); await sleep(1000);
ok(text(t) === at, "still running after Stop (" + at + " -> " + text(t) + ")");
B("Reset"); await sleep(50);
ok(secs() === 0, "Reset shows " + text(t));
`,
  mock: build(files, "Built a stopwatch with Start, Stop and Reset, served on :5173."),
  bad: { "stopwatch.js": js.replace("acc = elapsed(); started = 0;\n  clearInterval(timer); show();", "show();") },
};
