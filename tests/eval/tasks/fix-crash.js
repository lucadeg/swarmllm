import { xmlCall } from "./_util.js";

const BUG = `draw();
const ctx = canvas.getContext("2d");`;
const FIX = `const ctx = canvas.getContext("2d");
draw();`;
const game = `const canvas = document.getElementById("c");
const balls = Array.from({ length: 12 }, (_, i) => ({ x: 30 + i * 28, y: 40 + (i % 4) * 60, vx: 1 + (i % 3), vy: 1 + (i % 2) }));
function draw() {
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f2aa4c";
  for (const b of balls) { ctx.beginPath(); ctx.arc(b.x, b.y, 10, 0, Math.PI * 2); ctx.fill(); }
}
function step() {
  for (const b of balls) {
    b.x += b.vx; b.y += b.vy;
    if (b.x < 10 || b.x > canvas.width - 10) b.vx *= -1;
    if (b.y < 10 || b.y > canvas.height - 10) b.vy *= -1;
  }
  draw();
  requestAnimationFrame(step);
}
${BUG}
requestAnimationFrame(step);
`;
const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>Balls</title><style>body { margin: 0; background: #000; }</style></head>
<body><canvas id="c" width="400" height="300"></canvas><script type="module" src="game.js"></script></body></html>
`;

export default {
  id: "fix-crash",
  kind: "fix",
  prompt: "This bouncing-balls animation only shows a black page. Fix it.",
  files: { "index.html": html, "game.js": game },
  maxSteps: 12,
  check: `
const c = $("canvas");
ok(c, "no <canvas>");
await sleep(200);
ok(drawn(c), "the canvas is blank");
`,
  mock: [
    xmlCall("serve", { port: 5173 }),
    (req) => {
      const r = req.turns[req.turns.length - 1].text;
      return (/ReferenceError|before initialization/.test(r) ? "ctx is used before it is declared.\n" : "Reading the game.\n") + xmlCall("edit_file", { path: "game.js", old: BUG, new: FIX });
    },
    xmlCall("preview_logs", { port: 5173 }),
    "Fixed: draw() ran before ctx was declared (a ReferenceError at load).",
  ],
  bad: {},
};
