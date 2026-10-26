import { build, joined } from "./_util.js";

const html = `<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Snake</title>
<style>body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #111; }</style></head>
<body>
  <canvas id="game" width="400" height="400"></canvas>
  <script type="module" src="snake.js"></script>
</body>
</html>
`;
const js = `const N = 20, S = 20;
const ctx = document.getElementById("game").getContext("2d");
let snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }], dir = { x: 1, y: 0 }, food = place(), over = false;
function place() { return { x: Math.floor(Math.random() * N), y: Math.floor(Math.random() * N) }; }
function step() {
  if (over) return;
  const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
  if (head.x < 0 || head.y < 0 || head.x >= N || head.y >= N || snake.some((p) => p.x === head.x && p.y === head.y)) { over = true; draw(); return; }
  snake.unshift(head);
  if (head.x === food.x && head.y === food.y) food = place(); else snake.pop();
  draw();
}
function draw() {
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, N * S, N * S);
  ctx.fillStyle = "#e33"; ctx.fillRect(food.x * S, food.y * S, S - 1, S - 1);
  ctx.fillStyle = "#4c4"; for (const p of snake) ctx.fillRect(p.x * S, p.y * S, S - 1, S - 1);
  if (over) { ctx.fillStyle = "#fff"; ctx.font = "24px sans-serif"; ctx.fillText("Game over", 140, 200); }
}
const DIRS = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
document.addEventListener("keydown", (e) => {
  const d = DIRS[e.key];
  if (d && d[0] !== -dir.x && d[1] !== -dir.y) dir = { x: d[0], y: d[1] };
});
draw();
setInterval(step, 120);
`;
const files = { "index.html": html, "snake.js": js };

export default {
  id: "snake",
  kind: "build, canvas",
  prompt: "Build the Snake game in plain JS on a <canvas>: the snake moves by itself, arrow keys steer it, eating food makes it grow.",
  maxSteps: 16,
  check: `
const c = $("canvas");
ok(c, "no <canvas>");
ok(drawn(c), "the canvas is blank");
const a = shot(c); await sleep(600);
ok(shot(c) !== a, "nothing moved in 600 ms");
// the arrow keys steer: after ArrowUp the change between two pictures is a column (head up, tail
// following), after ArrowLeft a row
const px = () => c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
const moved = (a, b) => { let x0 = 1e9, x1 = -1, y0 = 1e9, y1 = -1; for (let i = 0; i < a.length; i += 4) if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) { const p = i / 4, x = p % c.width, y = (p / c.width) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } return { w: x1 - x0 + 1, h: y1 - y0 + 1 }; };
key("ArrowUp"); await sleep(450);
let p0 = px(); await sleep(300);
const up = moved(p0, px());
ok(up.h > up.w, "ArrowUp did not turn the snake up (the change was " + up.w + "x" + up.h + " px)");
key("ArrowLeft"); await sleep(450);
p0 = px(); await sleep(300);
const left = moved(p0, px());
ok(left.w > left.h, "ArrowLeft did not turn the snake left (the change was " + left.w + "x" + left.h + " px)");
`,
  mock: build(files, "Built Snake on a canvas, served on :5173."),
  bad: [
    { "snake.js": joined(files)["snake.js"].replace("setInterval(step, 120);", "") },
    // moves by itself, ignores the keys
    { "snake.js": joined(files)["snake.js"].replace('document.addEventListener("keydown"', 'document.addEventListener("nokey"') },
  ],
};
