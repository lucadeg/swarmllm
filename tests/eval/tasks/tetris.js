import { build, joined } from "./_util.js";

const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tetris</title>
  <style>body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #14161f; }</style>
</head>
<body>
  <canvas id="board" width="240" height="480"></canvas>
  <script type="module" src="game.js"></script>
</body>
</html>
`;
const part1 = `const COLS = 10, ROWS = 20, S = 24;
const canvas = document.getElementById("board"), ctx = canvas.getContext("2d");
const COLORS = ["#000", "#2b4eff", "#f2b134", "#1a9e5c", "#d1242f", "#8a5cf6", "#18a0b8", "#f06b2b"];
const SHAPES = [[[1, 1, 1, 1]], [[2, 2], [2, 2]], [[0, 3, 0], [3, 3, 3]], [[4, 4, 0], [0, 4, 4]], [[0, 5, 5], [5, 5, 0]], [[6, 0, 0], [6, 6, 6]], [[0, 0, 7], [7, 7, 7]]];
const board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let piece = spawn(), over = false;
function spawn() {
  const m = SHAPES[Math.floor(Math.random() * SHAPES.length)];
  return { m, x: Math.floor((COLS - m[0].length) / 2), y: 0 };
}
function collide(m, x, y) {
  return m.some((row, j) => row.some((v, i) => v && (y + j >= ROWS || x + i < 0 || x + i >= COLS || board[y + j][x + i])));
}
function rotate(m) { return m[0].map((_, i) => m.map((row) => row[i]).reverse()); }
`;
const part2 = `function merge() {
  piece.m.forEach((row, j) => row.forEach((v, i) => { if (v) board[piece.y + j][piece.x + i] = v; }));
  for (let y = ROWS - 1; y >= 0; y--) if (board[y].every(Boolean)) { board.splice(y, 1); board.unshift(Array(COLS).fill(0)); y++; }
  piece = spawn();
  if (collide(piece.m, piece.x, piece.y)) over = true;
}
function cell(x, y, v) { ctx.fillStyle = COLORS[v]; ctx.fillRect(x * S + 1, y * S + 1, S - 2, S - 2); }
function draw() {
  ctx.fillStyle = "#0b0c12"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  board.forEach((row, y) => row.forEach((v, x) => v && cell(x, y, v)));
  piece.m.forEach((row, j) => row.forEach((v, i) => v && cell(piece.x + i, piece.y + j, v)));
}
function tick() {
  if (over) return;
  if (collide(piece.m, piece.x, piece.y + 1)) merge(); else piece.y++;
  draw();
}
document.addEventListener("keydown", (e) => {
  const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
  if (d && !collide(piece.m, piece.x + d, piece.y)) piece.x += d;
  if (e.key === "ArrowDown" && !collide(piece.m, piece.x, piece.y + 1)) piece.y++;
  if (e.key === "ArrowUp") { const r = rotate(piece.m); if (!collide(r, piece.x, piece.y)) piece.m = r; }
  if (e.key === " ") { while (!collide(piece.m, piece.x, piece.y + 1)) piece.y++; merge(); }
  draw();
});
draw();
setInterval(tick, 500);
`;
const files = { "index.html": html, "game.js": [part1, part2] };

export default {
  id: "tetris",
  kind: "build, canvas",
  prompt: "Build a Tetris game in plain JS on a <canvas>: pieces fall on a timer, arrow keys move, ArrowUp rotates, Space drops.",
  maxSteps: 20,
  check: `
await sleep(500);
const c = $("canvas");
ok(c, "no <canvas>");
ok(drawn(c), "the canvas is blank after 500 ms");
// pieces fall on a timer: the picture changes by itself within 2 s
const g0 = shot(c);
let fell = false;
for (let i = 0; i < 20 && !fell; i++) { await sleep(100); fell = shot(c) !== g0; }
ok(fell, "nothing falls by itself in 2 s");
key("ArrowLeft"); key("ArrowUp"); await sleep(50);
// Space drops the piece to the bottom: the lower half changes at once (a gravity tick moves a
// piece that is still high up by one row, which leaves the lower half as it was)
const low = () => { const b = document.createElement("canvas"); b.width = c.width; b.height = c.height >> 1; b.getContext("2d").drawImage(c, 0, -(c.height >> 1)); return b.toDataURL(); };
const before = low();
key(" "); await sleep(50);
ok(low() !== before, "Space (drop) did not change the lower half of the canvas");
`,
  mock: build(files, "Built Tetris: index.html and game.js, served on :5173."),
  bad: [
    // a static picture: drawn, but nothing moves or reacts
    { "game.js": joined(files)["game.js"].replace(/document\.addEventListener[\s\S]*$/, "draw();\n") },
    // falls on a timer, but no keys
    { "game.js": joined(files)["game.js"].replace(/document\.addEventListener[\s\S]*?\n\}\);\n/, "") },
  ],
};
