/* The little apps the Code demo's agent "builds": Tetris, 2048, a space shooter, Snake, Breakout.
   Each one plays itself on a canvas; focus it (click or Tab) and the arrow keys take over, Escape or
   leaving hands it back. v2 is the version after the visitor's change request.
   PooledApps.make(name, canvas) -> { start, stop, draw, reset, warm, v2(on), play, auto, key, human } */
(() => {
  "use strict";
  const TAU = Math.PI * 2;
  const BG = "#0B0F1F", PANEL = "#121834", CELL = "#18204A", DOT = "rgba(147,166,255,.09)";
  const INK = "#EEF0F6", MUTED = "#A9B0C4", ACC = "#A5B4FC";   // ACC: the light accent on the blue ramp (bombs, the ball, its trail)
  const WARM = ["#F08A6C", "#F2C14E", "#6CC5A1", "#B18CF0", "#5EB8E8", "#F28DB2", "#9BD16B"];
  const BLUE = ["#2A45E0", "#6E86FF", "#A5B4FC", "#C9D1F7", "#8EA2FF", "#4A5FD0", "#DCE2FF"];
  const MONO = '"Geist Mono",ui-monospace,Menlo,monospace';
  const rng = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const rr = (c, x, y, w, h, r) => { r = Math.max(0, Math.min(r, w / 2, h / 2)); c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); };
  const label = (c, txt, x, y, al = "left", size = 11, col = MUTED) => { c.font = `400 ${size}px ${MONO}`; c.fillStyle = col; c.textAlign = al; c.textBaseline = "alphabetic"; c.fillText(txt, x, y); };
  const big = (c, txt, x, y, al = "left", size = 17, col = INK) => { c.font = `500 ${size}px ${MONO}`; c.fillStyle = col; c.textAlign = al; c.textBaseline = "alphabetic"; c.fillText(txt, x, y); };
  const fmt = n => n.toLocaleString("en-US");
  const burst = (fx, x, y, col, n, sp = 140) => { for (let i = 0; i < n; i++) { const a = Math.random() * TAU, v = sp * (.3 + Math.random() * .7); fx.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: .5 + Math.random() * .4, t: 0, col, r: 1 + Math.random() * 1.8 }); } };
  const stepFx = (fx, dt, g = 0) => { for (let i = fx.length - 1; i >= 0; i--) { const p = fx[i]; p.t += dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vy += g * dt; p.vx *= .985; p.vy *= .985; if (p.t > p.life) fx.splice(i, 1); } };
  const drawFx = (c, fx) => { fx.forEach(p => { c.globalAlpha = Math.max(0, 1 - p.t / p.life); c.fillStyle = p.col; c.beginPath(); c.arc(p.x, p.y, p.r, 0, TAU); c.fill(); }); c.globalAlpha = 1; };

  /* ---------------- Tetris ---------------- */
  const SH = { I: [[0, 1], [1, 1], [2, 1], [3, 1]], O: [[1, 0], [2, 0], [1, 1], [2, 1]], T: [[1, 0], [0, 1], [1, 1], [2, 1]], S: [[1, 0], [2, 0], [0, 1], [1, 1]], Z: [[0, 0], [1, 0], [1, 1], [2, 1]], J: [[0, 0], [0, 1], [1, 1], [2, 1]], L: [[2, 0], [0, 1], [1, 1], [2, 1]] };
  const KS = Object.keys(SH);
  const rot = (cs, n) => { let c = cs; for (let k = 0; k < n; k++) c = c.map(([x, y]) => [-y, x]); const mx = Math.min(...c.map(p => p[0])), my = Math.min(...c.map(p => p[1])); return c.map(([x, y]) => [x - mx, y - my]); };
  const ROTS = {}; KS.forEach(k => { ROTS[k] = [0, 1, 2, 3].map(n => rot(SH[k], n)); });
  const UNIQ = { I: 2, O: 1, S: 2, Z: 2, T: 4, J: 4, L: 4 };
  const tetris = {
    init(r) {
      const S = { r, bag: [], board: Array.from({ length: 20 }, () => Array(10).fill(0)), score: 0, lines: 0, flash: null, acc: 0, over: 0 };
      this.next(S); return S;
    },
    fits(S, cs, px, py) { return cs.every(([x, y]) => { const X = px + x, Y = py + y; return X >= 0 && X < 10 && Y < 20 && (Y < 0 || !S.board[Y][X]); }); },
    next(S) {
      if (S.bag.length < 2) { const b = KS.slice(); for (let i = b.length - 1; i > 0; i--) { const j = (S.r() * (i + 1)) | 0; [b[i], b[j]] = [b[j], b[i]]; } S.bag.push(...b); }
      const k = S.bag.shift(); S.cur = { k, r: 0, x: 3, y: -1, c: KS.indexOf(k) };
      if (!this.fits(S, ROTS[k][0], 3, -1)) { S.board = Array.from({ length: 20 }, () => Array(10).fill(0)); S.score = 0; S.lines = 0; }
      S.plan = this.think(S, k);
    },
    think(S, k) {
      let best = null;
      for (let r = 0; r < UNIQ[k]; r++) {
        const cs = ROTS[k][r], w = Math.max(...cs.map(p => p[0])) + 1;
        for (let x = 0; x <= 10 - w; x++) {
          let y = -1; if (!this.fits(S, cs, x, y)) continue;
          while (this.fits(S, cs, x, y + 1)) y++;
          const b = S.board.map(row => row.slice()); cs.forEach(([cx, cy]) => { if (y + cy >= 0) b[y + cy][x + cx] = 1; });
          let cl = 0; for (let yy = 0; yy < 20; yy++) if (b[yy].every(v => v)) cl++;
          const h = []; let holes = 0;
          for (let cx = 0; cx < 10; cx++) { let top = 20; for (let yy = 0; yy < 20; yy++) if (b[yy][cx]) { top = yy; break; } h.push(20 - top); for (let yy = top + 1; yy < 20; yy++) if (!b[yy][cx]) holes++; }
          let bump = 0; for (let cx = 0; cx < 9; cx++) bump += Math.abs(h[cx] - h[cx + 1]);
          const s = -.51 * h.reduce((a, v) => a + v, 0) + .76 * cl - .36 * holes - .18 * bump + S.r() * .01;
          if (!best || s > best.s) best = { s, r, x };
        }
      }
      return best || { r: 0, x: 3 };
    },
    cells(S) { return ROTS[S.cur.k][S.cur.r]; },
    lock(S) {
      this.cells(S).forEach(([x, y]) => { if (S.cur.y + y >= 0) S.board[S.cur.y + y][S.cur.x + x] = S.cur.c + 1; });
      const full = []; for (let y = 0; y < 20; y++) if (S.board[y].every(v => v)) full.push(y);
      if (full.length) S.flash = { rows: full, t: 0 }; else this.next(S);
    },
    fall(S) { if (this.fits(S, this.cells(S), S.cur.x, S.cur.y + 1)) { S.cur.y++; return true; } this.lock(S); return false; },
    update(S, dt, human) {
      S.acc += dt;
      if (S.flash) {
        S.flash.t += dt;
        if (S.flash.t > .28) { S.board = S.board.filter((_, y) => !S.flash.rows.includes(y)); while (S.board.length < 20) S.board.unshift(Array(10).fill(0)); S.lines += S.flash.rows.length; S.score += [0, 100, 300, 500, 800][S.flash.rows.length]; S.flash = null; this.next(S); }
        return;
      }
      if (human) { const g = Math.max(.11, .56 - S.lines * .022); while (S.acc >= g && !S.flash) { S.acc -= g; this.fall(S); } return; }
      while (S.acc >= .068 && !S.flash) {
        S.acc -= .068;
        const c = S.cur, p = S.plan;
        if (c.r !== p.r) { const r = (c.r + 1) % 4; if (this.fits(S, ROTS[c.k][r], c.x, c.y)) { c.r = r; continue; } }
        if (c.x !== p.x) { const x = c.x + Math.sign(p.x - c.x); if (this.fits(S, this.cells(S), x, c.y)) { c.x = x; continue; } }
        this.fall(S);
      }
    },
    key(S, k) {
      if (S.flash) return;
      const c = S.cur;
      if (k === "left" || k === "right") { const x = c.x + (k === "left" ? -1 : 1); if (this.fits(S, this.cells(S), x, c.y)) c.x = x; }
      else if (k === "up") { const r = (c.r + 1) % 4; for (const dx of [0, -1, 1, -2, 2]) if (this.fits(S, ROTS[c.k][r], c.x + dx, c.y)) { c.r = r; c.x += dx; break; } }
      else if (k === "down") { if (this.fall(S)) S.score += 1; S.acc = 0; }
      else if (k === "space") { let n = 0; while (this.fits(S, this.cells(S), c.x, c.y + 1)) { c.y++; n++; } S.score += n * 2; this.lock(S); S.acc = 0; }
    },
    render(c, W, H, S, v2) {
      const COL = v2 ? BLUE : WARM;
      c.fillStyle = BG; c.fillRect(0, 0, W, H);
      const side = W < 300 ? 70 : 88, gap = 18;
      const s = Math.floor(Math.min((H - 36) / 20, (W - 32 - side - gap) / 10));
      const bw = s * 10, bh = s * 20, ox = Math.round((W - bw - gap - side) / 2), oy = Math.round((H - bh) / 2);
      rr(c, ox - 4, oy - 4, bw + 8, bh + 8, 8); c.fillStyle = "#0E1430"; c.fill(); c.strokeStyle = "#1E2650"; c.lineWidth = 1; c.stroke();
      c.fillStyle = DOT; for (let y = 0; y < 20; y++) for (let x = 0; x < 10; x++) c.fillRect(ox + x * s + s / 2 - .75, oy + y * s + s / 2 - .75, 1.5, 1.5);
      const cell = (x, y, col, a = 1) => { c.globalAlpha = a; rr(c, ox + x * s + 1, oy + y * s + 1, s - 2, s - 2, Math.max(2, s * .16)); c.fillStyle = col; c.fill(); c.fillStyle = "rgba(255,255,255,.16)"; c.fillRect(ox + x * s + 3, oy + y * s + 2, s - 6, Math.max(1.5, s * .1)); c.globalAlpha = 1; };
      for (let y = 0; y < 20; y++) for (let x = 0; x < 10; x++) { const v = S.board[y][x]; if (!v) continue; const f = S.flash && S.flash.rows.includes(y); cell(x, y, f ? "#FFFFFF" : COL[v - 1], f ? (Math.floor(S.flash.t * 20) % 2 ? .95 : .55) : 1); }
      if (!S.flash) {
        const cs = this.cells(S); let gy = S.cur.y; while (this.fits(S, cs, S.cur.x, gy + 1)) gy++;
        c.strokeStyle = "rgba(147,166,255,.35)"; c.lineWidth = 1;
        cs.forEach(([x, y]) => { if (gy + y >= 0) { rr(c, ox + (S.cur.x + x) * s + 1.5, oy + (gy + y) * s + 1.5, s - 3, s - 3, 3); c.stroke(); } });
        cs.forEach(([x, y]) => { if (S.cur.y + y >= 0) cell(S.cur.x + x, S.cur.y + y, COL[S.cur.c]); });
      }
      const sx = ox + bw + gap; let sy = oy + 12;
      label(c, "SCORE", sx, sy); big(c, fmt(S.score), sx, sy + 22); sy += 50;
      label(c, "LINES", sx, sy); big(c, String(S.lines), sx, sy + 22); sy += 50;
      if (v2) {
        label(c, "NEXT", sx, sy); sy += 10;
        rr(c, sx, sy, side - 8, side - 8, 8); c.fillStyle = "#0E1430"; c.fill(); c.strokeStyle = "#1E2650"; c.stroke();
        const k = S.bag[0]; if (k) { const cs = ROTS[k][0], cw = Math.max(...cs.map(p => p[0])) + 1, ch = Math.max(...cs.map(p => p[1])) + 1, q = Math.floor((side - 8) / 5.2); cs.forEach(([x, y]) => { rr(c, sx + (side - 8 - cw * q) / 2 + x * q + 1, sy + (side - 8 - ch * q) / 2 + y * q + 1, q - 2, q - 2, 2); c.fillStyle = COL[KS.indexOf(k)]; c.fill(); }); }
      }
    }
  };

  /* ---------------- 2048 ---------------- */
  const T_WARM = { 2: ["#EEE4DA", "#3A3330"], 4: ["#EDE0C8", "#3A3330"], 8: ["#F2B179", "#fff"], 16: ["#F59563", "#fff"], 32: ["#F67C5F", "#fff"], 64: ["#F65E3B", "#fff"], 128: ["#EDCF72", "#fff"], 256: ["#EDCC61", "#fff"], 512: ["#EDC850", "#fff"], 1024: ["#EDC53F", "#fff"], 2048: ["#EDC22E", "#fff"] };
  const T_BLUE = { 2: ["#DCE2FF", "#1B2150"], 4: ["#C9D1F7", "#1B2150"], 8: ["#A5B4FC", "#1B2150"], 16: ["#8EA2FF", "#0B0F1F"], 32: ["#6E86FF", "#fff"], 64: ["#5B74FF", "#fff"], 128: ["#4A5FD0", "#fff"], 256: ["#2A45E0", "#fff"], 512: ["#2239C8", "#fff"], 1024: ["#1C33B8", "#fff"], 2048: ["#FFFFFF", "#2A45E0"] };
  const g2048 = {
    init(r) { const S = { r, tiles: [], id: 0, score: 0, best: 0, anim: 1, acc: 0, over: 0 }; this.spawn(S); this.spawn(S); S.tiles.forEach(t => { t.born = 1; }); return S; },
    grid(S) { const g = Array.from({ length: 4 }, () => Array(4).fill(null)); S.tiles.forEach(t => { if (!t.dead) g[t.y][t.x] = t; }); return g; },
    spawn(S) { const g = this.grid(S), e = []; for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) if (!g[y][x]) e.push([x, y]); if (!e.length) return; const [x, y] = e[(S.r() * e.length) | 0]; S.tiles.push({ id: S.id++, v: S.r() < .9 ? 2 : 4, x, y, px: x, py: y, born: 0, pop: 1 }); },
    // slide on a plain value grid; returns [grid, gained, moved]
    sim(vals, d) {
      const g = vals.map(r => r.slice()); let gained = 0, moved = false;
      for (let i = 0; i < 4; i++) {
        const line = []; for (let j = 0; j < 4; j++) { const [x, y] = this.at(d, i, j); line.push(g[y][x]); }
        const nz = line.filter(Boolean), out = [];
        for (let k = 0; k < nz.length; k++) { if (nz[k] === nz[k + 1]) { out.push(nz[k] * 2); gained += nz[k] * 2; k++; } else out.push(nz[k]); }
        while (out.length < 4) out.push(0);
        for (let j = 0; j < 4; j++) { const [x, y] = this.at(d, i, j); if (g[y][x] !== out[j]) moved = true; g[y][x] = out[j]; }
      }
      return [g, gained, moved];
    },
    at(d, i, j) { return d === "left" ? [j, i] : d === "right" ? [3 - j, i] : d === "up" ? [i, j] : [i, 3 - j]; },
    vals(S) { const g = this.grid(S); return g.map(r => r.map(t => t ? t.v : 0)); },
    move(S, d) {
      const g = this.grid(S); let moved = false;
      S.tiles = S.tiles.filter(t => !t.dead);
      S.tiles.forEach(t => { t.px = t.x; t.py = t.y; t.born = 1; t.pop = 1; t.fresh = false; });
      for (let i = 0; i < 4; i++) {
        const line = []; for (let j = 0; j < 4; j++) { const [x, y] = this.at(d, i, j); if (g[y][x]) line.push(g[y][x]); }
        let j = 0;
        for (let k = 0; k < line.length; k++) {
          const t = line[k], [x, y] = this.at(d, i, j);
          if (line[k + 1] && line[k + 1].v === t.v) {
            const u = line[k + 1]; [t, u].forEach(q => { if (q.x !== x || q.y !== y) moved = true; q.x = x; q.y = y; q.dead = true; });
            S.tiles.push({ id: S.id++, v: t.v * 2, x, y, px: x, py: y, born: 1, pop: 0, merged: true, fresh: true }); S.score += t.v * 2; k++;
          } else { if (t.x !== x || t.y !== y) moved = true; t.x = x; t.y = y; }
          j++;
        }
      }
      if (!moved) { S.tiles.forEach(t => { t.dead = false; }); S.tiles = S.tiles.filter(t => !t.fresh); return false; }
      S.best = Math.max(S.best, S.score); S.anim = 0; S.pending = true; return true;
    },
    think(S) {
      const v = this.vals(S); let best = null;
      for (const d of ["down", "left", "right", "up"]) {
        const [g, gained, moved] = this.sim(v, d); if (!moved) continue;
        let empty = 0, mono = 0, corner = 0; g.forEach(r => r.forEach(x => { if (!x) empty++; }));
        for (let y = 0; y < 4; y++) for (let x = 0; x < 3; x++) { if (g[y][x] >= g[y][x + 1]) mono += Math.log2(g[y][x] || 1); if (g[x][y] <= g[x + 1][y]) mono += Math.log2(g[x + 1][y] || 1); }
        const mx = Math.max(...g.flat()); if (g[3][0] === mx) corner = Math.log2(mx) * 4;
        const s = empty * 3 + Math.log2(gained + 1) + mono * .25 + corner - (d === "up" ? 12 : 0) + S.r() * .2;
        if (!best || s > best.s) best = { s, d };
      }
      return best && best.d;
    },
    update(S, dt, human) {
      if (S.anim < 1) { S.anim = Math.min(1, S.anim + dt / .11); if (S.anim >= 1 && S.pending) { S.pending = false; S.tiles = S.tiles.filter(t => !t.dead); this.spawn(S); } }
      S.tiles.forEach(t => { if (S.anim >= 1) { t.born = Math.min(1, t.born + dt / .14); t.pop = Math.min(1, t.pop + dt / .16); } });
      if (S.over) { S.over -= dt; if (S.over <= 0) { const best = S.best; Object.assign(S, this.init(S.r)); S.best = best; } return; }
      if (S.anim < 1) return;
      const stuck = !["down", "left", "right", "up"].some(d => this.sim(this.vals(S), d)[2]);
      if (stuck) { S.over = 1.4; return; }
      if (human) return;
      S.acc += dt; if (S.acc < .22) return; S.acc = 0;
      const d = this.think(S); if (d) this.move(S, d);
    },
    key(S, k) { if (S.anim < 1 || S.over) return; const d = { left: "left", right: "right", up: "up", down: "down" }[k]; if (d) this.move(S, d); },
    render(c, W, H, S, v2) {
      c.fillStyle = BG; c.fillRect(0, 0, W, H);
      const P = v2 ? T_BLUE : T_WARM;
      const size = Math.min(W - 40, H - 110, 360), gap = Math.round(size * .03), cs = (size - gap * 5) / 4;
      const ox = Math.round((W - size) / 2), oy = Math.round((H - size) / 2 + 30);
      // header: title, score, best
      big(c, "2048", ox, oy - 34, "left", Math.round(Math.min(34, size * .1)), INK);
      const box = (lbl, val, right) => { const w = 76, h = 42, x = right - w; rr(c, x, oy - 62, w, h, 8); c.fillStyle = PANEL; c.fill(); label(c, lbl, x + w / 2, oy - 46, "center", 10); big(c, fmt(val), x + w / 2, oy - 27, "center", 15); return x - 8; };
      let r = box("SCORE", S.score, ox + size);
      if (v2) box("BEST", Math.max(S.best, S.score), r);
      rr(c, ox, oy, size, size, 12); c.fillStyle = "#141B3C"; c.fill();
      for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { rr(c, ox + gap + x * (cs + gap), oy + gap + y * (cs + gap), cs, cs, 8); c.fillStyle = CELL; c.fill(); }
      const e = 1 - Math.pow(1 - S.anim, 3);
      const order = S.tiles.slice().sort((a, b) => (a.merged ? 1 : 0) - (b.merged ? 1 : 0));
      order.forEach(t => {
        if (t.fresh && S.anim < 1) return;
        if (t.dead && S.anim >= 1) return;
        const gx = t.px + (t.x - t.px) * e, gy = t.py + (t.y - t.py) * e;
        let sc = 1; if (t.born < 1) sc = .3 + .7 * (1 - Math.pow(1 - t.born, 3)); if (t.merged && t.pop < 1) sc = 1 + .14 * Math.sin(t.pop * Math.PI);
        const [bg, fg] = P[t.v] || P[2048];
        const cx = ox + gap + gx * (cs + gap) + cs / 2, cy = oy + gap + gy * (cs + gap) + cs / 2, w = cs * sc;
        if (v2 && t.v >= 128) { c.shadowColor = "rgba(90,116,255,.55)"; c.shadowBlur = 18; }
        rr(c, cx - w / 2, cy - w / 2, w, w, 8); c.fillStyle = bg; c.fill(); c.shadowBlur = 0;
        const fs = Math.round(cs * (t.v < 100 ? .42 : t.v < 1000 ? .34 : .27) * sc);
        c.font = `600 ${fs}px "Geist",ui-sans-serif,sans-serif`; c.fillStyle = fg; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(String(t.v), cx, cy + 1);
      });
      if (S.over) { c.fillStyle = "rgba(11,15,31,.6)"; rr(c, ox, oy, size, size, 12); c.fill(); big(c, "No moves left", W / 2, oy + size / 2 + 6, "center", 18); }
    }
  };

  /* ---------------- Space shooter ---------------- */
  const INV = [["..X..X..", "...XX...", "..XXXX..", ".XX..XX.", "XXXXXXXX", "X.X..X.X"], ["...XX...", "..XXXX..", ".XXXXXX.", "XX.XX.XX", "XXXXXXXX", ".X.XX.X."], ["X......X", ".X.XX.X.", ".XXXXXX.", "XX.XX.XX", "XXXXXXXX", "X.X..X.X"]];
  const ROWC = ["#B9C6FF", "#7C8FFF", "#2A45E0", "#7C8FFF", "#B9C6FF"];   // the blue ramp; the bombs are the light accent ACC
  const shooter = {
    init(r) {
      const S = { r, t: 0, ship: .5, vx: 0, shots: [], bombs: [], fx: [], score: 0, lives: 3, wave: 0, cool: 0, hit: 0, stars: [] };
      for (let i = 0; i < 90; i++) S.stars.push({ x: r(), y: r(), z: .2 + r() * .8 });
      this.wave(S); return S;
    },
    wave(S) { S.wave++; S.en = []; for (let y = 0; y < 4; y++) for (let x = 0; x < 7; x++) S.en.push({ gx: x, gy: y, alive: true, blink: 0 }); S.fx.length = 0; },
    geo(S, W, H) { const u = Math.min(W / 300, H / 360); const cw = 36 * u, ch = 30 * u, fw = cw * 7; const ox = (W - fw) / 2 + Math.sin(S.t * .7) * (W - fw) * .38, oy = 64 * u + Math.sin(S.t * 1.3) * 4; return { u, cw, ch, ox, oy }; },
    update(S, dt, human, W, H) {
      S.t += dt; S.cool -= dt; S.hit = Math.max(0, S.hit - dt);
      if (!W) return;
      const G = this.geo(S, W, H), shipY = H - 34 * G.u, sw = 22 * G.u;
      S.stars.forEach(s => { s.y += dt * .05 * (s.z * 2 + .2); if (s.y > 1) { s.y -= 1; s.x = S.r(); } });
      // the ship
      if (!human) {
        const alive = S.en.filter(e => e.alive); let tx = .5;
        if (alive.length) { const e = alive.reduce((a, b) => Math.abs(G.ox + (a.gx + .5) * G.cw - S.ship * W) < Math.abs(G.ox + (b.gx + .5) * G.cw - S.ship * W) ? a : b); tx = (G.ox + (e.gx + .5) * G.cw) / W; }
        let dodge = 0; S.bombs.forEach(b => { if (b.y > shipY - 90 * G.u && Math.abs(b.x - S.ship * W) < 20 * G.u) dodge += b.x > S.ship * W ? -1 : 1; });
        if (dodge) tx = S.ship + dodge * .12;
        S.vx = Math.max(-.9, Math.min(.9, (tx - S.ship) * 6));
        if (S.cool <= 0 && Math.abs(tx - S.ship) < .06 && !dodge) this.fire(S, W, shipY);
      }
      S.ship = Math.max(.06, Math.min(.94, S.ship + S.vx * dt));
      S.shots.forEach(s => { s.y -= 520 * G.u * dt; }); S.shots = S.shots.filter(s => s.y > -10);
      S.bombs.forEach(b => { b.y += 170 * G.u * dt; }); S.bombs = S.bombs.filter(b => b.y < H + 10);
      // hits on the invaders
      S.shots.forEach(s => {
        S.en.forEach(e => {
          if (!e.alive || s.dead) return;
          const ex = G.ox + e.gx * G.cw + G.cw / 2, ey = G.oy + e.gy * G.ch + G.ch / 2;
          if (Math.abs(s.x - ex) < G.cw * .42 && Math.abs(s.y - ey) < G.ch * .45) { e.alive = false; s.dead = true; S.score += [40, 30, 20, 10][e.gy]; if (S.v2) burst(S.fx, ex, ey, ROWC[e.gy], 18, 150 * G.u); else S.fx.push({ x: ex, y: ey, vx: 0, vy: 0, life: .18, t: 0, col: ROWC[e.gy], r: 5 * G.u }); }
        });
      });
      S.shots = S.shots.filter(s => !s.dead);
      // they fire back
      const alive = S.en.filter(e => e.alive);
      if (alive.length && S.r() < dt * 1.3) { const e = alive[(S.r() * alive.length) | 0]; S.bombs.push({ x: G.ox + e.gx * G.cw + G.cw / 2, y: G.oy + e.gy * G.ch + G.ch }); }
      S.bombs.forEach(b => { if (!S.hit && Math.abs(b.x - S.ship * W) < sw * .5 && Math.abs(b.y - shipY) < 12 * G.u) { b.y = H + 20; S.hit = 1.2; S.lives--; if (S.v2) burst(S.fx, S.ship * W, shipY, ACC, 22, 160 * G.u); } });
      if (S.lives <= 0) { const v2 = S.v2; Object.assign(S, this.init(S.r)); S.v2 = v2; }
      if (!alive.length) this.wave(S);
      stepFx(S.fx, dt);
    },
    fire(S, W, shipY) { S.shots.push({ x: S.ship * W, y: shipY - 12 }); S.cool = .3; },
    key(S, k, down, W, H) {
      if (k === "left" || k === "right") S.vx = down ? (k === "left" ? -.8 : .8) : 0;
      if (k === "space" && down && S.cool <= 0 && W) this.fire(S, W, H - 34 * this.geo(S, W, H).u);
    },
    render(c, W, H, S, v2) {
      S.v2 = v2;
      c.fillStyle = BG; c.fillRect(0, 0, W, H);
      if (v2) {
        const g = c.createRadialGradient(W * .5, H * .15, 0, W * .5, H * .15, H * .9); g.addColorStop(0, "rgba(42,69,224,.20)"); g.addColorStop(1, "rgba(42,69,224,0)"); c.fillStyle = g; c.fillRect(0, 0, W, H);
        S.stars.forEach(s => { c.globalAlpha = .25 + s.z * .6; c.fillStyle = s.z > .8 ? "#DCE2FF" : "#8EA2FF"; const r = .5 + s.z * 1.1; c.fillRect(s.x * W, s.y * H, r, r * (1 + s.z * 2)); }); c.globalAlpha = 1;
      }
      const G = this.geo(S, W, H), px = Math.max(2, Math.floor(G.cw / 10));
      S.en.forEach(e => {
        if (!e.alive) return;
        const art = INV[e.gy % 3], frame = Math.floor(S.t * 2) % 2;
        const x0 = G.ox + e.gx * G.cw + (G.cw - px * 8) / 2, y0 = G.oy + e.gy * G.ch + (G.ch - px * 6) / 2;
        c.fillStyle = ROWC[e.gy];
        art.forEach((row, yy) => { for (let xx = 0; xx < 8; xx++) if (row[xx] === "X" && !(frame && yy === 5)) c.fillRect(x0 + xx * px, y0 + yy * px, px, px); });
        if (frame) { c.fillRect(x0, y0 + 4 * px, px, px); c.fillRect(x0 + 7 * px, y0 + 4 * px, px, px); }
      });
      drawFx(c, S.fx);
      c.fillStyle = "#B9C6FF"; S.shots.forEach(s => { rr(c, s.x - 1.5, s.y - 8, 3, 12, 1.5); c.fill(); });
      c.fillStyle = ACC; S.bombs.forEach(b => { c.beginPath(); c.arc(b.x, b.y, 3, 0, TAU); c.fill(); });
      // the ship
      const sx = S.ship * W, sy = H - 34 * G.u, u = G.u;
      if (!(S.hit && Math.floor(S.t * 12) % 2)) {
        c.fillStyle = v2 ? "#6E86FF" : "#8EA2FF"; c.beginPath(); c.ellipse(sx, sy + 13 * u, 3.5 * u, (5 + Math.sin(S.t * 40) * 2) * u, 0, 0, TAU); c.fill();
        c.fillStyle = INK; c.beginPath(); c.moveTo(sx, sy - 13 * u); c.lineTo(sx + 11 * u, sy + 9 * u); c.lineTo(sx + 4 * u, sy + 6 * u); c.lineTo(sx - 4 * u, sy + 6 * u); c.lineTo(sx - 11 * u, sy + 9 * u); c.closePath(); c.fill();
        c.fillStyle = "#2A45E0"; c.beginPath(); c.arc(sx, sy - 1 * u, 2.6 * u, 0, TAU); c.fill();
      }
      label(c, "SCORE", 16, 24); big(c, fmt(S.score), 16, 44, "left", 16);
      label(c, "WAVE " + S.wave, W - 16, 24, "right");
      for (let i = 0; i < S.lives; i++) { const x = W - 22 - i * 16, y = 38; c.fillStyle = MUTED; c.beginPath(); c.moveTo(x, y - 5); c.lineTo(x + 5, y + 4); c.lineTo(x - 5, y + 4); c.closePath(); c.fill(); }
    }
  };

  /* ---------------- Snake ---------------- */
  const SC = 16, SR = 18, DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
  const snake = {
    init(r) { const S = { r, body: [[5, 9], [4, 9], [3, 9]], dir: [1, 0], q: [], acc: 0, score: 0, dead: 0, t: 0, prev: null }; this.food(S); return S; },
    food(S) { const occ = new Set(S.body.map(p => p + "")); const e = []; for (let y = 0; y < SR; y++) for (let x = 0; x < SC; x++) if (!occ.has(x + "," + y)) e.push([x, y]); S.food = e[(S.r() * e.length) | 0]; S.born = 0; },
    free(S, x, y, body = S.body) { if (x < 0 || y < 0 || x >= SC || y >= SR) return false; for (let i = 0; i < body.length - 1; i++) if (body[i][0] === x && body[i][1] === y) return false; return true; },
    bfs(S, from, to, body) {
      const prev = new Map(), k = p => p[0] + "," + p[1], q = [from]; prev.set(k(from), null);
      while (q.length) { const p = q.shift(); if (p[0] === to[0] && p[1] === to[1]) { const path = []; let c = p; while (prev.get(k(c))) { path.unshift(c); c = prev.get(k(c)); } return path; } for (const d of Object.values(DIRS)) { const n = [p[0] + d[0], p[1] + d[1]]; if (prev.has(k(n)) || !this.free(S, n[0], n[1], body)) continue; prev.set(k(n), p); q.push(n); } }
      return null;
    },
    area(S, start, body) { const seen = new Set([start + ""]), q = [start]; let n = 0; while (q.length && n < 400) { const p = q.pop(); n++; for (const d of Object.values(DIRS)) { const m = [p[0] + d[0], p[1] + d[1]]; if (seen.has(m + "") || !this.free(S, m[0], m[1], body)) continue; seen.add(m + ""); q.push(m); } } return n; },
    think(S) {
      const h = S.body[0], path = this.bfs(S, h, S.food, S.body);
      if (path) {
        // is it safe? follow the path on a copy and check the tail is still reachable
        let b = S.body.map(p => p.slice()); path.forEach((p, i) => { b.unshift(p); if (i < path.length - 1) b.pop(); });
        if (this.bfs(S, b[0], b[b.length - 1], b) || this.area(S, b[0], b) > b.length) return [path[0][0] - h[0], path[0][1] - h[1]];
      }
      let best = null;
      for (const d of Object.values(DIRS)) { const n = [h[0] + d[0], h[1] + d[1]]; if (!this.free(S, n[0], n[1])) continue; const b = [n, ...S.body.slice(0, -1)]; const a = this.area(S, n, b) + (this.bfs(S, n, b[b.length - 1], b) ? 50 : 0); if (!best || a > best.a) best = { a, d }; }
      return best ? best.d : S.dir;
    },
    update(S, dt, human) {
      S.t += dt; S.born = Math.min(1, (S.born || 0) + dt / .25);
      if (S.dead) { S.dead -= dt; if (S.dead <= 0) Object.assign(S, this.init(S.r)); return; }
      const tick = S.v2 ? Math.max(.055, .1 - S.body.length * .0012) : .095;
      S.acc += dt;
      while (S.acc >= tick && !S.dead) {
        S.acc -= tick;
        if (human) { const d = S.q.shift(); if (d && !(d[0] === -S.dir[0] && d[1] === -S.dir[1])) S.dir = d; } else S.dir = this.think(S);
        const h = S.body[0], n = [h[0] + S.dir[0], h[1] + S.dir[1]];
        if (!this.free(S, n[0], n[1])) { S.dead = 1.1; return; }
        S.prev = S.body.map(p => p.slice());
        S.body.unshift(n);
        if (n[0] === S.food[0] && n[1] === S.food[1]) { S.score += 10; if (S.body.length >= SC * SR - 2) { S.dead = 1; return; } this.food(S); } else S.body.pop();
      }
      S.frac = S.acc / tick;
    },
    key(S, k) { const d = DIRS[k]; if (d && S.q.length < 3) S.q.push(d); },
    render(c, W, H, S, v2) {
      S.v2 = v2;
      c.fillStyle = BG; c.fillRect(0, 0, W, H);
      const s = Math.floor(Math.min((W - 32) / SC, (H - 76) / SR)), bw = s * SC, bh = s * SR, ox = Math.round((W - bw) / 2), oy = Math.round((H - bh) / 2 + 20);
      label(c, "SCORE", ox, oy - 30); big(c, fmt(S.score), ox, oy - 11, "left", 16);
      label(c, "LENGTH " + S.body.length, ox + bw, oy - 11, "right");
      rr(c, ox - 4, oy - 4, bw + 8, bh + 8, 10); c.fillStyle = "#0E1430"; c.fill(); c.strokeStyle = "#1E2650"; c.lineWidth = 1; c.stroke();
      c.fillStyle = DOT; for (let y = 0; y < SR; y++) for (let x = 0; x < SC; x++) c.fillRect(ox + x * s + s / 2 - .75, oy + y * s + s / 2 - .75, 1.5, 1.5);
      // food
      const fx = ox + S.food[0] * s + s / 2, fy = oy + S.food[1] * s + s / 2, pulse = 1 + Math.sin(S.t * 6) * .08, fr = s * .34 * pulse * (.4 + .6 * S.born);
      if (v2) { const g = c.createRadialGradient(fx, fy, 0, fx, fy, s * 1.3); g.addColorStop(0, "rgba(242,193,78,.45)"); g.addColorStop(1, "rgba(242,193,78,0)"); c.fillStyle = g; c.fillRect(fx - s * 1.3, fy - s * 1.3, s * 2.6, s * 2.6); }
      c.fillStyle = v2 ? "#F2C14E" : "#6CC5A1"; c.beginPath(); c.arc(fx, fy, fr, 0, TAU); c.fill();
      // the snake
      const n = S.body.length, dead = S.dead && Math.floor(S.dead * 10) % 2;
      for (let i = n - 1; i >= 0; i--) {
        const p = S.body[i], x = ox + p[0] * s, y = oy + p[1] * s;
        let col;
        if (v2) { const k = i / Math.max(1, n - 1); const a = [42, 69, 224], b = [165, 180, 252]; col = `rgb(${a.map((v, j) => Math.round(v + (b[j] - v) * k)).join(",")})`; }
        else col = i === 0 ? "#F6A58B" : "#F08A6C";
        c.fillStyle = dead ? "#FFFFFF" : col;
        if (v2) {
          rr(c, x + 1, y + 1, s - 2, s - 2, s * .38); c.fill();
          const nx = S.body[i + 1]; if (nx) { const mx = (p[0] + nx[0]) / 2, my = (p[1] + nx[1]) / 2; rr(c, ox + mx * s + 1, oy + my * s + 1, s - 2, s - 2, s * .3); c.fill(); }
        } else { rr(c, x + 1.5, y + 1.5, s - 3, s - 3, 3); c.fill(); }
      }
      // eyes
      const h = S.body[0], hx = ox + h[0] * s + s / 2, hy = oy + h[1] * s + s / 2, d = S.dir;
      c.fillStyle = v2 ? "#FFFFFF" : "#3A1D14";
      [-1, 1].forEach(k => { c.beginPath(); c.arc(hx + d[0] * s * .16 + d[1] * k * s * .2, hy + d[1] * s * .16 - d[0] * k * s * .2, Math.max(1.5, s * .09), 0, TAU); c.fill(); });
    }
  };

  /* ---------------- Breakout ---------------- */
  const BRC = ["#B9C6FF", "#7C8FFF", "#2A45E0", "#1C33B8", "#2A45E0", "#7C8FFF"];   // the blue ramp; the ball is the light accent ACC
  const breakout = {
    init(r) { const S = { r, px: .5, pv: 0, score: 0, lives: 3, level: 1, fx: [], trail: [], t: 0, aim: 0 }; this.bricks(S); this.serve(S); return S; },
    bricks(S) { S.br = []; for (let y = 0; y < 6; y++) for (let x = 0; x < 9; x++) S.br.push({ x, y, alive: true }); },
    serve(S) { S.b = { x: .5, y: .72, vx: (S.r() < .5 ? -1 : 1) * .32, vy: -.62 }; S.wait = .5; S.aim = (S.r() - .5) * .6; },
    update(S, dt, human, W, H) {
      if (!W) return;
      S.t += dt; stepFx(S.fx, dt, 260);
      const A = W / H, pw = .2, py = .9;
      if (!human) {
        // predict where the ball comes down, then offset the paddle a little to aim
        let x = S.b.x, y = S.b.y, vx = S.b.vx, vy = S.b.vy;
        if (vy > 0) { let t = (py - y) / vy; x += vx * t * (1 / A) * A; while (x < 0 || x > 1) x = x < 0 ? -x : 2 - x; }
        else x = S.b.x;
        const tgt = x - S.aim * pw * .5;
        S.pv = Math.max(-1.6, Math.min(1.6, (tgt - S.px) * 9));
      }
      S.px = Math.max(pw / 2, Math.min(1 - pw / 2, S.px + S.pv * dt));
      if (S.wait > 0) { S.wait -= dt; S.b.x = S.px; S.b.y = py - .03; return; }
      const b = S.b, steps = 4;
      for (let k = 0; k < steps; k++) {
        const h = dt / steps; b.x += b.vx * h; b.y += b.vy * h * A;
        if (b.x < .012) { b.x = .012; b.vx = Math.abs(b.vx); } if (b.x > .988) { b.x = .988; b.vx = -Math.abs(b.vx); }
        if (b.y < .1) { b.y = .1; b.vy = Math.abs(b.vy); }
        if (b.vy > 0 && b.y > py - .012 && b.y < py + .02 && Math.abs(b.x - S.px) < pw / 2 + .015) {
          const off = (b.x - S.px) / (pw / 2), sp = Math.hypot(b.vx, b.vy) * 1.004; b.vx = sp * Math.sin(off * 1.0); b.vy = -Math.abs(sp * Math.cos(off * 1.0)); S.aim = (S.r() - .5) * 1.2;
        }
        const bw = 1 / 9, bh = .038, top = .16;
        for (const r of S.br) {
          if (!r.alive) continue;
          const x0 = r.x * bw, y0 = top + r.y * bh;
          if (b.x > x0 && b.x < x0 + bw && b.y > y0 && b.y < y0 + bh) {
            r.alive = false; b.vy = -b.vy; S.score += (6 - r.y) * 10;
            if (S.v2) burst(S.fx, (14 + (x0 + bw / 2) * (W - 28)), (y0 + bh / 2) * H, BRC[r.y], 14, 120);
            break;
          }
        }
        if (b.y > 1.02) { S.lives--; if (S.lives <= 0) { const v2 = S.v2; Object.assign(S, this.init(S.r)); S.v2 = v2; return; } this.serve(S); return; }
      }
      if (S.v2) { S.trail.push([b.x, b.y]); if (S.trail.length > 12) S.trail.shift(); } else S.trail.length = 0;
      if (!S.br.some(r => r.alive)) { S.level++; this.bricks(S); this.serve(S); }
    },
    key(S, k, down) { if (k === "left" || k === "right") S.pv = down ? (k === "left" ? -1.1 : 1.1) : 0; },
    pointer(S, x) { S.px = Math.max(.1, Math.min(.9, x)); S.pv = 0; },
    render(c, W, H, S, v2) {
      S.v2 = v2;
      c.fillStyle = BG; c.fillRect(0, 0, W, H);
      const mx = 14, bw = (W - mx * 2) / 9, bh = H * .038, top = H * .16;
      S.br.forEach(r => { if (!r.alive) return; rr(c, mx + r.x * bw + 2, top + r.y * bh + 2, bw - 4, bh - 4, 3); c.fillStyle = BRC[r.y]; c.fill(); if (v2) { c.fillStyle = "rgba(255,255,255,.18)"; c.fillRect(mx + r.x * bw + 4, top + r.y * bh + 3, bw - 8, 2); } });
      drawFx(c, S.fx);
      const X = x => mx + x * (W - mx * 2), pw = .2 * (W - mx * 2), py = .9 * H;
      rr(c, X(S.px) - pw / 2, py, pw, 8, 4); c.fillStyle = INK; c.fill();
      const bx = X(S.b.x), by = S.b.y * H;
      if (v2) {
        S.trail.forEach(([x, y], i) => { c.globalAlpha = i / S.trail.length * .5; c.fillStyle = ACC; c.beginPath(); c.arc(X(x), y * H, 5 * i / S.trail.length + 1, 0, TAU); c.fill(); }); c.globalAlpha = 1;
        c.shadowColor = "rgba(224,138,42,.8)"; c.shadowBlur = 16;
      }
      c.fillStyle = ACC; c.beginPath(); c.arc(bx, by, 5.5, 0, TAU); c.fill(); c.shadowBlur = 0;
      label(c, "SCORE", 16, 24); big(c, fmt(S.score), 16, 44, "left", 16);
      label(c, "LEVEL " + S.level, W - 16, 24, "right");
      for (let i = 0; i < S.lives; i++) { c.fillStyle = MUTED; c.beginPath(); c.arc(W - 20 - i * 14, 40, 4, 0, TAU); c.fill(); }
    }
  };

  shooter.hold = breakout.hold = true;
  const APPS = { tetris, "2048": g2048, shooter, snake, breakout };
  const KEYS = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down", " ": "space", Spacebar: "space", a: "left", d: "right", w: "up", s: "down" };

  function make(name, canvas, opts = {}) {
    const A = APPS[name], ctx = canvas.getContext("2d");
    let S = null, v2 = false, human = false, playing = false, visible = true, raf = 0, last = 0, W = 0, H = 0, dpr = 1, seed = opts.seed || 7;
    const size = () => {
      const r = canvas.getBoundingClientRect(); if (!r.width || !r.height) return false;
      dpr = Math.min(devicePixelRatio || 1, 2); W = r.width; H = r.height;
      const cw = Math.round(W * dpr), ch = Math.round(H * dpr);
      if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
      return true;
    };
    const reset = s => { if (s != null) seed = s; S = A.init(rng(seed)); S.v2 = v2; };
    const draw = () => { if (!size()) return; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); A.render(ctx, W, H, S, v2); };
    const step = dt => { size(); A.update(S, dt, human, W || 380, H || 480); };
    function frame(now) {
      const dt = last ? Math.min(.05, (now - last) / 1000) : 1 / 60; last = now;
      step(dt); draw();
      raf = (playing && visible && !document.hidden) ? requestAnimationFrame(frame) : 0;
    }
    const wake = () => { if (playing && visible && !document.hidden && !raf) { last = 0; raf = requestAnimationFrame(frame); } };
    let io = null;
    if ("IntersectionObserver" in window) { io = new IntersectionObserver(es => { visible = es[0].isIntersecting; wake(); }); io.observe(canvas); }
    const onVis = () => wake(); document.addEventListener("visibilitychange", onVis);
    reset();
    return {
      name,
      start() { playing = true; wake(); },
      stop() { playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; },
      draw, reset(s) { reset(s); draw(); },
      warm(sec) { for (let t = 0; t < sec; t += 1 / 30) step(1 / 30); draw(); },
      v2(on) { v2 = !!on; if (S) S.v2 = v2; draw(); },
      play() { human = true; reset(Date.now() % 9973); },
      auto() { human = false; },
      key(e, down) { const k = KEYS[e.key]; if (!k) return false; if (down && !human) this.play(); if (down || A.hold) A.key(S, k, down, W, H); return true; },
      pointer(x) { if (A.pointer) { if (!human) this.play(); A.pointer(S, x); } },
      destroy() { this.stop(); if (io) io.disconnect(); document.removeEventListener("visibilitychange", onVis); },
      get human() { return human; }
    };
  }
  window.PooledApps = { make, names: Object.keys(APPS) };
})();
