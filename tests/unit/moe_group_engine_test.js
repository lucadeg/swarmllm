// No-GPU check of the engine side of the expert-grouped prefill (engine/qwen35.js _prefillGrouped and the
// prefillTokens hook): a recording fake device / encoder and stubbed layer encoders. It checks the command
// stream: per layer, every sub-pass loads its NC residual columns, runs the layer with its own frame
// (group-0 override) and position, stops after routing on fused MoE layers, copies residual / normed input /
// routing to the right ubatch offsets; then one sort + grouped gate/up + grouped down + combine per fused
// layer; frames and embeddings written first; the draft-cache fill per sub-pass after the trunk; and the
// fallback to the ordinary passes for the tail and when engine.moeGroup is false.
import { Qwen35Engine } from "../../engine/qwen35.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error(`${m}:\n  got  ${ja}\n  want ${jb}`); };

function fake({ U = 32, NC = 16, nLayers = 3, fused = [true, false, true], mtp = true } = {}) {
  const log = [], buf = (name) => ({ name });
  const e = Object.create(Qwen35Engine.prototype);
  const B = { x: { buf: buf("B.x"), stride: 8192 }, xn: { buf: buf("B.xn"), stride: 8192 }, mSel: buf("B.mSel"), mSelw: buf("B.mSelw") };
  const gB = { U, nSub: U / NC, XW: buf("XW"), XNW: buf("XNW"), sel: buf("sel"), selw: buf("selw"), ind: buf("ind"),
    frames: Array.from({ length: U / NC }, (_, s) => buf(`frame${s}`)), common: Array.from({ length: U / NC }, (_, s) => ({ tag: `g0_${s}` })),
    bgSort: "bgSort", bgComb: "bgComb", sortU: buf("sortU"), sortArgs: [256, 128, 512], sortW: 0 };
  const q = {
    writeBuffer: (b, off, data) => log.push(["write", b.name, off, /^(frame|sortU)/.test(b.name) ? Array.from(data) : data.length]),
    submit: () => log.push(["submit"]),
    onSubmittedWorkDone: () => Promise.resolve(),
  };
  const encoder = () => ({
    copyBufferToBuffer: (a, ao, b, bo, n) => log.push(["copy", a.name, ao, b.name, bo, n]),
    beginComputePass: () => ({ end: () => log.push(["end"]) }),
    finish: () => ({}),
  });
  Object.assign(e, {
    NC, B, gB, pos: 100, moeGrpU: U, moeGroup: true, moe: { KS: 9 }, dims: { dim: 2048 },
    layers: Array.from({ length: nLayers }, (_, l) => ({ fused: fused[l], gusgPipe: "moe_gusg_q4_q8", dngPipe: "moe_dng_q4_q8" })),
    layerB: Array.from({ length: nLayers }, (_, l) => ({ mc: { gusG: `gusG${l}`, dngG: `dngG${l}` } })),
    x: buf("x"), mtp: mtp ? {} : null, mtpFill: true, mtpBatchFill: true, frameBufsB: [buf("fb0")],
    device: { queue: q, createCommandEncoder: encoder },
    _embedRowF32: (id) => new Float32Array(2048).fill(id),
    _encodeLayerBatch(enc, l, basePos, nCols, snap, routeOnly) { log.push(["layer", l, basePos, nCols, !!snap, !!routeOnly, this._g0 ? this._g0.tag : "g0_default"]); },
    _dxyz: (p, name, bg, x, y, z) => log.push(["disp", name, bg, x, y, z]),
    _dInd: (p, name, bg, b, off) => log.push(["ind", name, bg, b.name, off]),
    _mtpFillBatch: (ids, i, pos, n) => log.push(["mtpfill", i, pos, n]),
  });
  return { e, log };
}

for (const [U, W] of [[32, 32], [64, 32]]) Deno.test(`grouped prefill: command stream of one ubatch (${W} of ${U})`, async () => {
  const NC = 16, { e, log } = fake({ U, NC });
  const ids = Array.from({ length: 40 }, (_, i) => 1000 + i);
  await e._prefillGrouped(ids, 4, W);
  const want = [["write", "sortU", 0, [W * 9, 256, 128, 512]]];
  for (let s = 0; s < 2; s++) want.push(["write", `frame${s}`, 0, [100 + s * NC, 100 + s * NC + 1, NC, 0]]);
  for (let c = 0; c < W; c++) want.push(["write", "XW", c * 8192, 2048]);
  const xs = NC * 8192, ss = NC * 9 * 4;
  for (let l = 0; l < 3; l++) {
    const f = l !== 1;
    for (let s = 0; s < 2; s++) {
      want.push(["copy", "XW", s * xs, "B.x", 0, xs]);
      want.push(["layer", l, 100 + s * NC, NC, false, f, `g0_${s}`]);
      want.push(["copy", "B.x", 0, "XW", s * xs, xs]);
      if (f) want.push(["copy", "B.xn", 0, "XNW", s * xs, xs], ["copy", "B.mSel", 0, "sel", s * ss, ss], ["copy", "B.mSelw", 0, "selw", s * ss, ss]);
    }
    if (f) want.push(["disp", "moe_gsort", "bgSort", 1, 1, 1], ["ind", "moe_gusg_q4_q8", `gusG${l}`, "ind", 0], ["ind", "moe_dng_q4_q8", `dngG${l}`, "ind", 12],
      ["disp", "moe_combw", "bgComb", 32, W, 1], ["end"]);
  }
  want.push(["copy", "XW", (W - 1) * 8192, "x", 0, 8192], ["submit"]);
  for (let s = 0; s < 2; s++) want.push(["copy", "XW", s * xs, "B.x", 0, xs], ["submit"], ["mtpfill", 4 + s * NC, 100 + s * NC, NC]);
  eq(log, want, "command stream");
  if (e.pos !== 100 + W) throw new Error(`pos ${e.pos}`);
  // the same width again does not rewrite the sort's uniform
  log.length = 0; await e._prefillGrouped(ids, 4, W);
  if (log.some((r) => r[1] === "sortU")) throw new Error("sort uniform rewritten for an unchanged width");
  if (e._g0) throw new Error("group-0 override left set");
});

Deno.test("grouped prefill: prefillTokens takes whole ubatches, then the ordinary passes; moeGroup = false skips it", async () => {
  const cases = [
    [true, 32, 70, [["grouped", 0, 100, 32], ["grouped", 32, 132, 32], ["pass", 164, 4], ["token", 168], ["token", 169]]],
    [false, 32, 70, [["pass", 100, 16], ["pass", 116, 16], ["pass", 132, 16], ["pass", 148, 16], ["pass", 164, 4], ["token", 168], ["token", 169]]],
    // a partial last ubatch of whole passes (46 -> 32), then under two passes' worth through the twins
    [true, 64, 110, [["grouped", 0, 100, 64], ["grouped", 64, 164, 32], ["pass", 196, 8], ["pass", 204, 4], ["token", 208], ["token", 209]]],
    [true, 64, 90, [["grouped", 0, 100, 64], ["pass", 164, 16], ["pass", 180, 8], ["token", 188], ["token", 189]]],
  ];
  for (const [on, U, n, want] of cases) {
    const { e } = fake({ U, NC: 16 });
    const calls = [];
    e.B.init = true; e.moeGroup = on;
    e._noteDV = () => {};
    e._prefillGrouped = async (ids, i0, W) => { calls.push(["grouped", i0, e.pos, W]); e.pos += W; };
    e._mtpFillBatch = () => {}; e.mtp = null;
    e._encodeLayerBatch = (enc, l, basePos, n) => { if (l === 0) calls.push(["pass", basePos, n]); };
    e.device.queue.writeBuffer = () => {}; e.frameBufsB = Array.from({ length: 16 }, () => ({}));
    e.prefillToken = async () => { calls.push(["token", e.pos]); e.pos++; };
    await e.prefillTokens(Array.from({ length: n }, (_, i) => i));
    eq(calls, want, `moeGroup ${on}, U ${U}, ${n} tokens`);
  }
});
