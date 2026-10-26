// engine/qwen35.js draft-side bookkeeping, on the CPU: specStep / specStepDrafts with the GPU work
// replaced by a symbolic draft block. A hidden state is a string tag, the draft block's output for
// (hidden h, token t) is "m(h,t)", its draft is a hash of that output, and the trunk emits a fixed
// token sequence. The batched refill + pre-run first draft (mtpBatchRefill, mtpPreDraft) must give
// the same drafts, the same output and the same draft-cache rows as the one-submit-per-row path.
import { Qwen35Engine } from "../../engine/qwen35.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

const V = 50;
const target = Array.from({ length: 400 }, (_, i) => (i * 7 + ((i * i) % 11)) % V);   // trunk's tokens by position
const hash = (s) => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0; return h; };
// the draft head guesses right ~2/3 of the time, deterministically from the block output
const headOf = (x) => { const m = /,(\d+)\)$/.exec(x); const pos = m ? +m[1] : 0; return hash(x) % 3 ? target[pos + 1] ?? 0 : (hash(x) >> 3) % V; };

function fakeEngine({ refill, pre }) {
  const e = Object.create(Qwen35Engine.prototype);
  const log = { rows: new Map(), drafts: [] };
  Object.assign(e, {
    dims: { dim: 1, vocab: V }, mtp: { stats: { drafts: 0, accepted: 0 }, xNext: {} }, maxDrafts: 7, maxSeq: 1e9, NC: 4,
    mtpBatchRefill: refill, mtpPreDraft: pre, x: "h@-1", xNextTag: null, B: {}, log,
  });
  // tokens carry their position so the fake head can look the answer up: token value = target
  // value, but the block output tag records the row it was written for
  e._mtpRun = async function (src, tok, pos, want) {
    log.rows.set(pos, `${this.x}|${tok}`);
    this.x = `m(${this.x},${pos})`;
    return want ? headOf(this.x) : null;
  };
  e._mtpRefill = function (toks, hs, pos, m, head) {
    for (let c = 0; c < m; c++) log.rows.set(pos + c + 1, `${hs[c]}|${toks[c]}`);
    this.xNextTag = `m(${hs[m - 1]},${pos + m})`;
    return head ? Promise.resolve(headOf(this.xNextTag)) : null;
  };
  e._preDraft0 = async function (p) { this.x = this.xNextTag; return p.id ? await p.id : headOf(this.x); };
  e._restoreDN = () => {};
  e.device = { queue: { writeBuffer: (buf, off, h) => { if (buf === e.x || buf === "X") e.x = h[0]; } } };
  // this.x is a string, so route writes to it through a sentinel
  Object.defineProperty(e, "x", { get() { return this._x; }, set(v) { this._x = v; } });
  const wb = e.device.queue.writeBuffer;
  e.device.queue.writeBuffer = (buf, off, h) => { if (buf === e._x) e._x = h[0]; else wb(buf, off, h); };
  e.verifyN = async (tokens, pos) => ({
    lgs: tokens.map((_, k) => target[pos + k + 1]),
    hs: Object.assign(tokens.map((_, k) => `h@${pos + k}`), { subarray(a, b) { return this.slice(a, b); } }),
  });
  return e;
}

async function run(cfg, plan) {
  const e = fakeEngine(cfg);
  e.pos = 0; e._x = "h@-1";
  let next = target[0];
  const out = [next];
  for (const [kind, K] of plan) {
    let toks;
    if (kind === "mtp") toks = await e.specStep(next, (t) => t, K);
    else toks = await e.specStepDrafts(next, (t) => t, target.slice(e.pos + 1, e.pos + 1 + K).map((t, i) => (i === K - 1 ? (t + 1) % V : t)));
    e.log.drafts.push(toks.length);
    out.push(...toks); next = toks[toks.length - 1];
  }
  // rows below the final position are final; row pos is (re)written by the next step: the old path
  // leaves a drafted row there, the pre-run the exact one the next step would write
  const rows = [...e.log.rows].filter(([p]) => p < e.pos).sort((a, b) => a[0] - b[0]);
  return { out, pos: e.pos, rows, x: e._x, acc: e.mtp.stats.accepted, rowPos: e.log.rows.get(e.pos), next };
}

const plans = [
  Array.from({ length: 30 }, () => ["mtp", 3]),
  Array.from({ length: 30 }, (_, i) => ["mtp", 1 + (i % 7)]),
  Array.from({ length: 30 }, (_, i) => (i % 4 === 2 ? ["lookup", 2 + (i % 5)] : ["mtp", 3])),
];

Deno.test("batched refill + pre-run first draft == one submit per row (drafts, output, cache rows)", async () => {
  for (const plan of plans) {
    const ref = await run({ refill: false, pre: false }, plan);
    for (const cfg of [{ refill: true, pre: false }, { refill: true, pre: true }]) {
      const got = await run(cfg, plan);
      eq(got.out, ref.out, `output ${JSON.stringify(cfg)}`);
      eq(got.acc, ref.acc, `accepted drafts ${JSON.stringify(cfg)}`);
      eq(got.pos, ref.pos, "position");
      eq(got.x, ref.x, "trunk hidden left in x");
      eq(got.rows, ref.rows, `draft-cache rows ${JSON.stringify(cfg)}`);
      if (cfg.pre && plan.at(-1)[0] === "mtp") eq(got.rowPos, `h@${got.pos - 1}|${got.next}`, "pre-run row");
    }
  }
});

Deno.test("pre-run draft is dropped when the next call does not match it", async () => {
  const e = fakeEngine({ refill: true, pre: true });
  e.pos = 0; e._x = "h@-1";
  const o = await e.specStep(target[0], (t) => t, 3);
  const p = e._pre;
  if (!p || p.pos !== e.pos || p.tok !== o[o.length - 1]) throw new Error("no pending pre-run after a step");
  eq(e._takePre(o[o.length - 1] + 1, e.pos), null, "other token");
  eq(e._pre, null, "cleared");
  e._pre = p; e.setHidden([`h@x`]); eq(e._pre, null, "setHidden clears it");
});

Deno.test("draftVocabAuto: small head off above 5% ids >= draftVocab, back on below 2.5%", () => {
  const e = Object.create(Qwen35Engine.prototype);
  Object.assign(e, { headOpDraft: {}, draftVocab: 100, draftVocabAuto: true, _dvMiss: 0, _dvSmall: true });
  e._noteDV(Array(64).fill(5)); eq(e._smallHead(), true, "english-like");
  e._noteDV([500, 5, 5, 5, 500, 5, 5, 5]); eq(e._smallHead(), false, "25% rare ids");
  e._noteDV(Array(10).fill(5)); eq(e._smallHead(), false, "hysteresis: not yet below 2.5%");
  e._noteDV(Array(80).fill(5)); eq(e._smallHead(), true, "back on");
  e.draftVocabAuto = false; e._dvSmall = false; eq(e._smallHead(), true, "auto off: always small");
  e.headOpDraft = null; eq(e._smallHead(), false, "no small head built");
});

Deno.test("draftVocabAuto: chat-template control tokens do not count as misses", () => {
  const e = Object.create(Qwen35Engine.prototype);
  const skip = new Uint8Array(500); skip[400] = 1; skip[401] = 1;   // ids 500, 501 are control tokens
  Object.assign(e, { headOpDraft: {}, draftVocab: 100, draftVocabAuto: true, _dvMiss: 0, _dvSmall: true, _dvSkip: skip });
  e._noteDV([500, 5, 501, 5, 5, 500, 5, 501, 5, 5]); eq(e._smallHead(), true, "control tokens ignored");
  e._noteDV([300, 5, 5, 5, 300, 5, 5, 5]); eq(e._smallHead(), false, "normal rare ids still count");
});
