// CPU microbench of the tool-call constraint (harness/constrain.js) with the real Qwen tokenizer:
// replays a realistic answer (prose + a long write_file call + a serve call) through
// constrainedSampler and reports the per-token cost, cold (first scans) and warm (cached masks),
// against the bare sampler. No GPU. deno run --allow-read tests/bench/constrain_bench.js [model.gguf]
import { parseGGUFHeader, tokenizerFromGGUF } from "../../engine/gguf.js";
import { makeTokenizer } from "../../engine/engine.js";
import { constrainedSampler, tokenTexts } from "../../harness/model-common.js";
import { codingTools } from "../../harness/codetools.js";
import { previewTools } from "../../harness/preview-tools.js";
import { renderCalls } from "../../harness/tools.js";

const dir = new URL(".", import.meta.url).pathname;
const path = Deno.args[0] || dir + "../../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const f = await Deno.open(path, { read: true });
const head = new Uint8Array(48 << 20);
let got = 0;
while (got < head.length) { const n = await f.read(head.subarray(got)); if (n === null) break; got += n; }
const tok = makeTokenizer(tokenizerFromGGUF(parseGGUFHeader(head.buffer).meta));
let V = 0;
for (const v of Object.values(tok.vocab)) if (v >= V) V = v + 1;
const tools = [...codingTools({}), ...previewTools({})].map(({ name, description, parameters }) => ({ name, description, parameters }));
const stops = ["<|im_end|>", "<|endoftext|>"].map((s) => tok.vocab[s]).filter(Number.isInteger);

const code = Array.from({ length: 120 }, (_, i) => `  if (board[${i}] && x < 10) { ctx.fillRect(x * 20, y * 20, 20, 20); } // row ${i} <canvas>`).join("\n");
const answer = "I'll write the game.\n" + renderCalls([{ name: "write_file", arguments: { path: "game.js", content: code, append: false } }, { name: "serve", arguments: { port: 5173 } }], "xml");
const ids = tok.encode(answer);
const argmax = (lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; };
const noise = new Float32Array(V);
for (let i = 0; i < V; i++) noise[i] = Math.random();

const tt = tokenTexts(tok);
function run(constrained) {
  const cs = constrained ? constrainedSampler(argmax, tools, { tokenText: tt, vocabSize: V, style: "xml", stops }) : { sample: argmax, setText() {} };
  const lg = new Float32Array(V);
  let text = "", ms = 0, worst = 0, forced = 0;
  cs.setText("");
  for (const id of ids) {
    lg.set(noise); lg[id] = 2;
    const t0 = performance.now();
    const got = cs.sample(lg);
    const dt = performance.now() - t0;
    ms += dt; worst = Math.max(worst, dt);
    if (got !== id) forced++;
    text += tok.decode([id]);
    cs.setText(text);
  }
  return { perTok: ms / ids.length, worst, forced, keys: cs.constraint?.cache.size ?? 0 };
}
// warm the token-text table first (tokenTexts is per tokenizer and kept by the adapters)
const t0 = performance.now();
for (let i = 0; i < V; i++) tt(i);
console.log(`vocab ${V}, answer ${ids.length} tokens; decoding every token once: ${(performance.now() - t0).toFixed(0)} ms (once per tokenizer)`);
const base = run(false);
const cold = run(true);
const warm = run(true);
const fmt = (r) => `${r.perTok.toFixed(3)} ms/token (worst ${r.worst.toFixed(1)} ms)`;
console.log(`bare argmax:        ${fmt(base)}`);
console.log(`constrained, cold:  ${fmt(cold)}; ${cold.keys} masks computed`);
console.log(`constrained, warm:  ${fmt(warm)}; forced ${warm.forced} (want 0)`);
console.log(`overhead warm: +${(warm.perTok - base.perTok).toFixed(3)} ms/token`);
