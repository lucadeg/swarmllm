// CPU-only: how much a reduced-vocabulary draft head (?draftvocab=N) can cost in acceptance.
// A draft head restricted to the first N rows returns the same token as the full head whenever
// the full head's argmax is < N, so the acceptance it loses is at most the share of generated
// tokens with id >= N. This tokenizes sample text of each kind (prose, code, chat/markdown) with
// the model's own tokenizer and prints that share per N, plus the head's bytes saved per draft.
// No GPU. Usage (from the repo root):
//   deno run --allow-read benchmarks/draftvocab_coverage.js [model.gguf] [files...]
import { parseGGUFHeader, tokenizerFromGGUF, ggmlTypeBytes } from "../engine/gguf.js";
import { makeTokenizer } from "../engine/engine.js";
const args = Deno.args;
const PATH = args[0] || "models/q38/model.gguf";
const fh = await Deno.open(PATH);
const head = new Uint8Array(64 << 20); let got = 0;
while (got < head.length) { const n = await fh.read(head.subarray(got)); if (n === null) break; got += n; }
const G = parseGGUFHeader(head.buffer);
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const out = G.tensors["output.weight"] || G.tensors["token_embd.weight"];
const [vocab, dim] = out.shape;   // torch order: [rows = vocab, cols = hidden]
const rowBytes = ggmlTypeBytes(out.ggmlType, dim);
console.log(`${PATH}: vocab ${vocab}, dim ${dim}, head ggml type ${out.ggmlType}, ${(rowBytes * vocab / 2 ** 20).toFixed(0)} MiB per full-head read`);
const readText = (p) => { try { return Deno.readTextFileSync(p); } catch { return ""; } };
const walk = (dir, re) => { const o = []; try { for (const e of Deno.readDirSync(dir)) { const p = dir + "/" + e.name; if (e.isFile && re.test(e.name)) o.push(p); } } catch { /* missing */ } return o; };
const sets = args.length > 1 ? { given: args.slice(1) } : {
  prose: walk("docs/research", /\.md$/).slice(0, 8),
  js: [...walk("engine", /\.js$/), ...walk("room", /\.js$/)].slice(0, 12),
  wgsl: walk("engine/wgsl", /\.js$/).slice(0, 6),
  tests: walk("tests", /^test_.*\.js$/).slice(0, 12),
};
const Ns = [16384, 32768, 49152, 65536, 98304, 131072, 151643];
console.log("set".padEnd(8) + "tokens".padStart(9) + Ns.map((n) => (">=" + n).padStart(9)).join(""));
for (const [name, files] of Object.entries(sets)) {
  const ids = []; for (const f of files) ids.push(...tok.encode(readText(f)));
  if (!ids.length) continue;
  const row = Ns.map((n) => ((100 * ids.filter((t) => t >= n).length / ids.length).toFixed(2) + "%").padStart(9)).join("");
  console.log(name.padEnd(8) + String(ids.length).padStart(9) + row);
}
console.log("head MiB per draft at N: " + Ns.map((n) => `${n}: ${(rowBytes * Math.ceil(n / 64) * 64 / 2 ** 20).toFixed(0)}`).join(", "));
fh.close();
