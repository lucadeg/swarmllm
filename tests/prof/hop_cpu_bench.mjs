// CPU cost of the per-hop JS work in the room, no GPU: f16 pack, f16 unpack, NaN scan, the readback
// copy (Float32Array.from of a mapped range), transport slicing + reassembly (no channel), for one
// token (1 column) and speculative verify blocks (4 and 8 columns). node tests/prof/hop_cpu_bench.mjs [dim]
import { packF16, unpackF16, badF32 } from "../../room/wire.js";
import { makeLink, sendFrame } from "../../room/transport.js";
const DIM = +(process.argv[2] || 2048);   // Qwen 3.6 35B-A3B hidden size; the 27B is 5120
const time = (fn, n) => { for (let i = 0; i < 20; i++) fn(); const t0 = performance.now(); for (let i = 0; i < n; i++) fn(); return (performance.now() - t0) / n; };
const rows = [];
for (const cols of [1, 4, 8]) {
  const n = DIM * cols, f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = (Math.sin(i * 0.37) * 3) * (i % 97 === 0 ? 40 : 1);
  const u = packF16(f), mapped = new Float32Array(n).buffer;
  // a link with a fake open channel that only counts bytes: slicing and header writes, no network
  const link = makeLink(); let sent = 0; link.chans.push({ readyState: "open", send: (b) => { sent += b.byteLength; } });
  const r = {
    cols, KB: +(u.byteLength / 1024).toFixed(1),
    pack: time(() => packF16(f), 400), unpack: time(() => unpackF16(u), 400), nanScan: time(() => badF32(f), 400),
    readbackCopy: time(() => Float32Array.from(new Float32Array(mapped, 0, n)), 400), sliceCopy: time(() => new Float32Array(mapped, 0, n).slice(), 400),
    sendFrame: time(() => sendFrame(link, { t: "ai-hidden-b", basePos: 5, n: cols, data: u }), 400),
  };
  rows.push(Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" && k !== "cols" && k !== "KB" ? +(v * 1000).toFixed(1) : v])));
}
console.log(`dim ${DIM}, µs per call (node ${process.version})`); console.table(rows);
