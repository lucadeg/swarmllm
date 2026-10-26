// Tiny static server with HTTP Range support (python's http.server has none), for loading GGUF shards in a tab.
// Also serves pre-converted tensors from the weight cache (tests/weight_cache.js) for bench.html?wcache=1:
//   GET /__wcache?model=/models/q38/model.gguf&name=blk.0.attn_q.weight
// returns the exact entry convertEntry() makes (qs | pad | scales, or f32 data), converting and
// caching it on a miss. Headers X-Kind / X-A / X-Pad / X-B give the layout. The tab then uploads
// these bytes as-is instead of repacking in JS: same buffers, no conversion in the page.
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
import { parseGGUFHeader, convertEntry } from "../../engine/gguf.js";
import { openWeightCache } from "../weight_cache.js";
const root = path.resolve(process.argv[2] || "."), port = +(process.argv[3] || 8791);
// 8 MB read chunks: the default 64 KB caps a GGUF range at ~0.27 GB/s here, most of a tab's load time
const HWM = 8 << 20;
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json" };

const models = new Map();   // gguf path -> { G, fd, cache }
function model(p) {
  let m = models.get(p);
  if (!m) {
    const fd = fs.openSync(p, "r"), head = new Uint8Array(64 << 20);
    fs.readSync(fd, head, 0, head.length, 0);
    m = { fd, G: parseGGUFHeader(head.buffer, { skipTokenizer: true }), cache: openWeightCache(p) };
    console.log(`wcache: ${path.relative(root, p)} -> ${m.cache ? m.cache.dir : "cache disabled (converting per request)"}`);
    models.set(p, m);
  }
  return m;
}
function readAt(fd, off, len) {
  const out = new Uint8Array(len); let o = 0;
  while (o < len) { const n = fs.readSync(fd, out, o, Math.min(len - o, 1 << 30), off + o); if (n <= 0) throw new Error("short read"); o += n; }
  return out;
}
function serveEntry(req, res, q) {
  const p = path.join(root, q.get("model") || ""); if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
  const m = model(p), info0 = m.G.tensors[q.get("name")];
  if (!info0) { res.writeHead(404).end("no tensor"); return; }
  // same reshape as ggufEntry: stacked experts [nExp][dOut][dIn] -> [nExp * dOut][dIn]
  const info = info0.shape.length === 3 ? { ...info0, shape: [info0.shape[0] * info0.shape[1], info0.shape[2]] } : info0;
  const head = (kind, a, pad, b) => res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Length": a + pad + b, "Cache-Control": "no-store",
    "X-Kind": kind, "X-A": a, "X-Pad": pad, "X-B": b, "X-Shape": JSON.stringify(info.shape) });
  let loc = m.cache?.locate(info);
  if (!loc) {   // miss (or no cache): convert here, store, then serve what was stored
    const e = convertEntry(info, readAt(m.fd, info.byteOffset, info.byteLength));
    m.cache?.put(info, e);
    loc = m.cache?.locate(info);
    if (!loc) {   // cache disabled or read-only: serve from memory
      const u8 = (v) => Buffer.from(v.buffer, v.byteOffset, v.byteLength);
      const parts = e.kind === "f32" ? [u8(e.data)] : [u8(e.qs), Buffer.alloc((4 - (e.qs.byteLength & 3)) & 3), u8(e.scales)];
      head(e.kind, parts[0].length, parts[1]?.length || 0, parts[2]?.length || 0);
      for (const x of parts) if (x.length) res.write(x);
      res.end();
      return;
    }
  }
  head(loc.kind, loc.a, loc.pad, loc.b);
  fs.createReadStream(loc.file, { start: loc.offset, end: loc.offset + loc.a + loc.pad + loc.b - 1, highWaterMark: HWM }).pipe(res);
}

http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/__wcache") { try { serveEntry(req, res, url.searchParams); } catch (err) { console.error("wcache:", err); res.writeHead(500).end(String(err)); } return; }
  const p = path.join(root, decodeURIComponent(url.pathname)); if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
  fs.stat(p, (err, st) => { if (err || !st.isFile()) { res.writeHead(404).end(); return; }
    const h = { "Content-Type": types[path.extname(p)] || "application/octet-stream", "Accept-Ranges": "bytes", "Cache-Control": "no-store" };
    const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range || "");
    if (m) { const a = +m[1], b = m[2] ? +m[2] : st.size - 1; res.writeHead(206, { ...h, "Content-Range": `bytes ${a}-${b}/${st.size}`, "Content-Length": b - a + 1 }); fs.createReadStream(p, { start: a, end: b, highWaterMark: HWM }).pipe(res); }
    else { res.writeHead(200, { ...h, "Content-Length": st.size }); fs.createReadStream(p, { highWaterMark: HWM }).pipe(res); } });
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} on http://127.0.0.1:${port}`));
