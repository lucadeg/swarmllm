// CPU-only checks for Qwen35Engine fuseProj (merged projection GEMVs): the row-concatenated
// weights hold every original row byte for byte at its segment offset, every segment starts
// 256-byte aligned, and the row-grouping guard refuses layouts that would move a row between
// the kernel's full and tail branches. A mock device stands in for WebGPU (no GPU needed).
import { Qwen35Engine } from "../../engine/qwen35.js";
const assert = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const assertEquals = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
const U = GPUBufferUsage;

function mockDevice() {
  const dev = {
    limits: { maxStorageBufferBindingSize: 2 ** 30, maxBufferSize: 2 ** 30 },
    createBuffer: ({ size, usage }) => ({ size, usage, data: new Uint8Array(size), destroyed: false, destroy() { this.destroyed = true; } }),
    createCommandEncoder: () => {
      const ops = [];
      return {
        copyBufferToBuffer(src, so, dst, doff, n) {
          assert(so % 4 === 0 && doff % 4 === 0 && n % 4 === 0, "copy offsets and sizes must be multiples of 4");
          assert(src.usage & U.COPY_SRC, "source needs COPY_SRC");
          assert(dst.usage & U.COPY_DST, "destination needs COPY_DST");
          assert(so + n <= src.size && doff + n <= dst.size, "copy in bounds");
          ops.push(() => dst.data.set(src.data.subarray(so, so + n), doff));
        },
        finish: () => ops,
      };
    },
    queue: { submit: (bufs) => { for (const ops of bufs) for (const f of ops) f(); } },
  };
  return dev;
}
const fill = (b, seed) => { for (let i = 0; i < b.size; i++) b.data[i] = (i * 31 + seed * 17) & 255; return b; };
const bytesOf = (v) => v.__view ? v.buffer.data.subarray(v.offset, v.offset + v.size) : v.data;

function mkEntry(dev, kind, rows, dIn, seed) {
  const S = U.STORAGE | U.COPY_SRC;
  if (kind === "f32") return { kind, buf: fill(dev.createBuffer({ size: rows * dIn * 4, usage: S }), seed) };
  const qsRow = kind === "q4" ? dIn / 2 : dIn;
  return { kind, qs: fill(dev.createBuffer({ size: rows * qsRow, usage: S }), seed),
    sc: fill(dev.createBuffer({ size: Math.ceil(rows * dIn / 32 / 2) * 4, usage: S }), seed + 1) };
}

for (const kind of ["q4", "q8", "f32"]) {
  Deno.test(`fuseProj: ${kind} segments land byte for byte at aligned offsets`, () => {
    const dev = mockDevice(), eng = Object.create(Qwen35Engine.prototype);
    eng.device = dev;
    const dIn = 128, rows = [100, 48, 1];
    const ws = rows.map((r, i) => mkEntry(dev, kind, r, dIn, i + 1));
    const orig = ws.map((w) => (kind === "f32" ? [w.buf] : [w.qs, w.sc]).map((b) => b.data.slice()));
    const srcs = rows.map(() => ({}));
    const m = eng._fuseW(ws.map((w, i) => ({ src: srcs[i], w, rows: rows[i] })), dIn);
    assert(m);
    assertEquals(m.offs, [0, 128, 192]);
    assertEquals(m.rows, 193);
    assertEquals(m.lens, rows);
    const rb = kind === "q4" ? [dIn / 2, dIn / 16] : kind === "q8" ? [dIn, dIn / 16] : [dIn * 4];
    m.parts.forEach((pv, i) => {
      const views = kind === "f32" ? [pv.buf] : [pv.qs, pv.sc];
      views.forEach((v, j) => {
        assert(v.__view);
        assertEquals(v.offset % 256, 0, "segment start must be 256-byte aligned for a binding offset");
        assertEquals(v.offset, m.offs[i] * rb[j]);
        assertEquals(v.size, rows[i] * rb[j]);
        assertEquals([...bytesOf(v)], [...orig[i][j].subarray(0, rows[i] * rb[j])], `segment ${i} buffer ${j} bytes`);
      });
      assertEquals(srcs[i].gpu, pv, "loader entry now points at its view");
    });
    // the padding rows between segments stay zero
    const mw = kind === "f32" ? m.w.buf : m.w.qs, r0 = rb[0];
    assert(mw.data.subarray(100 * r0, 128 * r0).every((x) => x === 0));
    for (const w of ws) for (const b of kind === "f32" ? [w.buf] : [w.qs, w.sc]) assert(b.destroyed, "old buffers released");
  });
}

Deno.test("fuseProj: a merged entry can be merged again from its views (second engine, same weights)", () => {
  const dev = mockDevice(), eng = Object.create(Qwen35Engine.prototype);
  eng.device = dev;
  const a = mkEntry(dev, "q4", 64, 64, 3), b = mkEntry(dev, "q4", 64, 64, 4);
  const qa = a.qs.data.slice(), qb = b.qs.data.slice();
  const m1 = eng._fuseW([{ w: a, rows: 64 }, { w: b, rows: 64 }], 64);
  const m2 = eng._fuseW([{ w: m1.parts[0], rows: 64 }, { w: m1.parts[1], rows: 64 }], 64);
  assertEquals([...bytesOf(m2.parts[0].qs)], [...qa]);
  assertEquals([...bytesOf(m2.parts[1].qs)], [...qb]);
  assert(!m1.w.qs.destroyed, "views are never destroyed");
});

Deno.test("fuseProj: refuses mixed formats, odd widths and buffers without COPY_SRC", () => {
  const dev = mockDevice(), eng = Object.create(Qwen35Engine.prototype);
  eng.device = dev;
  assertEquals(eng._fuseW([{ w: mkEntry(dev, "q4", 64, 64, 1), rows: 64 }, { w: mkEntry(dev, "q8", 64, 64, 2), rows: 64 }], 64), null);
  assertEquals(eng._fuseW([{ w: mkEntry(dev, "f32", 4, 96, 1), rows: 4 }, { w: mkEntry(dev, "f32", 4, 96, 2), rows: 4 }], 96), null);
  const w = mkEntry(dev, "q4", 64, 64, 1);
  w.qs.usage = U.STORAGE;
  const w2 = mkEntry(dev, "q4", 64, 64, 2);
  assertEquals(eng._fuseW([{ w, rows: 64 }, { w: w2, rows: 64 }], 64), null);
  assert(!w.qs.destroyed && !w2.qs.destroyed, "nothing released when the merge is refused");
});

Deno.test("fuseProj: row-grouping guard", () => {
  assertEquals(Qwen35Engine._segOffs([10240, 6144]), [0, 10240]);
  assertEquals(Qwen35Engine._segOffs([48, 48]), [0, 64]);
  assertEquals(Qwen35Engine._segOffs([256, 1]), [0, 256]);
  for (const R of [1, 2, 4, 8]) {
    assert(Qwen35Engine._rowsKeep([10240, 6144], R));   // 27B qkv | z
    assert(Qwen35Engine._rowsKeep([48, 48], R));        // 27B beta | alpha
    assert(Qwen35Engine._rowsKeep([32, 32], R));        // MoE beta | alpha
    assert(Qwen35Engine._rowsKeep([1024, 1024], R));    // 27B k | v
    assert(Qwen35Engine._rowsKeep([256, 1], R));        // MoE router | shared gate (last may be partial)
  }
  assert(!Qwen35Engine._rowsKeep([48, 48], 3), "segment start 64 is not a multiple of 3");
  assert(!Qwen35Engine._rowsKeep([6, 64], 4), "first segment would end in a partial group");
});
