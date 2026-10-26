// A recording WebGPU device for CPU-only tests of the engine's command encoding (no GPU, no shader
// execution). It validates what WebGPU validation would reject in the encoding itself (bind group
// shape / range / alignment / usage, copy ranges, dispatch limits, a pipeline and both bind groups
// set before a dispatch) and, at submit time, "executes" the command buffers in queue order:
// buffer copies and clears move real bytes, and every dispatch is logged with its pipeline, grid,
// bind-group ids and the contents of its frame uniform (group 0, binding 1) at that point.
// Shaders are not compiled: the module's code is kept so tests can grep entry points.
export function installGPUGlobals() {
  globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
  globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
}

export function mockDevice({ wgMem = 16384 } = {}) {
  installGPUGlobals();
  const U = globalThis.GPUBufferUsage;
  let nextId = 0, g1 = 0;
  const fail = (m) => { throw new Error("mock WebGPU: " + m); };
  const log = [];
  const limits = {
    maxBufferSize: 2 ** 32, maxStorageBufferBindingSize: 2 ** 31, maxComputeWorkgroupStorageSize: wgMem,
    minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256, maxComputeWorkgroupsPerDimension: 65535,
    maxStorageBuffersPerShaderStage: 10, maxBindGroups: 4, maxComputeInvocationsPerWorkgroup: 256,
  };
  const buffer = (d) => {
    if (!(d.size >= 0)) fail("buffer size");
    const b = {
      __mock: "buffer", id: nextId++, size: d.size, usage: d.usage, data: new Uint8Array(d.size), destroyed: false, label: d.label,
      mapAsync: async (mode, o = 0, n) => { if (o + (n ?? 0) > b.size) fail("mapAsync range"); },
      unmap: () => {}, destroy: () => { b.destroyed = true; },
    };
    return b;
  };
  const checkBuf = (x, what) => { if (!x || x.__mock !== "buffer") fail(`${what}: not a buffer`); if (x.destroyed) fail(`${what}: destroyed buffer`); };
  const bgl = (d) => ({ __mock: "bgl", id: nextId++, entries: d.entries });
  const bindGroup = (d) => {
    const L = d.layout;
    if (!L || L.__mock !== "bgl") fail("bind group without a layout");
    if (d.entries.length !== L.entries.length) fail(`bind group has ${d.entries.length} entries, layout ${L.entries.length}`);
    const ents = d.entries.map((e, i) => {
      if (e.binding !== L.entries[i].binding) fail("binding order");
      const r = e.resource, b = r && r.buffer;
      checkBuf(b, `binding ${i}`);
      const off = r.offset || 0, size = r.size ?? b.size - off, type = L.entries[i].buffer.type;
      if (off % 256) fail(`binding ${i}: offset ${off} not 256-aligned`);
      if (off + size > b.size || size <= 0) fail(`binding ${i}: range ${off}+${size} outside a ${b.size} B buffer`);
      if (type === "uniform") { if (!(b.usage & U.UNIFORM)) fail(`binding ${i}: uniform binding of a non-uniform buffer`); }
      else { if (!(b.usage & U.STORAGE)) fail(`binding ${i}: storage binding of a non-storage buffer`); if (size % 4) fail(`binding ${i}: storage size ${size} % 4`); }
      if (size > limits.maxStorageBufferBindingSize) fail("binding over the storage limit");
      return { buffer: b, offset: off, size, type };
    });
    // WebGPU usage scopes are per buffer: a buffer bound writable may not also be bound read-only or uniform
    for (const a of ents) if (a.type === "storage" && ents.some((c) => c !== a && c.buffer === a.buffer && c.type !== "storage"))
      fail("bind group uses a buffer both writable and read-only");
    // group-1 bind groups are numbered on their own, so ids line up between two engines that differ only
    // in extra pipelines (whose group-0 bind groups are created in between)
    return { __mock: "bg", id: L.group === 1 ? g1++ : "g" + nextId++, layout: L, ents };
  };
  const pipelineOf = (d) => {
    const layouts = d.layout?.bindGroupLayouts;
    if (!layouts) fail("pipeline without an explicit layout");
    if (!d.compute.module.code.includes(`fn ${d.compute.entryPoint}(`)) fail(`entry point ${d.compute.entryPoint} not in the module`);
    layouts.forEach((l, i) => { if (l.group !== undefined && l.group !== i) fail("a layout used at two group indices"); l.group = i; });
    return { __mock: "pipe", id: nextId++, name: d.compute.entryPoint, layouts, getBindGroupLayout: (i) => layouts[i] };
  };
  const encoder = () => {
    const cmds = [];
    let open = false;
    return {
      beginComputePass: () => {
        if (open) fail("nested pass"); open = true;
        let pipe = null; const bgs = [];
        return {
          setPipeline: (p) => { if (!p || p.__mock !== "pipe") fail("setPipeline: not a pipeline"); pipe = p; },
          setBindGroup: (i, g) => { if (!g || g.__mock !== "bg") fail(`setBindGroup(${i}): not a bind group (${g})`); bgs[i] = g; },
          dispatchWorkgroups: (x, y = 1, z = 1) => {
            if (!pipe) fail("dispatch without a pipeline");
            for (const v of [x, y, z]) if (!Number.isInteger(v) || v < 0 || v > 65535) fail(`${pipe.name}: dispatch dimension ${v}`);
            pipe.layouts.forEach((l, i) => { if (!bgs[i]) fail(`${pipe.name}: bind group ${i} not set`); if (bgs[i].layout !== l) fail(`${pipe.name}: bind group ${i} has another layout`); });
            cmds.push({ op: "dispatch", pipe: pipe.name, grid: [x, y, z], bg0: bgs[0], bg1: bgs[1] });
          },
          // the grid comes from the GPU at run time: logged as "indirect" (buffer and offset checked)
          dispatchWorkgroupsIndirect: (b, off) => {
            if (!pipe) fail("dispatch without a pipeline");
            checkBuf(b, "indirect");
            if (!(b.usage & U.INDIRECT)) fail(`${pipe.name}: indirect buffer without INDIRECT usage`);
            if (off % 4 || off + 12 > b.size) fail(`${pipe.name}: indirect offset ${off}`);
            pipe.layouts.forEach((l, i) => { if (!bgs[i]) fail(`${pipe.name}: bind group ${i} not set`); if (bgs[i].layout !== l) fail(`${pipe.name}: bind group ${i} has another layout`); });
            cmds.push({ op: "dispatch", pipe: pipe.name, grid: ["indirect", off], bg0: bgs[0], bg1: bgs[1] });
          },
          end: () => { open = false; },
        };
      },
      copyBufferToBuffer: (s, so, d, dO, n) => {
        if (open) fail("copy inside a pass");
        checkBuf(s, "copy src"); checkBuf(d, "copy dst");
        if (s === d) fail("copy within one buffer");
        if (so % 4 || dO % 4 || n % 4) fail("copy alignment");
        if (so + n > s.size || dO + n > d.size) fail(`copy range: src ${so}+${n}/${s.size}, dst ${dO}+${n}/${d.size}`);
        if (!(s.usage & U.COPY_SRC) || !(d.usage & U.COPY_DST)) fail("copy usage");
        cmds.push({ op: "copy", s, so, d, dO, n });
      },
      clearBuffer: (b, o = 0, n) => { checkBuf(b, "clear"); cmds.push({ op: "clear", b, o, n: n ?? b.size - o }); },
      resolveQuerySet: () => {}, writeTimestamp: () => {},
      finish: () => { if (open) fail("finish with an open pass"); return { __mock: "cb", cmds }; },
    };
  };
  const execute = (cb) => {
    for (const c of cb.cmds) {
      if (c.op === "copy") c.d.data.set(c.s.data.subarray(c.so, c.so + c.n), c.dO);
      else if (c.op === "clear") c.b.data.fill(0, c.o, c.o + c.n);
      else {
        const f = c.bg0.ents[1];
        const frame = f && f.size >= 16 ? Array.from(new Uint32Array(f.buffer.data.buffer, f.offset, 4)) : null;
        log.push({ pipe: c.pipe, grid: c.grid, bg1: c.bg1.id, frame });
      }
    }
  };
  const device = {
    __mock: "device", limits, features: new Set(), log,
    addEventListener: () => {}, pushErrorScope: () => {}, popErrorScope: async () => null, lost: new Promise(() => {}),
    createShaderModule: (d) => ({ __mock: "module", code: d.code, getCompilationInfo: async () => ({ messages: [] }) }),
    createBindGroupLayout: bgl,
    createPipelineLayout: (d) => ({ __mock: "pl", bindGroupLayouts: d.bindGroupLayouts }),
    createComputePipeline: pipelineOf,
    createComputePipelineAsync: async (d) => pipelineOf(d),
    createBindGroup: bindGroup,
    createBuffer: (d) => {
      const b = buffer(d);
      if (d.mappedAtCreation) b.getMappedRange = () => b.data.buffer;
      else b.getMappedRange = (o = 0, n) => b.data.buffer.slice(o, o + (n ?? b.size - o));
      return b;
    },
    createCommandEncoder: encoder,
    createQuerySet: () => ({ destroy() {} }),
    queue: {
      writeBuffer: (b, off, src, so = 0, n) => {
        checkBuf(b, "writeBuffer");
        const bytes = ArrayBuffer.isView(src) ? new Uint8Array(src.buffer, src.byteOffset, src.byteLength) : new Uint8Array(src);
        const part = n != null ? bytes.subarray(so, so + n) : bytes.subarray(so);
        if (off % 4 || part.length % 4) fail("writeBuffer alignment");
        if (off + part.length > b.size) fail(`writeBuffer range ${off}+${part.length}/${b.size}`);
        if (!(b.usage & U.COPY_DST)) fail("writeBuffer to a buffer without COPY_DST");
        b.data.set(part, off);
      },
      submit: (cbs) => { for (const cb of cbs) execute(cb); },
      onSubmittedWorkDone: async () => {},
    },
  };
  return device;
}
