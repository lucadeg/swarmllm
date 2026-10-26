# Prefill profile, 2026-09: Qwen3.8 27B and Qwen3.6 35B-A3B MoE on the GB10

This document profiles prefill on both models, ranks the bottlenecks by model and prompt length, and estimates what each candidate optimization could win. It gives an implementation plan for each one.

- **Branch and base:** `prefill/base` at 8ce009e (feat/engine-opt plus opt/load-cache).
- **Platform:** Deno 2.9.5 with wgpu on Vulkan. The GB10 was otherwise idle for every measurement.
- **Measured vs estimated:** measured numbers carry **[M]**. Anything projected carries **[E]**, along with the arithmetic behind it.

## How it was measured

- **`tests/prof_prefill.js` (new).** It prefills with `prefillTokens` in segments that end at 512, 4096 and 16384 tokens, so all three lengths come from one run.
  - Wall time to each length is the cumulative time of the segments before it. It uses the same prompt recipe as `bench_ctx.js` (this repo's source code), `batchCols 16` and the MTP draft-cache fill on, which is the engine default.
  - A second, instrumented run of the same tokens puts each dispatch of the sampled passes in its own compute pass, with begin and end timestamps.
  - It instruments 32 passes per segment at a uniform stride, and each one is weighted by the number of passes it stands for. Uniform sampling integrates attention correctly, because attention cost grows with position.
  - Dispatches are labeled by the op that issued them (`_dop` is wrapped, and every batched op is mapped to its projection) and by pipeline name. MTP fill work goes in its own category.
  - Commands: `cd tests && MODEL=27b|moe [LENS=512,4096,16384] [SAMPLES=32] [MTP_FILL=0] [GEMM_EXTRA='{"8192x2048":2}'] [OUT=file.json] deno run --unstable-webgpu --allow-read --allow-env --allow-write prof_prefill.js`
- **`tests/bench/bench_wide_gemm.js` (new).** This is a standalone tiled Q4_0 GEMM that uses the engine's repacked weight layout. It has 64×64 to 128×128 tiles, a dequantize-once shared weight tile, a fully unrolled K loop, and runs at N = 64 to 512 columns. It is checked against a naive kernel (relDiff ≤ 1.8e-6). It measures what a wide prefill ubatch could reach (candidate B).
- **ALU peak microbenchmark.** Eight independent FMA or dot chains per thread, in Deno/wgpu on the GB10:

| Operation | Measured throughput |
|---|---|
| f32 FMA | 16.7 TFLOPS [M] |
| f16 `vec4<f16>` FMA | 8.6 TFLOPS [M] (slower than f32 through wgpu/naga) |
| `dot4I8Packed` (dp4a) | 59.7 TOPS [M] (3.6× f32) |

- **Overhead is negligible.** Wall time exceeds the kernel sum by only 0.9 to 4.8% on every run. Prefill is GPU-bound: CPU encoding, submits and syncs are not the problem.

## Reference: llama.cpp CUDA on the same GGUFs

These runs used build 749f688 (`~/llama.cpp/build-cuda/bin/llama-bench -m <gguf> -p 512,4096,16384 -n 0 -r 2 -fa 1`) [M]:

| Model | pp512 | pp4096 | pp16384 |
|---|---|---|---|
| Qwen3.8 27B Q4_0 (14.94 GiB) | 879 ± 26 | 893 ± 5 | 847 |
| Qwen3.6 35B-A3B Q4_0 (19.40 GiB) | 2356 ± 80 | 2374 ± 6 | 2271 |

Our prefill today [M]:

| Model | 512 | 4096 | 16384 | Gap to llama.cpp |
|---|---|---|---|---|
| 27B | 73.3 tok/s | 61.5 | 39.4 | 12× / 15× / 21× |
| MoE | 166.3 tok/s | 143.3 | 98.4 | 14× / 17× / 23× |

## How prefill is computed today

- **Passes.** Prefill runs as passes of 16 token columns (`prefillTokens` → `_encodeLayerBatch`), one command encoder per pass, which walks all layers. When the model has an MTP block, a second batched pass fills the draft cache (`_mtpFillBatch`).
- **Tail.** The tail goes through 8- and 4-column twins, then single tokens.
- **27B dispatch count.** About 2,570 dispatches per 16-token pass. 16 of them per layer are per-column `silu_mul` calls on the GEMM path.
- **MoE dispatch count.** About 550 per pass.

**DeltaNet runs per column batch, sequentially over the tokens inside the batch.**

- `dn_conv_mc`: one thread per channel loops over the 16 columns, with the conv state in registers.
- `dn_pre_mc`: gates plus L2 normalization, one workgroup per column.
- `dn_delta_mc`: one workgroup of 128 threads per value head (48 on the 27B, 32 on the MoE). Each thread keeps one 128-row column of the state in registers, loads it once per pass, runs the 16 tokens one after another, then stores it.
  - Measured cost: 97 µs per pass per layer (27B) and 89 µs (MoE), which is about 6 µs per token per layer [M].
  - It fills only 32 to 48 workgroups on a 48-SM GPU, but it is only **1.2 to 2.9% of prefill** today.
- There is no chunked (parallel-scan) form.

**Projections.**

- **27B:** at the full 16-column width, all shapes pinned in `GEMM_S` go through the row-stationary split-K GEMM (`engine/wgsl/gemm.js`). The Q8_0 `ffn_down`, `ssm_out` and `attn_output` use `kernel8`.
- **MoE:** `gemmOn` is false. No MoE shape (dim 2048) is pinned in `GEMM_S`, so every projection runs on the batched GEMV `matvec_*_coop_b`.

**MoE experts.** `moe_gus` and `moe_dnc` run one workgroup row per (column, slot): 16 columns × 9 slots (8 routed plus the shared expert) per pass. Every (column, slot) streams its expert's weights again.

**Attention.** It uses split-K flash (`attn_flash`) over 256-position splits, one workgroup per (split, column, KV head).

- The 27B uses `attn_flash_t2`, which pairs two columns.
- The MoE runs the single-column kernel, because `2·G·hd = 4096 > 3072` disables t2.
- Inside a workgroup, each thread computes a full 256-long dot product. The online softmax runs serially in G threads.
- K and V are re-read for every column (or pair) of the pass.

## Ranked breakdown: kernel time by category

Values are ms of GPU time for the whole prefill and the percentage of kernel time [M]. Categories:

| Category | What it contains |
|---|---|
| `ffn` | Dense gate, up and down GEMMs, plus split-K reduces |
| `dn_proj` | DeltaNet qkv, z, β/α and out projections |
| `attn_proj` | q+gate, k, v and o projections |
| `attn_core` | kv_store, flash and combine |
| `attn_glue` | attn_glue and sigmoid_mul |
| `dn_recurrence` | `dn_delta_mc` |
| `dn_glue` | `dn_pre_mc` and `dn_gatenorm_mc` |
| `norms_glue` | rmsnorm, add_res and silu_mul |
| `glue` | GEMM activation transposes |
| `mtp` | The whole draft-cache fill pass |
| `moe_*` | Router and route; fused gate/up over routed plus shared (`moe_gus`); fused down plus combine (`moe_dnc`) |

### Qwen3.8 27B (64 layers: 48 DeltaNet, 16 attention)

| Category | 512: ms | % | 4096: ms | % | 16384: ms | % |
|---|---|---|---|---|---|---|
| ffn | 3,787 | **55.5%** | 30,311 | **46.2%** | 120,701 | 29.3% |
| attn_core | 500 | 7.3% | 14,408 | 22.0% | 199,218 | **48.3%** |
| dn_proj | 1,375 | 20.1% | 10,954 | 16.7% | 43,717 | 10.6% |
| attn_proj | 384 | 5.6% | 3,106 | 4.7% | 12,252 | 3.0% |
| mtp (draft fill) | 178 | 2.6% | 2,043 | 3.1% | 17,046 | 4.1% |
| norms_glue (incl. 16× silu_mul per layer) | 265 | 3.9% | 2,102 | 3.2% | 8,395 | 2.0% |
| dn_recurrence | 149 | 2.2% | 1,194 | 1.8% | 4,753 | 1.2% |
| dn_glue | 83 | 1.2% | 665 | 1.0% | 2,659 | 0.6% |
| glue (xpose) | 67 | 1.0% | 530 | 0.8% | 2,118 | 0.5% |
| dn_conv | 23 | 0.3% | 186 | 0.3% | 740 | 0.2% |
| attn_glue | 17 | 0.3% | 138 | 0.2% | 551 | 0.1% |
| **kernel sum** | 6,828 | | 65,637 | | 412,150 | |
| **wall** | 6,988 (73.3 tok/s) | | 66,563 (61.5 tok/s) | | 415,782 (39.4 tok/s) | |

**Top kernels at 512 tokens:**

| Kernel | Share |
|---|---|
| `ffn_down` Q4 GEMM | 21.4% |
| `ffn_gate` | 14.8% |
| `ffn_up` | 14.8% |
| `dn_qkv` | 7.5% |
| `attn_flash_t2` | 7.2% |
| `dn_out` (Q8) | 6.5% |
| `dn_z` | 4.4% |
| `ffn_down` Q8 | 3.6% |
| `silu_mul` | 2.5% (5 µs × 16 per layer) |

**At 16384 tokens,** `attn_flash_t2` alone is 48.1%: 12.1 ms per pass per layer.

### Qwen3.6 35B-A3B MoE (40 layers: 30 DeltaNet, 10 attention; 256 experts, top-8, plus shared)

| Category | 512: ms | % | 4096: ms | % | 16384: ms | % |
|---|---|---|---|---|---|---|
| dn_proj (batched GEMV) | 831 | **27.7%** | 6,589 | 23.8% | 26,447 | 16.2% |
| moe_experts_gu (`moe_gus`) | 812 | **27.1%** | 6,651 | **24.0%** | 27,017 | 16.5% |
| moe_experts_down (`moe_dnc`) | 554 | 18.5% | 4,479 | 16.2% | 18,118 | 11.1% |
| attn_core | 183 | 6.1% | 4,781 | 17.2% | 66,002 | **40.4%** |
| attn_proj (batched GEMV) | 231 | 7.7% | 1,856 | 6.7% | 7,512 | 4.6% |
| mtp (draft fill) | 107 | 3.6% | 1,163 | 4.2% | 9,361 | 5.7% |
| dn_recurrence | 86 | 2.9% | 682 | 2.5% | 2,728 | 1.7% |
| moe_router | 85 | 2.8% | 678 | 2.4% | 2,745 | 1.7% |
| dn_glue | 50 | 1.7% | 398 | 1.4% | 1,584 | 1.0% |
| norms_glue | 35 | 1.2% | 275 | 1.0% | 1,100 | 0.7% |
| dn_conv | 13 | 0.4% | 102 | 0.4% | 409 | 0.3% |
| attn_glue | 10 | 0.3% | 77 | 0.3% | 327 | 0.2% |
| **kernel sum** | 2,997 | | 27,731 | | 163,350 | |
| **wall** | 3,079 (166.3 tok/s) | | 28,585 (143.3 tok/s) | | 166,543 (98.4 tok/s) | |

### Bottleneck ranking

| Model | 512 | 4096 | 16384 |
|---|---|---|---|
| 27B | FFN GEMMs 56% ≫ DN projections 20% > attention 7% | FFN 46% > attention 22% > DN projections 17% | **attention 48%** > FFN 29% > DN projections 11% |
| MoE | experts 46% (gate/up 27%, down 18%) ≈ DN-projection GEMV 28% > attention projections 8% | experts 40% > DN projections 24% > attention 17% | **attention 40%** > experts 28% > DN projections 16% |

On both models, the DeltaNet recurrence plus conv is under 3.5% at every length. The MTP draft-cache fill is 2.6 to 5.7%.

## Achieved efficiency of the prefill projections

Each dispatch covers one 16-column pass. "GB/s" is weight bytes over kernel time; TFLOPS is 2·M·K·16 over kernel time [M].

| Model | Op (shape, format) | Kernel | µs per pass | GB/s | TFLOPS |
|---|---|---|---|---|---|
| 27B | ffn_gate / ffn_up 17408×5120 Q4 | gemm_q4 split-K 4 | 492 | 102 | **5.8** |
| 27B | ffn_down 5120×17408 Q4 | gemm_q4 split-K 2 | 816 | 61 | 3.5 |
| 27B | ffn_down 5120×17408 Q8 | gemm_q8 | 950 | 100 | 3.0 |
| 27B | dn_qkv 10240×5120 Q4 | gemm_q4 | 334 | 88 | 5.0 |
| 27B | dn_z 6144×5120 Q4 | gemm_q4 | 194 | 91 | 5.2 |
| 27B | dn_out 5120×6144 Q8 | gemm_q8 split-K 8 | 288 | 116 | 3.5 |
| 27B | attn q+gate 12288×5120 Q4 | gemm_q4 | 352 | 100 | 5.7 |
| 27B | attn_out 5120×6144 Q4 | gemm_q4 | 250 | 71 | 4.0 |
| 27B | attn k / v 1024×5120 Q4 | gemm_q4 split-K 16 | 67 / 50 | 44 / 58 | 2.5 / 3.3 |
| 27B | dn β+α 96×5120 f32 | GEMV coop_b | 51 | 39 | 0.3 |
| MoE | dn_qkv+z 12288×2048 Q4 | GEMV coop_b | 614 | **23** | **1.3** |
| MoE | dn_out 2048×4096 Q4 / Q8 | GEMV coop_b | 218 / 228 | 22 / 39 | 1.2 |
| MoE | attn q+gate 8192×2048 Q4 / Q8 | GEMV coop_b | 418 / 439 | 23 / 41 | 1.3 / 1.2 |
| MoE | attn kv 1024×2048 Q8 | GEMV coop_b | 82 | 27 | 0.8 |
| MoE | `moe_gus` (16 columns × 9 slots, gate+up) | fused | 638 | **≈266 (DRAM peak)** | ≈0.47 |
| MoE | `moe_dnc` (down + combine) | fused | 412 | ≈190 to 200 | ≈0.37 |

For the MoE expert rows, GB/s counts every (column, slot) stream: 144 × 1.18 MB for gate+up.

What the efficiency numbers show:

- **The 27B GEMMs are ALU-bound, not bandwidth-bound.** They reach 3.0 to 5.8 TFLOPS against a measured f32 FMA peak of 16.7, which is 18 to 35%. They stream only 60 to 116 GB/s of the 273 GB/s DRAM.
  - The 16-column row-stationary GEMM is already about as fast as the first-cut wide tile below.
  - The shapes that are furthest behind are the down-shaped ones (dIn 17408 or 6144, small dOut) and Q8.
- **The MoE projections run at GEMV speed: 1.2 to 1.3 TFLOPS.** No MoE shape is pinned in `GEMM_S`.
- **The MoE experts run at DRAM bandwidth.** Each (column, slot) re-streams its expert, and 128 assignments in a 16-column pass touch about 102 distinct experts (expected value), so batching 16 columns barely helps.
- **Attention runs at 0.26 to 0.33 TFLOPS.**
  - 27B at 16k: 3.2 GFLOP per token over 12.2 ms per token.
  - MoE at 16k: 1.34 GFLOP per token over 4.0 ms per token.
  - This is the least efficient kernel family in the engine.

**Wide-tile GEMM measurement** (`bench_wide_gemm.js`, f32 FMA with a dequantize-once shared tile, best tile per shape) [M]:

| Shape | N=64 | N=128 | N=256 | N=512 |
|---|---|---|---|---|
| 17408×5120 | 6.6 TF | 6.8 | 7.1 | **7.2** |
| 5120×17408 | 5.5 | 6.2 | 6.6 | 6.9 |
| 10240×5120 | 5.5 | 6.6 | 6.8 | 6.9 |
| 8192×2048 (MoE) | 5.4 | – | 6.8 | – |
| 2048×4096 (MoE) | 3.4 | 4.4 | 6.2 | 6.3 |

A first-cut tiled kernel sits at about 7 TFLOPS, 42% of the measured f32 peak. Unrolling the K loop was worth 1.25×; a runtime loop index gets clamped. Vectorizing the dequantize stores gained nothing.

## Candidates

### (G1) Put the MoE projections on the existing prefill GEMM: quick win, measured

- **What.** Pin split-K factors for the MoE shapes in `GEMM_S` (engine/wgsl/gemm.js). Add `attn_q` to the Q8 GEMM list so the Q8 q+gate and kv stop running on the GEMV.
- **Measured** with `GEMM_EXTRA='{"8192x2048":2,"4096x2048":4,"2048x4096":8,"512x2048":16}'` [M]:
  - dn_proj went from 831 to 361 ms at 512 tokens, and attn_proj from 231 to 143 ms.
  - The projections went from 1.3 to 3.0 to 3.7 TFLOPS.
  - **End to end: 166.3 → 198.1 tok/s at 512 (+19%) and 143.3 → 166.8 at 4096 (+16%).**
  - attn q+gate in Q8 (8192×2048) and kv (1024×2048 Q8) are still on the GEMV at 1.2 and 0.8 TFLOPS. Adding them is about another 0.15 ms per token [E].
- **Risk: low.** The kernel already ships for the 27B. `S` must divide the stage count: 32 stages for dIn 2048, 64 for 4096.
- **Exactness.** Prefill summation order changes from the batched GEMV to the GEMM for full-width passes only. The 27B already works this way under "prefill tolerance". Verify and decode stay on the GEMV (`maxDrafts = NC − 2`), so spec == plain is preserved by construction.
  - **Must be validated:** `tests/test_moe.js` (3 × MATCH, spec == plain) and the prefill-vs-sequential tolerance test on the MoE (`tests/test_batch*.js` pattern). Report the max relDiff.
  - `GEMM_S` is a pinned protocol table (every peer must agree), so this is a protocol bump under GOVERNANCE.md.
- **Plan.**
  1. `engine/wgsl/gemm.js`: add `"8192x2048": 2, "4096x2048": 4, "2048x4096": 8, "512x2048": 16`, plus `"1024x2048"` for the merged kv if it stays merged.
  2. `engine/qwen35.js` `_init`: add the MoE's `[convDim, dim]`, `[nH·hd·2, dim]` and `[kvDim·2, dim]` to the shape list; they are already there except the merged kv. Extend the gemm8 `note()` to `L.wq`, `L.wk` and `L.wv`.
  3. Rerun `prof_prefill.js MODEL=moe`, `test_moe.js`, and `run.sh quick`.

### (E) Tiled causal flash attention for prefill passes: the #1 item at long context

- **Why.** Attention is 48% (27B) and 40% (MoE) of prefill at 16k, and 22% / 17% at 4k. It runs at about 0.3 TFLOPS: per-column K/V re-reads, serial 256-long dot products per thread, and a softmax serialized in G threads.
- **What.** For a pass of NC query columns and one KV head, load a tile of K (64 positions × hd, as f16 → f32 in shared memory) once, and compute the scores of all NC × G query rows against it: 16 × 8 = 128 rows on the MoE, 16 × 6 = 96 on the 27B.
  - Use a register-blocked S = Q·Kᵀ micro-GEMM, a parallel row-wise online softmax (one subgroup-free workgroup reduction per row), then O += P·V from the same V tile.
  - Causal masking only matters on the last tile (positions basePos..basePos+15).
  - Keep split-K over the context, because long contexts need the occupancy. Then reuse `attn_combine`.
- **Expected.** K/V traffic drops 8 to 16× per pass. The FLOPs are small: 27B at 16k is 3.2 GFLOP per token; MoE is 1.34.
  - At a conservative 4 TFLOPS [E], attention goes from 12.2 to about 0.8 ms per token (27B, 16k) and from 4.0 to about 0.35 ms per token (MoE, 16k).
  - **End to end [E]:**

| Model | 512 | 4096 | 16384 |
|---|---|---|---|
| 27B, tok/s | 73 → 79 | 62 → 77 | **39 → 71 (1.8×)** |
| MoE, tok/s | 166 → 177 | 143 → 170 | **98 → 155 (1.6×)** |

  - This also shrinks the MTP fill, which runs the same attention.
- **Risk: medium.** It is a new kernel. Workgroup memory is 16 KB by default: request `maxComputeWorkgroupStorageSize` (the adapter has 48 KB) or use hd-split tiles for hd = 256.
- **Exactness.** It changes the prefill summation order of scores and softmax relative to `attn_flash`. Decode and verify keep `attn_flash`, so it should be gated exactly like the GEMM: full-width prefill passes only, so the spec == plain structure is unchanged.
  - It needs the goldens plus the prefill tolerance test. Report relDiff against `attn_flash` (tests/e2e/flash_synth.mjs already has a harness pattern).
  - **Subtle case:** a prefill column's attention output then differs slightly from what a single-token pass would produce. That is already true of the GEMM path.
- **Plan.**
  1. `engine/wgsl/qwen35.js`: add `attn_flash_tile` (bindings as `attn_flash`: q, kCache, vCache, faO, faML, faUB) with the grid (splits, nKV). Each workgroup handles every column of the pass for one (split, KV head).
  2. `engine/qwen35.js`: build the pipeline in `G1` and a bind group in `_initBatch` next to `flashT2`. In `_encodeLayerBatch`, when `nCols === this.NC && this.attnTileP !== false`, dispatch it instead of `attn_flash` / `attn_flash_t2`. Add a runtime kill switch `engine.attnPrefillTile = false`.
  3. Add `tests/test_attn_tile.js` (random Q/K/V against the `attn_flash` path, relDiff), plus `prof_prefill.js` and the goldens.

- **Status (branch prefill/flash-attn): GPU-validated on the GB10 and on by default.** Implemented as `attn_flash_tile` + `attn_combine_tile` in `engine/wgsl/attn_tile.js`, behind `attnPrefillTile` (default true; `ATTN_PREFILL_TILE=0` for every Deno test that loads through `tests/load_model.js`, `?attnptile=0` / `ATTN_PREFILL_TILE=0` for the Chrome bench, or `engine.attnPrefillTile = false` at runtime). The WGSL compiles in Deno/wgpu and in Chrome 131 (Tint), TK 8 and 16, both shapes.
  - Goldens with it on: `run.sh quick` 8/8, `run.sh q38` 9/9 (MATCH, spec == plain, twins bit-identical, test_batch_q38 relDiff 1.74e-7), `test_moe.js` 3/3 MATCH llama.cpp with spec identical and acceptance unchanged (28/33, 28/39, 25/45).
  - Logits vs `attn_flash` (`tests/test_attn_tile.js`, last prompt token): 27B 1.61e-4 (700), 2.48e-4 (3000), 9.62e-5 (16000). The MoE sits at 4.1e-3 (700, 3000) and 1.6e-2 (16000) from `attn_flash`, but that is the router: `tests/calib_attn_tile.js` shows `attn_flash`'s own batched prefill is 4.14e-3 from decoding the same 700 tokens one at a time while the tiled path is 1.29e-4 (3000: 6.6e-3 vs 5.6e-3; 16000: 1.84e-2 vs 1.40e-2). A near-tie in the top-8 routing flips an expert either way. Argmax and greedy continuations are identical at every length on both models.
  - Measured (GB10, Deno, `bench_ctx.js` CTX=16640 TOKENS=8; each prefill figure is the rate over the segment ending at that fill): 27B 65.9 / 57.4 / 30.2 → 69.8 / 74.3 / 66.1 tok/s at 512 / 4096 / 16384; MoE 151.3 / 133.3 / 80.8 → 159.7 / 166.0 / 117.2. Whole-prompt 16000-token prefill (`test_attn_tile.js`): 27B 39.7 → 69.1, MoE 99.5 → 153.9. Attention kernel time at 4096 (`prof_prefill.js`): 27B 14.5 s → 2.2 s (22.1% → 4.2% of prefill), MoE 4.8 s → 1.0 s (17.2% → 4.3%). Decode is unchanged (it never uses the kernel).
  - `ATTN_PREFILL_TK=16` compiles and is numerically fine on the 27B (1.78e-4 at 3000), but its speed was only measured while another job shared the GPU, so it stays opt-in.
  - Differences from the plan above: one workgroup per (split, KV head, group of up to 64 query rows = `floor(64/G)` columns × G heads), and 4 threads per row, each keeping a quarter of q and of the output in registers for the whole split. That avoids the S = Q·Kᵀ micro-GEMM's Q staging, which does not fit 16 KB. Per tile of TK = 8 positions (16 when the device grants 32 KB of workgroup memory), K and then V are staged once as f32 vec4s. Partial dots go through workgroup memory, and every row runs its online softmax per tile.
  - The split length is derived from the pass (about 32 splits, at most `faSplit`, within the `faO` slot count), so short contexts still launch enough workgroups.
  - Its own shader module: if it fails to compile, the option turns itself off with a warning.
  - CPU check: `tests/unit/attn_tile_test.js` runs the generated kernel bodies as JavaScript against exact attention. GPU A/B: `tests/test_attn_tile.js`.

### (B) Wide prefill ubatch with shared-memory tiled GEMMs (dequantize once)

- **Why.** The GEMMs are 81% of 27B prefill at 512 and still 43% at 16k. They sit at 3.0 to 5.8 TFLOPS. A 16-column pass cannot amortize dequantization well, and down-shaped and Q8 shapes lag.
- **What.** A prefill mode that processes U = 128 to 512 columns per layer instead of 16:
  - Classic BM×BN tiles: 64×64 at 4×4 per thread, or 128×128 at 8×8. Q4_0 and Q8_0 are dequantized once into shared memory as k-major `vec4` over rows. Activations are column-major. The K loop is fully unrolled.
  - Measured first cut: **6.9 to 7.2 TFLOPS on the 27B shapes at N ≥ 256** [M]. That is 1.25× the gate/up GEMM today, 2× `ffn_down` Q4, and more than 2.3× the Q8 shapes (3.0 → about 7 with a Q8 twin [E]).
  - With tuning (double-buffered stages, 2 workgroups per SM with a 48 KB limit, BK = 32 vs 64 sweep, bank-offset padding), 10 TFLOPS is a reasonable target [E], at 60% of the measured peak.
- **Expected [E].** 27B GEMM work is 48.6 GFLOP per token: FFN 34.2, DeltaNet projections 11.0, attention projections 3.4. Today it takes 10.8 ms per token (4.5 TFLOPS effective).

| Stage | 27B GEMM time per token | 27B end to end |
|---|---|---|
| Today | 10.8 ms (4.5 TF effective) | 73 tok/s at 512 |
| B at 7 TF | 6.9 ms | about 105 tok/s at 512 |
| B at 10 TF | 4.9 ms | about 125 at 512 |
| B at 10 TF with E | 4.9 ms | **about 150 at 512, about 125 at 16k** |

  - MoE dense projections: 2.55 GFLOP per token, from 2.08 ms per token (GEMV) → 0.98 (G1) → about 0.36 [E].
- **Risk: medium-high.** A wide ubatch touches the whole batched path.
  - Buffers are sized NC today (`_initBatch` `mkB`), and there are frame buffers per column (`frameBufsB`, one uniform per column).
  - Attention must handle U queries with causal masking inside the ubatch (E becomes a prerequisite).
  - The DeltaNet kernels already loop over `frame.nCols`, but `dn_delta_mc` walks U tokens serially on 32 to 48 workgroups: about 6 µs per token per layer, which is fine up to about 1,000 tok/s (see A).
  - Memory: U = 256 at 17408-wide FFN activations is 17.8 MB per buffer, which is fine.
- **Exactness.** Summation order changes: tile K order and no split-K. The ubatch is prefill-only (like the GEMM today), so decode and verify are unchanged. It needs the prefill tolerance tests and the 27B and MoE goldens. If any golden flips, ship it off by default (`prefillUbatch: 16` default, `256` opt-in).
- **Plan.**
  1. New `engine/wgsl/gemm_wide.js`: a `gemmWideWGSL({BM, BN, TM, TN, KB, q8})` generator. Port it from `tests/bench/bench_wide_gemm.js`, add the Q8_0 twin, reuse the `matvec_q4_coop_b` binding layout (qs, sc, xT, y, shape), and use the fused `_acc` epilogue for the residual. The engine's column-strided y layout can be written directly: no split-K, so no reduce.
  2. `engine/qwen35.js`:
     - `prefillUbatch` option (default 16). `_initBatch` sizes the prefill-only buffers (`B.*` for x, xn, g, u, qkv, convOut, z, dOut, gated, qFull, q, gAttn, k, v, attnOut, and the xT transposes) to U, kept separate from the NC-sized verify buffers so verify memory does not grow.
     - A per-pass uniform `{basePos, nCols}` replaces the per-column `frameBufsB` for the `_mc` kernels, which read `frame.seqLen + col` today.
     - A new `_encodeLayerWide(enc, i, basePos, U)` next to `_encodeLayerBatch`. `prefillTokens` uses it for full U chunks, then falls back to the 16/8/4 tail.
  3. The `_mc` glue kernels (rmsnorm_mc, dn_pre_mc, dn_gatenorm_mc, attn_glue, kv_store) already take a column count through the dispatch y dimension. Check the 65535 limit: U ≤ 512 is fine.
  4. Replace the 16 per-column `silu_mul` dispatches with one multi-column dispatch. This is exact and worth 2.5% on the 27B today; see G2.
  5. Tests: extend `tests/test_batch_q38.js` / `test_gemm.js` with a U-wide pass against sequential (relDiff), then `run.sh q38`, `test_moe.js` and `bench_ctx.js`.

#### B as implemented (branch prefill/gemm-tiles): opt-in, not yet measured on a GPU

- **Option.** `prefillUbatch: U` (engine create option, default 0 = off; U a multiple of the tile width BN = 64, e.g. 256) and `prefillTile: { BM, BN, TM, TN, KB }`. `engine.prefillWide = false` turns it off at runtime. Deno tests and benches: `PREFILL_UBATCH=256 [PREFILL_TILE=json] [WGMEM=0]` (tests/load_model.js `wideOpts`); Chrome bench: `?ubatch=256&prefilllen=2048`.
- **Kernel** (`engine/wgsl/gemm_wide.js`): the bench_wide_gemm.js tile (BM×BN = 64×64, 4×4 per thread, 256 threads), ported to the engine: Q4_0 and Q8_0 (K-quants are Q8_0 after load), `=` and `+=` (residual) variants, activations read straight from the engine's column-strided layout (no transpose kernel), output written in the engine's layout (no split-K reduce), rows clamped for any dOut. KB = 2 quant blocks per K stage when the device has 32 KB of workgroup memory (the test/bench devices request the adapter's limit when PREFILL_UBATCH is set), else KB = 1 within the 16 KB default. One f32 accumulator per output, k ascending.
- **Pass structure** (`engine/qwen35.js` `_encodeLayerWide`, `_prefillWide`): `prefillTokens` first runs chunks of up to U tokens (multiples of BN). Per layer: one wide rmsnorm, every Q4/Q8 projection as ONE tiled GEMM over the chunk (merged [q|k|v]-style fuseProj weights as one launch), one multi-column SiLU (G2 for the wide path); the DeltaNet conv / gates / recurrence / gated norm, the β/α GEMV, the attention glue / KV store / flash / combine, the MoE router and experts and the MTP draft-cache fill run on the **unchanged** 16-column kernels, sub-batch by sub-batch in order, each with its own frame uniform. Every batched tensor has a U-wide twin with the same column stride, so a sub-batch moves with one `copyBufferToBuffer` per tensor. The tail (< 64 tokens) takes the existing 16 / 8 / 4 / 1 path.
- **Exactness.** Decode and verify never use it. Only the projection sums change order (and the MoE's dense projections move from GEMV to GEMM, like G1). CPU checks (no GPU): `tests/unit/gemm_wide_test.js` runs the generated kernel body as JS for Q4/Q8, both epilogues and five tile shapes (exact index math, clamped tails, padded strides untouched, no unstaged shared-memory read) and estimates f32 error on a 27B ffn_down row: wide order 4.4e-6 vs f64, 16-column GEMM 2.1e-6, wide vs 16-column 4.9e-6. `tests/unit/prefill_wide_test.js` encodes whole prefills on a recording mock device (synthetic dense, Q8-out and MoE models): valid commands, and every non-projection kernel dispatched with the same bind group, grid and frame, in the same order, as the default path; with the option off the command stream is identical to the pre-change engine (also checked against 4c9a991 on a 27B-shaped synth).
- **Expected [E].** 27B at 512: projections 10.8 → about 7.5 ms per token at 6.5 TFLOPS, minus the xposes and 16 SiLU launches: **73 → about 95 to 105 tok/s**; 16k: 39 → about 45 (attention, E, is the wall). MoE at 512: dense projections 2.08 → about 0.5 ms per token: **166 → about 220 tok/s** (more than G1's 198, which it subsumes for the wide path); experts unchanged until D.
- **Still to do on a GPU:** `tests/test_prefill_wide.js` (MODEL=27b and moe), the goldens with and without PREFILL_UBATCH, `bench_ctx.js`, and a tile sweep (`PREFILL_TILE`). `prof_prefill.js` samples 16-column passes and does not instrument wide chunks yet.

### (D) MoE prefill grouped by expert (llama.cpp `mul_mat_id`)

- **Why.** Experts are 46% of MoE prefill at 512 tokens (40% at 4k). `moe_gus` already runs at about 266 GB/s, the DRAM peak, because every (column, slot) re-streams its expert.
- **Reuse depends on ubatch width.** The expected number of distinct experts per layer for U tokens is 256·(1 − (1 − 8/256)^U):

| U | Distinct experts | Reuse | Expert traffic |
|---|---|---|---|
| 16 | about 102 of 128 assignments | 1.25× | about 0.72× of today, including the shared expert once |
| 128 | about 251 of 1024 | 4.1× | |
| 512 | 256 of 4096 | 16× | |

- **What.**
  1. A `moe_sort` kernel builds per-expert token lists (counting sort over U·K (token, slot) pairs, stable by token), plus offsets and an indirect-dispatch argument buffer.
  2. A `moe_gemm_gu` kernel does one tile row-block per (expert, 64-token tile): it gathers the tokens' `xn` columns, runs the tiled Q4/Q8 GEMM for gate and up, and applies SiLU·mul.
  3. A `moe_gemm_dn` kernel runs the down GEMM and scatter-adds with the router weights. For a fixed order, write per-(token, slot) outputs and reduce in slot order, the way `moe_dnc` does. That keeps it deterministic.
  4. The shared expert is just a dense GEMM over all U tokens.
- **Expected [E].** Expert FLOPs are 2.26 GFLOP per token (8 routed plus shared, over 40 layers).
  - With U = 512 and about 4 TFLOPS (ragged tiles, many experts with 8 to 24 tokens), that is about 0.57 ms per token against 2.67 today: **about 4.7× on experts**.
  - At U = 16 (no ubatch change), only about 1.3×: 2.67 → about 2.0 ms per token, +13% end to end.
  - MoE end to end with G1+B+D+E at U = 512: about 1.7 ms per token, **about 550 to 600 tok/s at 512** and about 500 at 16k, against 166 / 98 today and llama.cpp's 2,356.
- **Risk: medium-high.** It needs indirect dispatch or a worst-case grid with early exit, and ragged tiles. Router top-k ties must match `moe_route` exactly (reuse its selection output: only the GEMMs change).
- **Exactness.** Per-token expert sums change order (tiled K) relative to the `moe_gus` GEMV. Prefill only; verify and decode keep `moe_gus` / `moe_dnc`. It needs the `test_moe.js` 3× MATCH, and stays off by default if a golden flips.
- **Plan.**
  1. New `engine/wgsl/moe_grouped.js` (moe_sort, moe_gemm_gu, moe_gemm_dn), reusing the tile code from `gemm_wide.js`.
  2. `engine/qwen35.js`: `_encodeLayerWide` MoE branch: router GEMM (G1), `moe_route` (unchanged), `moe_sort`, the grouped GEMMs, then a shared-expert dense GEMM and the combine.
  3. `tests/test_moe.js` plus a new `tests/test_moe_grouped.js` (grouped against per-column on random routing, relDiff; count of experts touched).

- **Status (branch prefill/moe-group): implemented, off by default, not yet measured on the GPU.** It does not wait for B:
  - **Layer-major ubatches instead of a wide GEMM.** `moeGroupPrefill: U` (for example 256) runs prompt prefill in ubatches of U tokens, layer by layer. Each layer's attention / DeltaNet part still runs as the usual 16-column passes, with the same kernels and a frame uniform per sub-pass. Only the MoE FFN runs once over all U tokens: `moe_gsort` (one workgroup, a stable counting sort of the U·9 (token, slot) pairs by expert, with the shared expert as expert 256), then `moe_gusg` / `moe_dng` (one workgroup per row block and chunk of up to `moeGroupUC` = 8 pairs of one expert, launched indirectly), then `moe_combw`. The last ubatch is cut to the remaining whole passes. Under 32 tokens goes the old way.
  - **Bit-exact by construction, not within tolerance.** Each pair keeps `moe_gus` / `moe_dnc`'s arithmetic: the same thread map, the same term template (`termW`), the same block order, the same reduction tree and the same combine expression and order. Only the weight loads are shared. So the goldens cannot move, and spec == plain is untouched. `tests/unit/moe_group_test.js` runs both paths' generated kernel bodies on the CPU and requires `===`. `tests/test_moe_group.js` requires bit-identical logits on the GPU.
  - **Expected [E].** Streamed expert weights per ubatch drop from U·9 pair streams to about (distinct experts × chunks) ≈ U·9 / 5 at U = 256. The kernels keep the old per-pair reduction and activation reads, so the gain is bounded by those, not by DRAM. Estimate: experts 1.5 to 2.5× faster, MoE prefill about +25 to 40% at 512 tokens and +10 to 20% at 16k (attention-bound). It stacks with G1 and E.
  - **Validate / A-B.** `MOEGROUP=256` in `prof_prefill.js` and `bench_ctx.js`; `MOEGROUP=16` in `test_moe.js` (BCOLS 4); `?moegroup=16&batchcols=4` in `bench.html` (chrome_bench.mjs argv[4]); `engine.moeGroup = false` at runtime.

### (C) f16 compute and Chrome's subgroup-matrix (tensor cores), with fallback

- **f16 ALU: no gain here.** Measured `vec4<f16>` FMA reaches 8.6 TFLOPS against 16.7 for f32 in Deno/wgpu. NVIDIA consumer parts run non-tensor FP16 at the FP32 rate anyway. f16 is only useful as a storage format for shared tiles, meaning bigger tiles in the 16 KB default. It would change numerics, so it is not recommended as a compute type.
- **Subgroup-matrix (`chromium-experimental-subgroup-matrix`, Chrome/Dawn only, behind a flag).** f16 × f16 → f32 MMA on tensor cores. llama.cpp gets about 43 effective TFLOPS on the 27B (879 tok/s × 48.6 GFLOP) with int8 MMA.
  - A WebGPU port at 20 to 30 TFLOPS would put 27B GEMMs at about 2 ms per token [E]: **about 250 to 300 tok/s at 512 with E**.
  - It can't be measured in Deno (the feature is not exposed), so this estimate is **UNCERTAIN**.
  - Numerics: activations rounded to f16 → off by default, opt-in per device, with the f32 tiled GEMM as fallback. A room must agree, because pinned numerics are a protocol matter.
- **dp4a (`dot4I8Packed`, core WGSL, works in Deno and Chrome): measured 59.7 TOPS, 3.6× f32 FMA.**
  - An MMQ-style path (activations quantized to Q8_1 per 32 values, int32 accumulate, scale per block) is what llama.cpp's CUDA MMQ does. At 20 to 25 effective TOPS [E], 27B GEMMs take about 2.2 ms per token: **about 250 tok/s at 512 with B's ubatch and E**.
  - Numerics: activation quantization changes logits. It must stay off by default unless the goldens (MATCH vs llama.cpp) happen to hold. They might: llama.cpp quantizes activations exactly this way for pp. That can only be tested, not assumed.
- **Plan.**
  1. `engine/wgsl/gemm_mmq.js`: `quant_q8_1` (activation quantize, per 32-column block) and `gemm_mmq_q4` / `gemm_mmq_q8` using `dot4I8Packed` with the tiling from B.
  2. `engine/qwen35.js`: `prefillMath: "f32" | "dp4a" | "sgmatrix"` (default "f32"), feature-probed. The subgroup-matrix kernel goes behind `adapter.features.has("chromium-experimental-subgroup-matrix")`.
  3. Measure in `tests/bench/chrome_bench.mjs` (Chrome) and `prof_prefill.js` (Deno, dp4a).

### (A) Chunked / parallel Gated DeltaNet prefill (flash-linear-attention chunk form)

- **Why it is low today.** `dn_delta_mc` plus conv plus glue is 3.7% (27B) and 5.0% (MoE) of prefill at 512, and less at longer lengths. The recurrence costs about 6 µs per token per layer: 0.29 ms per token on the 27B, 0.17 on the MoE.
- **When it matters.** It becomes significant only after B/D/E. Once the 27B reaches about 5 ms per token it is about 6%; at the dp4a / tensor-core level (about 3 ms per token) it is about 10%. For the MoE at about 1.7 ms per token it is also about 10%.
- **What.** The WY/UT chunk form (chunk C = 64):
  - Per chunk, compute the intra-chunk attention A = tril(K·Kᵀ ⊙ decay) and solve (I + βA)·U = … by forward substitution in shared memory.
  - Then state updates as dense 128×128 GEMMs: S ← γ·S + Kᵀ·(U − W·S), and outputs O = Q·S + tril(Q·Kᵀ)·U.
  - It turns 64 serial rank-1 steps into about 4 small GEMMs per chunk, run in parallel across heads and chunks for the intra-chunk parts. The inter-chunk state pass stays sequential over chunks (U/64 steps).
- **Expected [E].** 3 to 5× on the recurrence at U ≥ 256: 0.29 → about 0.08 ms per token on the 27B. That is **about +4% end to end after B/E, about +8% at the tensor-core level**. It is not worth doing before B.
- **Risk: high.** Numerically sensitive (the decay products need log-space or chunk-local scaling, as fla does). The state has to match decode closely, because decode continues from the prefill state.
- **Exactness.** It changes the state's summation order. The state carries over into decode, so any drift persists. Off by default until the goldens and a long-context state-drift test (prefill 16k chunked vs sequential, then 256 greedy tokens identical) pass.
- **Plan.**
  1. `engine/wgsl/qwen35.js`: `dn_chunk_intra` (per (head, chunk): the A matrix, the triangular solve, W and U), `dn_chunk_state` (sequential over chunks per head: state GEMMs, keeping the state in registers like `dn_delta_mc`), and `dn_chunk_out`.
  2. `engine/qwen35.js`: used only in `_encodeLayerWide` when `U ≥ 128 && dnChunk === true`.
  3. `tests/test_dn_chunk.js`: chunked vs `dn_delta_mc` on random inputs, state and output relDiff.

### (F) Pre-baked system-prompt state for Code mode

- **Today.**
  - `harness/statecache.js` (OPFS) stores per-session states.
  - `harness/prefix.js` / `room.js` `ckpt*` resume from the longest checkpoint, with the saved checkpoints in GPU memory.
  - Code mode's system prompt plus 8 tool schemas is held under about 1,200 tokens (`harness/code-prompt.js`).
  - A new session or a fresh page load still prefills it: **about 16 s on the 27B and about 7 s on the MoE at today's 512-length speeds** [E, from the measured 73 / 166 tok/s].
- **What.** On the first Code-mode prefill of a given (model, `stateSignature()`, token ids of system + tools), `exportState()` right after the system block. Store it under `tokenKey(sig, ids, model)` in a dedicated `StateCache({ dirName: "tabby-sysprompt" })`, and register it in the `PrefixIndex` at startup, so every later session and page load resumes from it.
  - The state is about 236 MB on the 27B (DeltaNet 48 × 48 × 128² × 4 B = 151 MB, conv 6 MB, KV 16 × 1,200 × 1,024 × 2 × 2 B = 79 MB), and about 100 MB on the MoE.
  - The restore is an OPFS read plus a GPU upload, **about 0.3 to 1 s [E]**.
  - In a room, each device stores its own layers under the same key. This is already the statecache design.
  - Shipping a prebuilt state with the model is possible, but it is tied to the exact prompt bytes and the device's layer split, so compute-once-per-device is the robust form.
- **Expected.** It saves 6 to 15 s on the first request of every session. It does not help arbitrary prompts.
- **Risk: low.** It invalidates automatically when the prompt, tools or engine signature change (the key is a hash).
- **Exactness.** Bit-exact: it is the same state the prefill would have produced on this device. This holds as long as `stateSignature()` covers the prefill kernel choice (GEMM/ubatch on or off). **Add the prefill mode to `stateSignature()`**, or a state made by the old path would be restored into a new-path session. The results differ only within prefill tolerance, but "identical to a fresh prefill" would no longer hold.
- **Plan.**
  1. `harness/engine-model.js` / `harness/agent.js`: after tokenizing, record the system-block length.
  2. `harness/sessions.js`: `ensureSystemState(ids.slice(0, sysLen))`, which exports and puts once, and `resume()` consults it.
  3. In the `room.js` room path, the host broadcasts the key and devices load their part or report a miss (the existing `ckpt` protocol).
  4. `engine/qwen35.js` `stateSignature()`: include `gemm`, `prefillUbatch` and `attnPrefillTile`.
  5. Tests: `tests/unit/` statecache round trip plus an e2e prefix resume.

### (G) Other findings from the profile

- **G2: one multi-column `silu_mul` on the GEMM path.** Exact. `_encodeLayerBatch` issues 16 `silu_mul` dispatches per FFN layer when `G` is on: 168 ms, **2.5% of 27B prefill at 512**.
  - Add `silu_mul_mc` (the MC uniform with the column stride) and dispatch it once over `nCols`. The same arithmetic per element, so it is bit-identical.
  - Files: `engine/wgsl/qwen35.js` and `engine/qwen35.js` (`_encodeLayerBatch`, the FFN and shared-expert branches).
- **G3: MTP draft-cache fill is 2.6 to 5.7% of prefill.** At 16k it is 17 s on the 27B and 9.4 s on the MoE, mostly its attention.
  - The draft only affects acceptance, never outputs, so fill the MTP cache only for the last `mtpFillTail` tokens of a prompt, for example 2,048. Drafts beyond that window see a truncated draft context.
  - Expect 3 to 5% of prefill at 4 to 16k. Acceptance must be measured with `bench_ctx.js` (spec identical is guaranteed).
  - Files: `engine/qwen35.js` `prefillTokens` (skip `_mtpFillBatch` while `i + NC < ids.length − mtpFillTail`). The MTP attention also needs a start offset, or its KV below the window must be zero-masked. Simplest: keep writing its KV with the kv projection only, not the full layer. The cheap form is `mtpFill: "tail"` off by default until acceptance is shown.
- **G4: `dn β+α` (96×5120 f32 GEMV) runs at 0.3 TFLOPS.** 1.2% on the 27B. Converting these f32 weights to f16 at load would halve the bytes, but that changes numerics. Fusing them into the qkv GEMM tile as extra rows is a better fit for B.
- **G5: rmsnorm_mc (24 µs per call on the 27B) plus the xposes (8 µs) are about 2.4%.** Fold the transpose into the rmsnorm epilogue: rmsnorm writes both the column-strided and the column-major copy. Exact.
- **Not a bottleneck:** CPU encode, submit and sync overhead (1 to 5%), DeltaNet conv (0.2 to 0.4%), attention glue (0.1 to 0.3%).

## Summary: ranked by expected win

| # | Candidate | Model | Speedup at 512 / 4k / 16k [E unless marked] | Risk | Exactness | Default |
|---|---|---|---|---|---|---|
| 1 | **G1** MoE projections on the existing GEMM | MoE | **1.19× / 1.16× [M]** / about 1.1× | low | prefill order (like the 27B GEMM) | on if the goldens pass |
| 2 | **E** tiled causal flash (prefill passes) | both | 27B 1.08 / 1.25 / **1.8×**; MoE 1.06 / 1.19 / **1.6×** | medium | prefill order | on if the goldens pass |
| 3 | **G2** one multi-column silu_mul | 27B | 1.025× | trivial | bit-identical | on |
| 4 | **B** wide ubatch (U = 256) plus tiled GEMM (7 → 10 TF) | both | 27B 1.4 to 1.7× on top of E; MoE projections 2.7× | medium-high | prefill order | on if the goldens pass, else opt-in |
| 5 | **D** grouped MoE experts (needs B's ubatch) | MoE | about 4.7× on experts: MoE total with G1+B+D+E about 3.5× at 512 (about 580 tok/s), about 5× at 16k | medium-high | prefill order | on if the goldens pass |
| 6 | **F** pre-baked Code system-prompt state | both | −6 to −15 s on each new session's first request; bit-exact | low | exact (with signature fix) | on |
| 7 | **C** dp4a MMQ / subgroup-matrix | both | 27B GEMMs 2 to 2.5× beyond B (about 250 tok/s at 512) | high | activations quantized to int8/f16 | opt-in |
| 8 | **G3** MTP fill on the prompt tail only | both | 1.03 to 1.05× at 4 to 16k | low | outputs exact, acceptance may drop | opt-in |
| 9 | **A** chunked DeltaNet | both | +4 to 10% once 1 to 7 land; ≈0 today | high | state order (persists into decode) | opt-in |

**Recommended order:**

1. G1 and G2: days, with a measured win.
2. E: the long-context wall.
3. B: the ubatch refactor is the enabler.
4. D on top of B.
5. F: independent, small.
6. C (dp4a first, because it works in Deno and every browser).
7. A last.

**Realistic targets [E]:**
- 27B: about 150 tok/s at 512 and 125 at 16k after 1 to 4, and about 250 with 7.
- MoE: about 550 to 600 tok/s at 512 after 1 to 5.

That is still 3.5 to 4× behind llama.cpp's int8 tensor-core MMQ. Closing the rest needs tensor cores (subgroup-matrix), which WebGPU exposes only experimentally.

## Raw data

- `tests/prof_prefill.js` output for both models: the full per-kernel top lists and the projection efficiency tables are printed by the tool. Rerun them with the commands above: about 7 min for the MoE and 15 min for the 27B at 16k.
- The G1 measurement: `GEMM_EXTRA='{"8192x2048":2,"4096x2048":4,"2048x4096":8,"512x2048":16}' MODEL=moe LENS=512,4096`.
  - Result: 198.1 / 166.8 tok/s.
  - dn_qkv 146.9 µs (3.65 TF), dn_z 87 µs, dn_out 88.6 / 98.6 µs (Q4/Q8), attn q+gate Q4 147 µs.
- The wide GEMM: `SHAPES=17408x5120,... NS=64,128,256,512 CFGS='[{"BM":64,"BN":64,"TM":4,"TN":4,"KB":1},{"BM":128,"BN":128,"TM":8,"TN":8,"KB":1}]' deno run --unstable-webgpu --allow-read --allow-env tests/bench/bench_wide_gemm.js`.

#### B measured on GB10 (2026-09-27)

The WGSL compiled and ran first try on Deno/wgpu; 0 GPU errors on every run. Default tile changed to
KB 1 (one quant block per K stage, 16 KB workgroup memory): the 27B prefill of 1024 tokens ran 90.4 tok/s
with KB 1 vs 77.3 with KB 2 (default 16-column path 72.0). Tile sweep (`tests/bench_wide_tiles.js`,
27B, 1024 tokens): 64x64/4x4 KB1 90.4 · 128x64/8x4 KB1 86.7 · 64x64/8x4 KB1 86.8 · 64x32 KB1 86.0 ·
128x128/8x8 KB1 84.8 · 128x64/8x4 KB2 81.0 · 64x64 KB2 77.3 · 64x128 KB2 76.0 · 128x32 KB2 65.9.
MoE: 64x64 KB1 223.4, KB2 220.3, 128x64 KB1 221.2, default 163.3.

`bench_ctx.js` (FILLS=512,4096,16384 TOKENS=8; 4096/16384 are the rate of that segment), U=256:

| | 512 off | 512 on | 4096 off | 4096 on | 16384 off | 16384 on |
|---|---|---|---|---|---|---|
| 27B prefill tok/s | 66.7 | 80.0 | 59.7 | 71.8 | 35.2 | 39.0 |
| 27B plain decode | 8.79 | 8.69 | 8.50 | 8.50 | 7.40 | 7.47 |
| MoE prefill tok/s | 153.0 | 181.6 | 140.8 | 183.1 | 89.1 | 105.3 |
| MoE plain decode | 24.50 | 23.92 | 24.14 | 22.92 | 20.34 | 21.38 |

Spec identical to plain at every fill, both ways. Wide GEMMs reach 5.5-6.0 TFLOPS (Q4 FFN) in
`PREFILL_UBATCH=256 prof_prefill.js`, which now profiles wide chunks. At 4096 on the 27B the FFN GEMMs are
still 44% and attention 28% of kernel time; on the MoE, experts are 51% and attention 24%.

Correctness (`test_prefill_wide.js`, LENS=150,700,2100): the 27B max logits relDiff wide vs default is
5.2e-4, argmax equal, greedy and spec identical; PASS. The MoE is argmax-equal and greedy/spec identical,
but relDiff at 2100 tokens is 4.04e-3 (vs sequential: default 8.1e-4, wide 3.97e-3), above the 2e-3 gate.
It is the same for U=128/256/512, so it comes from the GEMM's summation order vs the GEMV (a near-tie
amplified through routing is likely), not from chunking. Under the policy the MoE keeps it off by default.
Goldens with PREFILL_UBATCH=256: run.sh quick 8/8, run.sh q38 9/9, test_moe 3/3 MATCH, spec identical.
