# Qwen 3.6 35B-A3B: where decode time goes, on one GPU and across devices

Measured 2026-09-27 on the GB10 (DGX Spark), branch `exp/base` (= `feat/engine-opt` at 8cdc1f2 plus the profiling tools below; no engine or room code changed). Model: bartowski `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (40 layers + MTP, 256 experts top-8 + shared expert, hidden 2048, vocab 248,320). Prompt: the two-sum chat prompt from `tests/test_moe.js`, greedy. Reference: llama.cpp CUDA 85.5 tok/s plain (11.7 ms per token).

## Tools (new, all opt-in, nothing runs them automatically)

| file | what it does |
|---|---|
| `tests/prof/gpuprof.js` | Profiler for Deno and the browser. Counts every `queue.submit`, `mapAsync` and `writeBuffer`, labels each with the engine method that created it, and timestamps on the GPU. **submit mode**: an empty timestamped compute pass at the start and end of each command encoder, so every submit gets its real GPU span (the engine's passes are untouched). **kernel mode**: every dispatch in its own timestamped pass (like `tests/prof_ts.js`), attributed to a category (router, expert gate/up, attention, DeltaNet, head, MTP draft...) from the engine op or layer it was encoded for. |
| `tests/prof/moe_decode_prof.js` | The decode profile: clean wall time, then submit mode (plain token, speculative step), kernel mode, and the sync floor. |
| `tests/prof/prof_moe_decode.js` | Deno runner. `cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write prof/prof_moe_decode.js out.json` |
| `tests/bench/prof.html`, `tests/prof/prof_chrome.mjs` | The same profile in Chrome with the real GPU: `node tests/prof/prof_chrome.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf out.json` |
| `tests/e2e/room_prof.mjs` | Room hop profiler. N devices, one headless Chromium each (own profile, own GPU process, like separate machines), real PeerJS and WebRTC on loopback, the one GPU. Serves `room.js` and `room/transport.js` with trace marks added at serve time (the files on disk are not changed), times every submit and `mapAsync` per tab, puts a GPU timestamp pair around every command buffer, and aligns the browsers' clocks with a measured offset (round trip 0.8 to 1.1 ms). Plain and speculative answers, 48 tokens each, after a warm-up answer. `node tests/e2e/room_prof.mjs --model qwen3.6-35b-moe --devices 2 --out trace.json` |
| `tests/e2e/room_prof_report.mjs` | Tables from a trace: every lap split into segments, medians. |
| `tests/prof/hop_cpu_bench.mjs` | CPU cost of the per-hop JS work (f16 pack/unpack, NaN scan, readback copy, slicing), Node, no GPU. |

Chrome's `performance.now()` is coarsened to 0.1 ms in these pages, so the room's CPU-side segments are only good to 0.1 ms. GPU spans come from timestamp queries, with `--enable-webgpu-developer-features` so they are not quantized.

## 1. One device

### Summary

| | Deno (wgpu) | Chrome |
|---|---|---|
| plain, ms per token (tok/s) | 35.9 (27.8) | **22.1 (45.2)** |
| GPU busy per plain token (sum of submit spans) | 19.8 ms (56 %) | 18.8 ms (85 %) |
| CPU encode per plain token | 2.8 ms | 0.28 ms |
| submits / mapAsyncs per plain token | 2 / 1 | 2 / 1 |
| speculative K=3, ms per step, tok/step, tok/s | 60.6, 3.67, 60.5 | **48.6, 3.67, 75.5** |
| GPU busy per speculative step | 44.1 ms (72 %) | 42.4 ms (87 %) |
| submits / mapAsyncs per step | 2.25 / 1 | 2.25 / 1 |
| sync floor: empty submit + 4-byte map | 11.7 ms | 0.93 ms |
| 1 MB logits readback (submit + map + copy) | 12.1 ms | 3.25 ms |

Acceptance 37/42 on both; `draftVocab` off (the bench default; the room uses 65,536). Deno is sync-bound (11.7 ms per sync, as the bench log already says), so everything below is Chrome.

The engine's default decode is already one submit per plain token and one per speculative step (`_verifyFused` runs the K drafts, the 4-column trunk and the 4-column head in one command buffer; `_mtpRefill` and `_restoreDN` follow without a sync). What is left between the 18.8 ms of GPU and the 22.1 ms wall is almost entirely the 1 MB logits readback: `mapAsync` resolves 21.0 ms after the forwardToken submit, 2.2 ms after the GPU span ends. A speculative step reads 4 MB of logits (4 columns), and its map resolves 45.4 ms after submit for 41.1 ms of GPU.

### GPU time per category (Chrome, kernel mode)

Every dispatch in its own pass, so the sums (21.2 and 42.9 ms) are a little above the real spans (18.8 and 42.4 ms).

| category | plain token: ms | dispatches | % | K=3 step (4 columns + 3 drafts): ms | dispatches | % |
|---|---|---|---|---|---|---|
| DeltaNet projections (qkvz, ba, out GEMVs) | 4.69 | 90 | 22.1 | 5.84 | 90 | 13.6 |
| DeltaNet core (conv, dn_pre gates + L2, delta + gate norm) | 3.48 | 90 | 16.4 | 4.84 | 135 | 11.3 |
| MoE experts gate/up (+ shared, fused `moe_gus`) | 3.27 | 40 | 15.4 | 8.09 | 40 | 18.9 |
| MoE experts down + combine (+ shared, fused `moe_dnc`) | 2.24 | 40 | 10.6 | 4.98 | 40 | 11.6 |
| LM head (final norm + GEMV) | 2.21 | 2 | 10.4 | 2.45 | 2 | 5.7 |
| attention projections (q, kv, o) | 1.41 | 30 | 6.6 | 1.73 | 30 | 4.0 |
| norms | 1.33 | 80 | 6.3 | 1.13 | 80 | 2.6 |
| attention core (flash, glue, kv store) | 1.19 | 50 | 5.6 | 1.31 | 50 | 3.1 |
| MoE router GEMV (router + shared gate) | 0.76 | 40 | 3.6 | 1.04 | 40 | 2.4 |
| MoE router top-k (`moe_route`) | 0.65 | 40 | 3.1 | 0.62 | 40 | 1.5 |
| MTP draft (3 drafts: block + full LM head + argmax) | — | — | — | 10.89 | 82 | 25.4 |

Top pipelines, plain token (Chrome):

| pipeline | ms | dispatches | µs each | note |
|---|---|---|---|---|
| matvec_q4_coop | 3.32 | 36 | 92 | DeltaNet / attention projections |
| moe_gus_q4_q8 | 3.27 | 40 | 82 | 8 experts + shared gate/up |
| matvec_q8_coop | 2.79 | 15 | 186 | includes the LM head (~2.2 ms) |
| dn_pre | 1.93 | 30 | 64 | **one workgroup**: 32 threads each L2-normalize a 128-wide head serially |
| moe_dnc_q4_q8 | 1.87 | 35 | 54 | |
| rmsnorm | 1.34 | 81 | 17 | **one workgroup** each |
| dn_delta_gn | 1.29 | 30 | 43 | |
| matvec_coop | 1.18 | 70 | 17 | |
| matvec_q4_coop_acc | 1.00 | 26 | 39 | |
| attn_flash | 0.83 | 10 | 83 | |
| moe_route | 0.65 | 40 | 16 | **one workgroup** (dispatch 1×1×1) |

In the speculative step the three drafts cost 10.9 ms of 42.9: each runs the full 248k-row LM head (~2.2 ms; `matvec_q8_coop` is 12 dispatches × 602 µs there) and a **single-workgroup argmax over 248,320 logits at 490 µs** each. Going from 1 to 4 columns multiplies expert gate/up by 2.5× (8.09 vs 3.27 ms: the 4 tokens route to different experts) and DeltaNet projections by only 1.25×.

### One speculative step, Chrome (submit mode)

| submit | CPU submit at (ms) | CPU encode (ms) | GPU start at (ms) | GPU (ms) |
|---|---|---|---|---|
| `_verifyFused` (3 drafts + 4-column trunk + head) | 0 | 0.40 | 0 | 40.43 |
| `_restoreDN` (after a rejection) | 50.3 | 0.20 | 47.2 | 2.17 |
| `_mtpRefill` (draft cache rows + next first draft) | 50.3 | 0 | 49.4 | 0.61 |

The GPU sits idle from 40.4 to 47.2 ms: the 4 MB logits map, the CPU argmax over 4 × 248k logits, and the next step's encode. Over 12 steps the GPU idles 6.4 ms per step between submits (Chrome) and 16.8 ms (Deno).

## 2. Several devices (room emulator)

`tests/e2e/room_prof.mjs`, one Chromium per device, loopback WebRTC (`?wire=stripe4`), greedy (`exact`), two-sum prompt, 48 tokens, measured answer after a warm-up. Split by pledge: 2 devices = host 20 layers + embed/head, worker 20; 3 devices = 14 + 13 + 13. The room's `draftvocab` default (65,536) is on.

| | plain tok/s | speculative tok/s (acceptance) |
|---|---|---|
| 2 devices | 30.8 | 54.7 (77 %) |
| 3 devices | 26.2 | 49.0 (77 %) |

### Plain decode: one token's lap (medians of 48 laps)

| segment | 2 devices | 3 devices |
|---|---|---|
| host: its layers (embed → hidden read back) | 10.9 | 9.1 |
| of which GPU span | 8.45 | 5.78 |
| of which CPU encode | 0.3 | 0.3 |
| of which readback wait (submit → `mapAsync` resolved, minus GPU) | 2.25 | 2.34 |
| host: NaN scan + f16 pack + send | < 0.1 | < 0.1 |
| **per worker hop** (hop 1 / hop 2) | | |
| wire (sender's last send → receiver's last slice) | 0.88 | 0.79 / 1.01 |
| deliver + queue + unpack f16 | < 0.1 | < 0.1 / < 0.1 |
| CPU encode | 0.3 | 0.2 / 0.2 |
| GPU span | 8.2 | 5.49 / 5.41 |
| readback wait | 2.36 | 2.32 / 2.26 |
| NaN scan + pack + send | < 0.1 | 0.1 / 0.1 |
| hop total, arrival → next arrival (incl. wire) | 11.98 | 9.09 / 9.11 |
| **hop overhead (total − GPU)** | **3.8** | **3.6 / 3.7** |
| return wire | 1.22 | 0.90 |
| lap (host starts → hidden back) | 24.9 | 30.1 |
| head after the lap (upload, norm, LM head, 1 MB readback) | 4.6 | 4.7 |
| **token** | **31.2** | **36.3** |

GPU work per token is the same as solo (host 8.45 + worker 8.2 + head ~2.2 ≈ 18.9 ms for 2 devices; 18.8 solo). Everything else, 12.3 ms at 2 devices and 17.4 ms at 3, is overhead:

| overhead per plain token | 2 devices | 3 devices |
|---|---|---|
| readback waits (host + each worker), ~2.3 ms each | 4.6 | 6.9 |
| wire, ~0.8 to 1.2 ms per link on loopback | 2.1 | 2.7 |
| head outside the lap beyond its GPU time (separate submit, 1 MB map) | 2.4 | 2.5 |
| CPU encode (~0.25 ms per device) | 0.6 | 0.8 |
| host between laps other than the head (sampling 0.17 ms, UI, emit) | 1.5 | 2.0 |
| JS pack / unpack / NaN scans / slicing | < 0.3 | < 0.4 |
| not attributed (segment medians do not add up exactly; lap-to-lap variance) | 1.1 | 2.6 |

**Per hop, the non-GPU cost is ~3.6 to 3.8 ms, not ~15 ms**: about 2.3 ms of readback wait (the device reads the hidden back to the CPU: a copy submit then `mapAsync`), ~1 ms of wire, 0.3 ms of encode, and well under 0.2 ms of JS. The ~15 ms per hop in `docs/bench-log.md` came from the 27B with 16 tabs in one browser (one GPU process for all tabs, phone tabs with a 4× CPU throttle); with one browser per device and the MoE's 4 KB hidden state the hop is cheap, and the readback wait is the biggest piece of it.

The JS work per hop is microseconds (`tests/prof/hop_cpu_bench.mjs`, Node 22, per call):

| frame | bytes | f16 pack | f16 unpack | NaN scan | readback copy | sendFrame (slicing) |
|---|---|---|---|---|---|---|
| MoE, 1 column (dim 2048) | 4 KB | 12 µs | 5 µs | 3 µs | 1 µs | 4 µs |
| MoE, 4 columns | 16 KB | 33 µs | 15 µs | 5 µs | 10 µs | 12 µs |
| MoE, 8 columns | 32 KB | 40 µs | 24 µs | 9 µs | 3 µs | 11 µs |
| 27B, 1 column (dim 5120) | 10 KB | 15 µs | 7 µs | 3 µs | 4 µs | 8 µs |
| 27B, 8 columns | 80 KB | 104 µs | 30 µs | 22 µs | 7 µs | 25 µs |

Greedy sampling over 248,320 logits is 0.17 ms, top-40 0.28 ms, copying 1 MB 0.2 ms (Node).

### Speculative decode across hops

Lap by verify width (2 devices; the room's K picker runs K=3 for three steps, probes K=5 and K=7 once each, then keeps the best):

| verify columns | steps | step ms | lap ms | host outside the lap ms | tok/step | host GPU per step ms | worker GPU per lap ms | hop wire ms | worker readback ms | bytes per hop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 41.5 | 32.8 | 8.7 | 2.0 | 16.7 | 12.0 | 1.52 | 3.28 | 8 KB |
| 4 (K=3) | 10 | 50.5 | 37.2 | 12.6 | 3.7 | 22.2 | 14.8 | 1.22 | 2.47 | 16 KB |
| 6 (K=5) | 2 | 69.8 | 52.6 | 18.2 | 2.0 | 35.1 | 22.7 | 1.62 | 2.40 | 24 KB |
| 8 (K=7) | 2 | 79.9 | 57.7 | 24.1 | 2.5 | 42.7 | 23.4 | 2.62 | 3.37 | 32 KB |

3 devices, 4 columns: step 55.3 ms, lap 41.1, outside 13.4, 3.7 tok/step, hop overhead 2.7 and 2.5 ms plus ~1 ms of wire per hop.

The host's side of one K=5 step (2 devices), ms from the step's start:

| at | what | GPU ms | wait ms |
|---|---|---|---|
| 0.2 | draft chain (5 drafts, small head) submit + 80-byte map | 6.48 | 7.9 |
| 8.2 | lap starts: host's 20 layers on 6 columns, submit + 48 KB map | 21.09 | 23.5 |
| 32.4 | frame sent; worker runs its 20 layers | | |
| 59.8 | hidden back, unpacked | | |
| 59.9 | verify head on 6 columns (`headBatch`: upload, norm, LM head) + **5.8 MB logits map** | 5.72 | 7.2 |
| 67.1 → 69.6 | CPU: argmax of 6 × 248k logits, rollback, emit | | |
| 69.6, 69.8 | `_restoreDN`, `_mtpRefill` | 1.14, 0.70 | |

What the table shows:
- A speculative step in a room takes **three syncs on the host and one per worker**: the draft chain, the host's layers, the verify head (solo it is one). The draft chain (6 to 10 ms GPU plus its sync) runs serially before the lap starts, and the verify head plus a 4 to 8 MB logits map runs serially after it.
- Verify columns are almost free on the wire (8 → 32 KB: +1 ms) and in JS; they cost GPU (each device's span grows ~1.6× from 4 to 8 columns) and acceptance. At 77 % acceptance the K=5 and K=7 probes gave 2.0 and 2.5 tokens per step for 38 % and 58 % more time than K=3: 4 of the 15 steps of a 48-token answer are probes, about 20 % of the answer's time at a loss. On a fast link (wire ~1 ms of a ~40 ms lap) deeper drafts never pay; they only pay when the network round trip dominates the lap, which the picker could read from the lap split it already has (lap − host − worker compute).
- Readback wait per hop is flat in the column count (2.4 to 3.4 ms): it is a fixed sync cost, so wider frames amortize it.
- Each device's GPU is idle while the others work (2 devices: host GPU busy ~22 of 50 ms per step). That is the price of one sequence through a pipeline; only more work per lap (wider verify, or several sequences) fills it.

## 3. Bottlenecks, ranked

Estimated savings are from the measurements above (Chrome, GB10); none of these were tried yet.

| # | bottleneck | where | measured cost | idea | expected gain | exact? |
|---|---|---|---|---|---|---|
| 1 | Logits read back to the CPU for sampling: 1 MB per plain token, 4 to 8 MB per verify | solo and room host | solo plain: map resolves 2.2 ms after the GPU ends (22.1 ms token); solo K=3: ~4.3 ms per step; room: head outside the lap 4.6 ms (2.4 beyond GPU), verify head map 7.2 to 7.6 ms (1.5 to 2 beyond GPU) + ~2.5 ms CPU argmax | GPU argmax for greedy (16 bytes per column back) and a GPU top-k (k ≤ 40: 320 bytes per column) for the temperature presets; sampling from the top-k on the CPU. The draft chain already does GPU argmax. | solo plain ~−2 ms (+10 %), solo spec ~−3.5 ms per step (+7 %), room ~−2 ms per plain token and ~−3.5 ms per spec step | greedy: yes if ties break the same (lowest index, as `greedy()`); top-k: yes if the k candidates and their order match the CPU scan |
| 2 | Single-workgroup kernels | every token | `dn_pre` 1.93 ms (30 × 64 µs: 32 threads each L2-normalize 128 values serially with strided loads), `rmsnorm` 1.34 ms (81 × 17 µs), `moe_route` 0.65 ms (40 × 16 µs), `argmax` 490 µs per draft over the full vocab | `dn_pre`: one workgroup (or subgroup) per head with a tree reduction, or fold the L2 into `dn_delta_gn`, which reads q and k anyway; `rmsnorm`: fold the sum of squares into the consuming GEMV's prologue (each workgroup recomputes 2048 squares) or a two-level reduction; argmax: two-stage multi-workgroup | up to ~3 ms of 18.8 ms GPU per plain token (~15 %); argmax −1.4 ms per full-head draft chain | `dn_pre`/`rmsnorm`: no (summation order changes), so off by default behind a flag and checked against llama.cpp; `moe_route` and argmax: yes |
| 3 | Per-hop readback wait | every device, every lap | ~2.3 ms per device per lap (host and each worker), flat in the column count; Chrome's floor for an empty submit + 4-byte map is 0.93 ms | put the hidden's `copyBufferToBuffer` into the compute command buffer (today `_readback` makes a second encoder and submit); measure whether Chrome's map callback latency, not the GPU, is the rest | 2 devices: up to ~1.4 ms per device per lap if it reaches the 0.93 ms floor (~9 % of a plain token) | yes |
| 4 | Draft chain before the lap, verify head after it, in separate syncs | room speculative host | draft chain 6.5 to 10 ms GPU + sync before every lap; head + map after it | the chain-mode twin of `specFuse`: draft chain + the host's own trunk layers in one command buffer (the draft embeddings gathered into the verify columns on the GPU, as `_verifyFused` does), one map returning drafts and the frame | −1 sync and a CPU round trip per step (~1.5 to 2.5 ms of ~50) | yes (same kernels, same order) |
| 5 | K probing on fast links | room speculative | K=5/7 probes: 2.0 and 2.5 tok/step for +38 %/+58 % step time; 4 of 15 steps of a 48-token answer | only probe deeper K when the measured non-compute part of the lap (lap − host − worker compute, already reported by `ai-tele`) is a large share of the lap | ~+15 % on short answers at 1 to 3 devices on a LAN; a few % on 400-token answers | yes (drafts only change speed) |
| 6 | Full-vocab draft head in the bench | solo bench only | 3 × 2.2 ms head + 3 × 0.49 ms argmax = 8 ms of a 48.6 ms step | already solved in the room (`draftvocab=65536`); the bench default hides it | the bench-log's 74 → 83 tok/s | yes |
| 7 | Expert GEMVs | every token | gate/up 82 µs and down 54 µs per layer, 5.5 ms per plain token; 13.1 ms per 4-column verify | layout tuning (`opt/moe-kernel-tuning` has the knobs); at 4 columns, group the columns by expert so each expert's rows are read once per step | up to ~2 ms plain, more at 4 columns | yes |
| 8 | Head after the lap, on the host | room | head 4.6 ms outside the lap per plain token | have the last worker run the head and send back only top-k (or the argmax), if it holds the head weights; or keep the head on the host but read back only top-k (#1) | with #1 most of it | yes |
| 9 | Deno sync cost | Deno only | 11.7 ms per sync | not a product path; profile in Chrome | — | — |

What is **not** a bottleneck on the GB10: f16 pack/unpack, the NaN scans, slicing and reassembly (all < 0.2 ms per hop together), CPU encode in Chrome (0.3 ms per submit), queueing on workers (< 0.1 ms: frames never wait behind each other in decode), and sampling on the CPU (< 0.3 ms).

## 4. Candidate experiment branches

Not created yet; each would branch from `exp/base`, keep the change off by default unless `tests/run.sh quick`, `tests/run.sh q38` and `tests/test_moe.js` pass bit-identically, and check split == solo with `tests/e2e/room_synth.mjs` for room changes.

| branch | idea (from the table) | overlap with other jobs |
|---|---|---|
| `exp/gpu-sample` | #1 and #8: GPU argmax / top-k for the plain head, the verify head and the room head | none |
| `exp/one-sync-hop` | #3: readback copy in the compute submit, then measure what is left of the 2.3 ms | none |
| `exp/chain-fuse` | #4: draft chain + host trunk in one submit in a room | none |
| `exp/k-probe` | #5: probe deep K only when the network share of the lap is large | none |
| `exp/wide-reduce` | #2: multi-workgroup `dn_pre` / `rmsnorm` / `moe_route` / `argmax` | `dn_pre` and `rmsnorm` are shared with the 27B: check with the kernel agent (`kopt/*`) before starting |
| (none) | #7 expert GEMVs at decode | `opt/moe-kernel-tuning` exists; prefill grouping is the prefill workflow's (`prefill/moe-group`) |

## Raw numbers

- Solo: `tests/prof/prof_moe_decode.js` (Deno) and `tests/prof/prof_chrome.mjs` (Chrome), MoE, TOKENS=24, STEPS=12, K=3.
- Room: `tests/e2e/room_prof.mjs --model qwen3.6-35b-moe --devices 2|3 --maxnew 48`, then `tests/e2e/room_prof_report.mjs`. Load time 100 to 120 s per room with the weights in the page cache.
- For comparison, the latency sweep on `bench/latency` (japan prompt, 128 tokens, no emulated lag) measured the MoE room at 1 device 32 to 33 tok/s plain, 44 tok/s speculative; 2 devices 27.6 to 28.1 plain, 30.7 to 34.6 speculative (40 to 42 % acceptance on that prompt).
