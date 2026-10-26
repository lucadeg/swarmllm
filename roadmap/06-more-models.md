# 06 · More models

**Phase:** done · **Status:** shipped: Qwen 3.8 27B, Qwen 3.6 35B MoE (in rooms), Qwen3.5-122B MoE (in the engine) · follow-ups tracked in #6

## What shipped
- **Qwen 3.8 27B** (Q4_0): the hybrid Gated DeltaNet + attention model the project started with, with MTP speculation.
- **Qwen 3.6 35B MoE** (Qwen3.6-35B-A3B, Q4_0): 256 experts, 8 active per token. Router, top-k experts and the shared expert run on the GPU (`engine/wgsl/moe.js`), on the one-token and batched paths, and speculative decoding stays exact. It is in the room's picker and decodes several times faster than the 27B because it reads ~3B parameters per token.
- **Qwen3.5-122B-A10B MoE**: runs in the engine and its tests. It is not in the room's picker yet.
- **Qwen3 1.7B** (Q8_0): the small model in the picker, for rooms of phones and light laptops.

## Still open
- Qwen 3.5 2B and 9B share the `qwen35` layout, so they are entries in `room/models.js` plus a golden test.
- Splitting a MoE model by experts across devices is roadmap 10; checking that the 35B MoE split by layers is exact across devices is roadmap 31.

## Done when (for each new model)
- It has a golden test and a bench-log row on at least one device.
