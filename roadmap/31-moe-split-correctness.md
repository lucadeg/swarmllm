# 31 · 35B MoE split across devices: check correctness

**Phase:** multi-device · **Status:** research · placeholder until someone reproduces a problem

## Why
Qwen 3.6 35B MoE is split by layers across devices in a room, like the dense models. The engine's own tests check each MoE layer against a float64 reference and check that speculative decoding equals plain decoding. We have no written record of a room split giving wrong output, but we have not checked a real multi-device MoE room against a single device either. One related open item: on the combined optimization branch, `tests/e2e/moe_synth.mjs`'s "batched prefill == one token" check failed with 202 differing logits (docs/bench-log.md, 2026-09-26). A room runs batched prefill, so this may matter.

## Design
- First reproduce: build a small synthetic MoE model with `tests/e2e/synth.mjs` and run `tests/e2e/room_synth.mjs --model <file>` with it, solo and split over 2 and 3 tabs, greedy, and compare the answers (`--compare`).
- Then on real hardware: the 35B MoE solo on one device against a two-device room, greedy, on the standard prompts.
- If they differ, find the first layer and token where they diverge, and file a bug with that.

## Done when
- A greedy 35B MoE room over two or more devices gives the same answer as one device, or there is a bug report that says where and why it does not.
- The check runs in a no-GPU test so it stays true.
