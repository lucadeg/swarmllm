# 23 · Contributor on-ramp: labels, seeded issues, no-GPU track, hardware-verifier role

**Phase:** now · **Status:** in part · labels (`area: *`, `impact: *`, `no-gpu`, `good first issue`) exist, and #56, #57 and #59 are good first issues

## Why

Merges: *Contributor on-ramp*; *Contributor ladder with criteria, CODEOWNERS, verifier role*.
Master plan NOW #6 and the weeks-2–6 metric ("ten merged community PRs") have no roadmap file, no labels and no issues; CONTRIBUTING says GPU tests run on the maintainer's hardware before merge, so every PR is gated on the one person who is also writing the paper. A launch converts contributors in the first days or not at all, and most arrivals lack a WebGPU GPU. Seeds exist in the tree: `room/models.js`, the unit tests in `tests/unit/`, and the no-GPU browser tests in `tests/e2e/` (SwiftShader). A "hardware verifier" rung, someone who owns a device class and runs the item-21 report, is the project-specific role no generic template supplies; the ladder text is a one-PR edit to GOVERNANCE.md and belongs inside this item, not its own.

## Design

- Labels: `area: engine`, `area: kernels`, `area: room`, `area: protocol`, `area: code` (Code mode), `area: docs` and the others, plus `impact: *`, `no-gpu` and `good first issue`. Every roadmap issue carries an area label.
- A no-GPU track: issues labelled `no-gpu` can be built and tested with `npm test` and the SwiftShader browser tests (`tests/e2e/room_synth.mjs`, `harness_tetris.mjs`, `preview_browser.mjs`), so contributors without a WebGPU GPU can still ship.
- A hardware-verifier role: someone who owns a device class (a Mac, an iPhone, an AMD or Intel GPU) and runs the benchmark report from roadmap 21 on it. PR #51 was exactly this.
- CODEOWNERS names an owner per path (`/engine/`, `/harness/`, `/p2p.html`); collaborators are added per path as they arrive.

The full merged proposals, including what was folded into other items and what was rejected, are in [docs/archive/roadmap-review.md](../docs/archive/roadmap-review.md) under item 23.

## Done when

- Every open roadmap issue has an area label, and at least five issues are marked `good first issue` or `no-gpu`.
- CONTRIBUTING.md lists the no-GPU tests, and GOVERNANCE.md describes the hardware-verifier role.
