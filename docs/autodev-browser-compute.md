# AutoDev browser-compute bridge

SwarmLLM can act as an opportunistic inference backend for Hermes/FounderOS AutoDev. The browser room remains the compute plane; AutoDev remains the control plane.

## Contract

A host tab exposes `window.SwarmLLMAutoDev`:

- `status()` reports model readiness, peer count, WebGPU capability and the non-persistent trust model.
- `submit(job)` accepts only bounded `observe`, `recommend` and `prepare` work.
- Jobs must declare `sensitivity` as `public` or `internal_nonsecret`.
- Credential-like fields and common credential value shapes are rejected before inference.
- Results are returned to the caller and emitted as `swarmllm:autodev-job-result` DOM events.

Example:

```js
const result = await window.SwarmLLMAutoDev.submit({
  jobId: "JOB-AUTODEV-001",
  label: "architecture-review",
  sensitivity: "internal_nonsecret",
  actionClass: "recommend",
  prompt: "Review the supplied non-secret architecture excerpt and list test gaps."
});
```

## Operational model

SwarmLLM does not become a scheduler and does not become an authority. Browser peers contribute WebGPU compute only while their tabs/devices remain available. Static hosting can therefore have zero mandatory inference-server cost, but the system is not a guaranteed 24/7 compute service.

AutoDev must keep durable task state, retries, authorization, evidence and release gates outside the browser room. A browser-swarm answer is untrusted model output until independently verified.

## Security boundary

Use trusted room participants only. Do not submit secrets, API tokens, private keys, session cookies, credentialed actions or sensitive private-repository material. SwarmLLM's existing threat model applies: intermediate activations are not encryption and remote compute is not currently attested.

The bridge intentionally exposes no peer transport primitives, no arbitrary JavaScript execution hook and no credential channel.


## Zero-server static deployment

The repository includes `.github/workflows/pages.yml` for GitHub Pages. After GitHub Pages is enabled for the repository, a push to `main` or a manual workflow dispatch publishes the static runtime. No inference server is deployed: browsers download the static assets and model shards, then contribute WebGPU compute directly.

On GitHub Pages, use the static `/p2p.html` path because Vercel-specific rewrites are not available. Provider quotas and GitHub's service terms still apply.


## Fleet boot URLs

A browser can opt in as a worker directly from the URL:

```text
/p2p.html?code=ABCD&autojoin=1&name=tablet&gb=6
```

A host can opt in to automatic room creation:

```text
/p2p.html?autocreate=1&name=autodev-host&gb=8
```

The host-side AutoDev API also exposes `workerLink({name, gb})` and `startModel(model)`. Auto-start is intentionally URL-opt-in; it does not bypass browser wake-lock/background-throttling rules. Mobile workers should remain foreground/awake for predictable contribution.

For Hermes/CDP orchestration, use `dispatch(job)` followed by `job(jobId)` polling. Do not await a full model generation inside one CDP `Runtime.evaluate`: the browser supervisor keeps individual evaluation calls short and bounded.
