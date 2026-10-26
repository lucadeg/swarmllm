# Code mode with the real 35B MoE (manual, needs the GPU)

The automated test (`tests/e2e/harness_tetris.mjs`) drives Code mode with a scripted model. This
checklist is the same flow on the real model, to measure what the design (docs/design/harness-app.md
F, G.4) can only estimate. Run it when nothing else is using the GPU.

For numbers rather than a checklist, run the eval suite (`tests/eval/README.md`): the same
twelve tasks on one local engine (`node tests/eval/run.mjs --model engine`) or in this room
(`p2p.html?eval=all`, then `/eval` in the Code prompt). Its `.jsonl` gives success, steps, tokens
and time per task, and a trajectory per task to read the failures.

## Setup

1. Serve the repo: `npm run serve` (http://localhost:8080). Use `localhost`, not a LAN IP: WebGPU
   and OPFS need a secure origin.
2. Host: Chrome on the GPU machine, `http://localhost:8080/p2p.html`. Name it, pledge the memory
   the model needs, **Create room**. Do not add `?mock=code`.
3. Optional second device (a laptop, or **+ virtual device** in the sidebar): join with the room
   code. Two devices exercise the split and the peer preview.
4. Model: **Qwen 3.6 35B MoE · Q4** (32k context by default; `?ctx=N` up to 64k). **start**, wait for
   "cluster online".
5. Click **Code**. A project is created from the first request, or use **New** first to name it.

Open DevTools on the host and keep the Console visible: a failure in the harness shows there.

## Run 1: build from nothing

Prompt: `Build a Tetris game in plain JS with a canvas. Arrow keys move, up rotates, space drops.`

Record, from the timeline and the `.cm-stats` line at the end:

| What | Where to read it | Value |
|---|---|---|
| system prompt + tools, tokens | the `context N / M` meter after step 1 is prompt + first answer; the room status line gives the answer's token count | |
| steps, tool calls | `.cm-stats` | |
| tokens generated, tok/s | `.cm-stats` | |
| re-prefills (compaction) | "older steps shortened…" notes | |
| did serve report errors? | the `serve` card's result | |
| did the agent fix them without help? | the steps after serve | |
| final context used | `context N / M` under the timeline | |
| wall time | `.cm-stats` | |

Check:
- [ ] every `write_file` card shows a diff and **done**; nothing stuck in **running**
- [ ] a long file is written in parts (`append: true`) rather than cut off at the answer cap
- [ ] the preview tab `:5173` opens by itself on the first `serve`, and the game is playable
      (click the frame first so it has keyboard focus)
- [ ] errors in the preview console match what the agent read (`serve` / `preview_logs` results)

## Run 2: follow-up (prefix reuse)

Prompt: `Add a score and a next-piece box.`

- [ ] the first step reuses the previous context (the room status line says how many tokens were
      reused; expect about the whole previous context)
- [ ] edits use `edit_file`, with small diffs, and the preview reloads after each one (a new
      `rev N · reloaded` line in the console strip)

## Run 3: peers and control

- [ ] on the second device: the **Code** tab appears with a dot, the timeline matches the host's,
      **Run preview :5173** runs the same game at the same `rev`, and a host-side edit reloads it
- [ ] untick **auto-approve edits** on the host: the next edit waits with Approve / Reject… /
      Allow edits for this task, and the peer shows "waiting for the host's approval"
- [ ] **Reject…** with a reason: the agent reads "declined by the user: <reason>" and adapts
- [ ] **Stop** (or Esc) during a long `write_file`: the run ends after the lap in flight with a
      "stopped" note; the next request works; switching to **Chat** and asking a question still
      answers (the lock was released)
- [ ] **new task** clears the agent's memory but keeps the files and the preview

## Run 4: the 27B at 16k

Repeat run 1 on **Qwen 3.8 27B · Q4** (16k). Expect compaction notes; the run should still finish,
or stop with "context full: start a new task (the files are kept)" and recover after **new task**.

## Report

Paste the tables and any console errors into the PR or `docs/bench-log.md`, with the date, the GPU,
the number of devices and the commit.
