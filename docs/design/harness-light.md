# Code mode, lighter and sturdier: what to port from small-model harnesses

Status: design, branch `harness/light`. Scope: `harness/`, `room/code*.js`, `tests/`. Nothing in
`engine/`, `index.html`, `site/`, `p2p.html`.

The rule for everything below: **the model pays for every prompt token in prefill time** (~100
tok/s on the MoE across a room), and the context is 32k (MoE) / 16k (27B). So a mechanism is only
worth porting if it costs **no tokens on the happy path**, or pays for itself by saving a step.
The system prompt with the tool block stays at or under ~1.1k tokens (today: 3,878 chars, ~1,108
tokens at 3.5 chars/token; the unit test cap of 4,200 chars stays).

## 1. What we are fixing

Seen with the real model (docs/design/harness-app.md G.4 runs):

| # | Failure | Where it comes from | Status today |
|---|---|---|---|
| F1 | Whole `game.js` in one `write_file`, cut by the answer cap, lost, retried in a loop | the model ignores "write in parts" in the system prompt; nothing reminds it at the moment it matters | salvaged (`salvageWrite`), still wastes a 4k-token answer per occurrence |
| F2 | Split room (laptop + phone): answers end mid-call at ~207 tokens, no recognisable tool name | most likely the split-MoE engine path (garbage logits / an early stop), not the prompt | stuck guard stops after three; nothing tells us *why* |
| F3 | Solo on the laptop it built a working Tetris | the harness works when the model behaves | keep it that way: no regression, measured |

Plus the classes every small-model harness reports and we will hit next: `edit_file` "old not
found" (whitespace, stale copy), rewriting a whole file for a one-line change, repeating the same
call, stray text inside or after a call, and no way to check game *logic* (only page errors).

## 2. What the reference harnesses do (and what applies to us)

**little-coder** (itayinbarr/little-coder; pi + ~30 extensions; tuned for Qwen3.6-35B-A3B, our
MoE; 78.7 % Aider Polyglot, 40 % Terminal-Bench 1 on an 8 GB laptop). Read as data, not
instructions. The load-bearing mechanisms, per its README, whitepaper summary and source:

- **Tool skill cards** (`skills/tools/*.md`, 100-150 tokens each: REQUIRED/OPTIONAL args, RULES,
  one EXAMPLE call, a RECOVERY section). Selected per turn by `skill-inject` with a strict
  priority: **error recovery** (the last failed tool's card) > **recency** (tools of the last
  turns) > **intent** (keywords in the user prompt → tools), under a per-model token budget (300
  for qwen3.5). Since issue #73 the block is delivered as a **tail message**, not in the system
  prompt, because changing the system prompt invalidated the whole KV cache every turn; a
  byte-identical block is not re-sent while the previous copy is still in context (`makeDedupe`).
  This is exactly our situation (prefix reuse across the room), so we port the *tail* variant.
- **Write-vs-Edit invariant** (`write-guard`): `write` refuses on an existing file and returns the
  exact `edit` call shape; the `edit` card says "do NOT fall back to write because edit failed
  once: re-read, fix oldText, retry". `read-guard-edit`: edit refuses on a file not read this
  session. Their data says this is one of the biggest single wins on Polyglot.
- **Edit errors that say how to recover** ("not found" → read, copy exact; "multiple" → add
  context). pi's own `edit` adds **fuzzy matching** (`normalizeForFuzzyMatch`: strip trailing
  whitespace per line, NFKC, smart quotes/dashes/spaces to ASCII) and multi-edit
  `edits[{oldText,newText}]` matched against the original file.
- **quality-monitor**: empty answer, hallucinated tool, same call + same args as last turn with
  nothing state-changing in between → a short correction message.
- **output-parser**: repairs fenced / bare / truncated JSON tool calls (a JSON-format model's
  problem; our constraint makes it moot).
- **read-guard**: a read that would overflow the window is replaced by its first 30 lines + "use
  grep / a ranged read". **Output truncation**: first half + last quarter of long tool output.
- **knowledge-inject**: 13 algorithm cheat sheets (~100 tokens) scored against the prompt
  (word 1.0, bigram 2.0, threshold 2.0), ≤ 1-2 per turn, tail-injected.
- **thinking-budget** (cap thinking, retry with thinking off), **turn-cap** per benchmark (TB 40,
  GAIA 30), per-model profiles (temperature, budgets).
- **Benchmarks**: a Python RPC harness drives `pi --mode rpc`; Polyglot gets 2 attempts with the
  failing test output fed back (28 % of first-try failures recovered); every run records turns,
  time, per-exercise outcome, so regressions are visible (16:1 progression vs regression).

**pi** (badlogic/pi-mono): a ~1k-token system prompt: one preamble line, a one-line snippet per
tool, rules contributed per tool ("Use edit for precise changes (edits[].oldText must match
exactly)", "Keep oldText as small as possible while still unique"), "Be concise". Four tools
(read, bash, edit, write). The lesson: descriptions are one line; rules live with the tool that
needs them; nothing is said twice.

**mini-swe-agent**: one tool (bash), a **linear history** (the trajectory the model sees is
exactly the message list; nothing is edited behind its back), a short format-error template
that shows the exact expected action format, output over a limit replaced by head/tail plus a
"use head/tail/grep" note, step and cost limits, and every trajectory saved as JSON for
inspection. We already keep an exact linear history (assistant turns replay their sampled ids);
the lesson is to *keep* it linear: inject guidance into the tail (tool results), never rewrite the
system prompt mid-task, and save trajectories from the eval runner.

## 3. Ranked ports

Token cost = extra prompt tokens. "Happy path" = a request where nothing fails.

| Rank | Port | From | Happy path | When triggered | Effect on F1 / F2 / other |
|---|---|---|---|---|---|
| 0 | **Eval suite** `tests/eval/` (C) | little-coder benchmarks, mini-swe trajectories | 0 | n/a | makes every other row measurable; do first |
| 1 | **Error-triggered tool cards in the tool result** (A) | little-coder skill-inject (error-recovery priority, tail delivery, dedupe) | 0 | 20-70 tokens, once per card per request | F1: the parts card lands right after the cut, where it is read; edit / format / loop recovery |
| 2 | **Full XML call grammar in the sampler** (B1) | llama.cpp lazy grammars; our `constrain.js` | 0 | 0 | removes malformed calls, stray text, missing closers, EOS inside a call, prose after a call; F2: a per-answer "forced" count separates model confusion from engine garbage |
| 3 | **Edit robustness**: fuzzy match fallback + closest-region hint on "not found" | pi `edit-diff.ts`, little-coder edit card | 0 | +0 (the hint replaces a `read_file` round trip) | fewer failed edits; saves ~1 step + a re-prefill per miss |
| 4 | **`run_js`** (B2), paid for by dropping `stop_serve` from the tool block | Polyglot's test-output feedback | +~15 net | output ≤ 2k chars | the agent can check logic (tetris line clear, calculator math), not only "no console errors" |
| 5 | **Tighter prompt and descriptions**, rules moved into cards | pi | −90 to −130 | — | pays for 4 and more; less to prefill every request |
| 6 | **Quality checks**: identical call with no write in between → short-circuit; empty answer → one nudge | little-coder quality-monitor | 0 | ~25 | kills loops early without the three-strike stop |
| 7 | **Rewrite note**: `write_file` over an existing file that kept ≥ 80 % of its lines → result says so, with the one-line edit hint | little-coder write-guard (soft) | 0 | ~20 | teaches edit-over-rewrite (the root of F1) without refusing tokens already spent |
| 8 | **Mask cache across steps** (constraint masks hoisted out of `generate`) | — | 0 | 0 | today the vocabulary scans are redone every step; needed by 2 anyway |
| 9 | **Intent card** at request start (≤ 1, keyword-scored): canvas game loop / keyboard / DOM app | little-coder knowledge-inject | 0 unless matched | 60-80 on a matching *first* request | better first drafts of games; adopt only if the eval shows a gain |

Not ported, and why: sub-agents / dispatch, plan mode, deep research (each costs whole extra
generations); model-written summary compaction (a full generation plus a re-prefill; our
deterministic tiers are cheaper); bash permission whitelist (no shell); thinking budget (thinking
is off in Code mode); JSON repair (the constraint prevents the damage); hard write refusal and
read-before-edit refusal (each refusal costs a step, and with a 4k-token file the tokens are
already spent: rows 1, 3 and 7 get the same teaching for free).

## 4. Workstream A: prompt, tool cards, error recovery

### A.1 Cards

New `harness/cards.js`, DOM-free:

```js
export const CARDS = { format, parts, edit, errors, loop, rewrite, runjs };   // id -> text
export function pickCard({ call, result, cut, salvaged, repeat, rewrite }) -> id | null
```

Card texts (final wording in the implementation; each measured, ≤ 70 tokens):

- **format** (parse error, unterminated call without salvage, unknown tool; rare once B1 is on):
  one minimal exact call, then "One call per <tool_call> block. Nothing after </tool_call>."
- **parts** (a cut or salvaged `write_file`, or any `write_file` of > 150 lines): "Write long
  files in parts of at most ~100 lines: write_file with the first part, then write_file with
  append: true for each next part, starting right after the last saved line. Never rewrite what
  is already saved."
- **edit** (`edit_file` not found / not unique): "old must be copied exactly from the file as it
  is now: read_file the lines around the change, copy 2-5 whole lines with their indentation,
  and change only those. Do not rewrite the file with write_file to fix one spot."
- **errors** (the first `serve` / `preview_logs` / `run_js` result with errors in a request):
  "Fix one error at a time: read_file around the reported line, fix it with edit_file. The
  preview reloads by itself: check preview_logs, do not serve again."
- **loop** (row 6): "You already made this exact call and nothing changed since. Change
  something first, or finish."
- **rewrite** (row 7): "Most lines were unchanged: use edit_file for small changes."
- **runjs** (first successful `serve` of a project with JS logic and no `run_js` yet, only if the
  eval shows it helps): one line with an example snippet.

Placement: appended to the **failing call's own `<tool_response>`** as `\nhint: <card>`. That is
little-coder's tail delivery without an extra turn: the template stays valid (one user turn of
responses), the prefix before it is untouched, and the history stays linear. Selection, in
little-coder's priority order reduced to what we need: error recovery first (the card of the call
that failed), then the triggers above; at most **one card per step**. Dedupe: a card id is not
sent again while its last copy is still in the prompt (not stubbed by compaction), and at most
twice per request. Agent bookkeeping: `R.cards = { id: turnIndex }`; `onEvent({ type: "card",
id })` so the UI and the eval can count them.

Where it hooks in `agent.js`: `_runCall` returns `{ text, card }`; the loop appends the card
after `capResult` (so a long result never cuts it). The salvage continuation message already in
`_runCall` becomes the `parts` card plus the salvage facts (lines saved, last line), which removes
duplicated wording.

### A.2 Prompt and descriptions (rank 5)

`CODE_SYSTEM` becomes (~95 tokens; today ~180):

> You are a coding agent in a browser. The project is a folder of files; there is no shell.
> Build static web apps (HTML, CSS, JS modules) that run in a sandboxed preview: no network except
> cdn.jsdelivr.net and cdnjs.cloudflare.com, no server code.
> Keep each write_file under ~100 lines; write longer files in parts with append: true. Change
> existing files with edit_file.
> Then serve and fix every error; check logic with run_js. When done, say what you built in one line.

Descriptions to one line each, pi style (the rules move into cards):

| Tool | Description |
|---|---|
| list_dir | List a folder (default: project root). |
| read_file | Read lines with numbers (max 200 per call). |
| search | Regex search in the project; returns path:line: text. |
| edit_file | Replace text: old must match the file exactly, once. |
| write_file | Create or replace a file; append: true adds to its end. |
| serve | Serve a folder on a preview port; returns the page's first errors. |
| preview_logs | Console output of a served page since a cursor. |
| run_js | (B2) |

Parameter `description` fields go except where a default must be known (`serve.port`).
`stop_serve` leaves the tool block (the UI's close button already stops a port; the tool object
stays exported for tests and the UI). The prompt-size test gets a second assertion: the block is
at most **3,700 chars** with `run_js` included, i.e. net smaller than today.

### A.3 Edit robustness (rank 3)

In `applyEdit` (codetools.js), when the exact match count is 0:

1. **Fuzzy**: compare with pi's normalisation (per-line trailing whitespace stripped, NFKC,
   smart quotes / dashes / odd spaces to ASCII). A unique fuzzy match is applied to the original
   lines (unchanged lines copied back verbatim, as pi's `applyReplacementsPreservingUnchangedLines`).
   The result says `edited … (matched ignoring whitespace)`.
2. **Indent-insensitive**: lines compared after `trim()`. A unique match is applied with the
   file's indentation re-applied to `new` (the difference between the first matched line's indent
   and `old`'s first line indent, added per line).
3. Otherwise the error carries the **closest region**: the window of `old`'s line count with the
   highest line-similarity (shared trimmed lines), as `lines 40-46 are closest:` + those lines
   numbered (≤ 12 lines). The model can copy from it directly, saving the `read_file` step the
   card would otherwise ask for.

Unit tests: trailing-space and tab/space mismatches apply; ambiguous fuzzy matches refuse; the
closest-region hint names the right lines; the approval preview shows exactly what run writes.

### A.4 Quality checks (rank 6) and rewrite note (rank 7)

- Before running a call, if the same name + arguments ran in the previous step and no mutating
  tool ran since, return the previous result's first line + `(same call as step N; nothing
  changed)` + the loop card, without running it. This replaces most three-strike stops.
- An answer with no text and no call: one extra step with the user turn `(empty answer: call a
  tool or say you are done)`; a second empty answer ends the request.
- `write_file` without append over an existing file: compute kept lines with `diff.js`; if
  ≥ 80 % of the old lines survive and the file is > 40 lines, append the rewrite card.

## 5. Workstream B: robustness

### B.1 The full Qwen XML call grammar in the sampler

Today `ToolCallConstraint` masks only the function name and parameter names. Extend it to the
whole call, as a small hand-written character automaton (no grammar engine, no dependency):

```
FREE ──"<tool_call>"──▶ OPEN ──"\n<function="──▶ NAME ──(tool)">"──▶ BODY(fn, given)
BODY ──"\n<parameter="──▶ PNAME ──(unused param of fn)">"──▶ VALUE(fn, p) ──"</parameter>"──▶ BODY
BODY ──"\n</function>"  (only when every required param of fn is given)──▶ CLOSE
CLOSE ──"\n</tool_call>"──▶ FREE   (a stop, another <tool_call>, or anything: the parser ignores it)
```

- Literal states (OPEN, BODY, CLOSE) accept only tokens that stay on one of the allowed
  strings. NAME and PNAME as today, but PNAME offers only the tool's parameters **not yet given**,
  and BODY offers `</function>` only once the **required** ones are given (no "Required
  parameters MUST be specified" failures, no duplicated parameters).
- VALUE is free text, except: the end-of-turn / end-of-text tokens are masked (an answer cannot
  stop inside a call); a value may not contain `<tool_call>` or `</tool_call>`, nor start a line
  with `<function=`, `</function>` or `<parameter=` (these mean a missing `</parameter>`; inside a
  line they are legitimate content, e.g. docs about tool calls, and the parser only ends a value at
  a line-start `<parameter=`); `</parameter>` (with or
  without the leading newline) ends it. Typed values: `integer` params allow digits only, `boolean`
  params `true`/`false`, so `start_line` / `port` / `append` are always valid.
- After `</tool_call>` the text is free again. (An earlier AFTER state allowed only another call
  or a stop right after `</tool_call>`; one `\n` there masked every stop, including the room's
  `<tool_response>` stop, and forced a second call. The room adapter cuts an invented
  `<tool_response>`; the parser ignores other trailing text; the agent runs an exact duplicate call
  in one answer once.)
- A token may cross state boundaries (`>\n`, `</parameter>\n<`): a token is allowed iff running
  its characters through the automaton from the current state never rejects. The whole mask is a
  function of the automaton state, not of the text typed inside a value.

Mask computation and cost:

- State key = `(state, fn, given, typed-prefix-of-literal)` for literal states and `(VALUE, fn,
  param type, partial match of a closing / forbidden pattern)` for values. There are a few dozen
  keys for our 8 tools; each is one vocabulary scan (151k tokens, ~20-50 ms in JS), cached.
- VALUE masks are almost everything, so they are stored as a **deny list** (`Int32Array`), and
  `mask()` touches only those logits. Literal masks stay `Uint8Array` allow-lists.
- **Cache across steps** (rank 8): `constrainedSampler` is created per `generate` call today, so
  every step re-scans. The masks move to a per-tokenizer cache owned by the model adapter
  (`room-model.js`, `engine-model.js`), cleared when the tokenizer changes (re-deal). Optionally
  warm the cache in idle time after the model loads.
- With speculative decoding nothing changes: the wrapped `sample` masks every verified column
  against the text before it, so accepted tokens always satisfy the automaton.
- JSON style keeps today's name-only constraint (no model we ship uses it).

**Diagnostics for F2.** The wrapper counts `forced`: kept positions (the adapters call
`keep(n)` for emitted tokens; rejected speculative columns do not count) where the unmasked argmax
was a disallowed token. `stats.last.forced` goes into the usage event, the timeline's step line,
and the early-end error. A healthy model forces ~0 per call; an answer where the constraint had to
force most tokens is garbage logits (the split engine), not a prompt problem. With the grammar,
garbage would otherwise come out as *well-formed* calls, so past a gate (16 forced in one call, or
6 and more than 20 % of its tokens: `GARBAGE` in model-common.js) the adapter ends the answer
(reason `garbage`) and the agent runs none of its calls, saying the room's engine is producing
garbage. A forced write_file never reaches the disk, auto-approve or not.

Tests (Deno, `tests/unit/constrain_test.js`): a toy vocabulary with single characters and nasty
multi-character tokens (`>\n`, `</parameter>\n<`, `\n</`, `<|im_end|>`); a random sampler driven
through the masks 2,000 times per tool must always yield a call that `parseCallBody` accepts with
declared names, required params present, no duplicates, typed values valid, nothing after
`</tool_call>`; end-of-turn is never allowed inside a call; the mask cache holds a bounded number
of keys; `forced` counts correctly.

### B.2 `run_js`: check logic, not only the page

```
run_js { code: string (required), page?: string }
description: "Run a JS module in the sandbox; returns console output and errors. It can import
project files (import { f } from './game.js'). page: load this HTML first so code sees its DOM."
```

~60 tokens in the tool block; `stop_serve` leaving pays for most of it.

How it runs (host only, `harness/run-js.js` + a small `runProbe` in `preview-frame.js`):

1. Snapshot the project the same way `serve` does (`PreviewServer._read`), without registering a
   port and without touching the visible preview or its logs.
2. Add a virtual file `__run.js` = `code` and build the document with `buildPreviewDoc`: with
   `page`, that page with a loader appended before `</body>`; without, a blank page with only
   the loader. The loader waits for `load` plus one animation frame (so the page's own modules
   ran), then `await import("./__run.js")` (the builder rewrites relative imports in it and in
   the files it imports, as for any module), then posts `{ t: "done", ms }`. The snippet may use
   top-level `await` (e.g. wait 500 ms, dispatch a `KeyboardEvent`, read `getImageData`).
3. Mount it in a **hidden, fresh, sandboxed frame** through the same relay as the preview (its own
   process, so `while(true){}` cannot freeze the room), with the same capture script. **Relay
   only**: without a preview-origin (production today) run_js is not offered at all, and a run
   frame never falls back to local mode. While a run frame lives the visible previews' watchdogs
   pause (one relay site, one process); when it ends with the relay hung they get fresh frames,
   quietly. Collect console and uncaught errors until `done` plus 150 ms (a timer or the first
   frame that throws right after), or until **3 s** (then the frame is destroyed), or Stop. The
   done message carries a per-run token, so page code calling the hook does not end the run.
4. Result (capped at 30 lines / 1,200 chars; the row that overflows is cut, the last error always
   kept): `ok in 412 ms` or `error in 38 ms` / `timed out after 3 s (a loop that never ends?)`,
   then printed values as they are and warnings / errors with `file:line`. Nothing printed: `ok
   (no output; await async work and print with console.log)`. `page` is a path from the project
   root (else from a served folder).

A test file is just `run_js { code: "import './game.test.js'" }` with `console.assert` /
`throw` inside: no test framework, no new dependency. Peers never run it (the host's tool, like
every tool). Unit tests (Deno) inject a fake runner to cover document building and the result
format; `tests/e2e/preview_browser.mjs` gets three real cases: an import of a project function
printing a value, a throw reported at `file:line`, and an infinite loop that times out while the
page stays responsive.

## 6. Workstream C: the eval suite `tests/eval/`

### C.1 Layout

```
tests/eval/
  tasks/<id>.js       one task per file (below)
  suite.js            runTask / runSuite: DOM side, runs in a page (no Node APIs)
  eval.html           loads suite.js; ?tasks=a,b&model=mock|engine|room
  run.mjs             Node + Playwright driver: serves the repo, opens eval.html, collects results
  results/            .jsonl per run (gitignored), one trajectory JSON per task
```

A task module:

```js
export default {
  id: "calculator",
  prompt: "Build a calculator: buttons 0-9, + - * /, =, C, and a display with id 'display'.",
  files: {},                         // seed project (for fix / extend / refactor tasks)
  maxSteps: 20,
  check: `...`,                      // a JS module run with run_js's probe in the built page
  mock: [xmlCall("write_file", …), …, "Built a calculator."],   // the golden scripted run
};
```

`check` runs through the **same probe as `run_js`** (B.2) on `index.html` (or `task.page`),
so the checks run in the real sandbox, with the real build, and prove `run_js` works too. A check
throws (or `console.error`s) to fail; any uncaught error or console error from the page itself
also fails the task. Checks read the DOM, dispatch events, inspect canvas pixels
(`getImageData`: the canvas is drawn by the same opaque-origin document, so it is not tainted),
and import project modules for pure logic.

### C.2 The 12 tasks

| id | kind | automatic check (in the sandboxed page) |
|---|---|---|
| tetris | build, canvas | a `<canvas>` with non-blank pixels after 500 ms; ArrowLeft / ArrowUp / Space dispatched without errors; the pixel hash changes after a drop |
| snake | build, canvas | canvas non-blank; pixel hash changes over 600 ms (it moves); arrow keys without errors |
| todo | build, DOM | type into the input + Enter / Add twice → 2 items; toggle one → it is marked (class or `checked`); delete → 1 item |
| calculator | build, DOM | click `1`,`2`,`+`,`7`,`=` → display shows `19`; `C` clears; `8`,`/`,`0`,`=` does not throw |
| stopwatch | build, timers | start → after 1.1 s the display changed and is ≥ `00:01`; stop → unchanged after 300 ms; reset → back to zero |
| landing | build, static | one `h1`, ≥ 3 feature cards (repeated sibling elements), a button or link as CTA, a viewport meta, no overflow at 400 px wide (`scrollWidth <= innerWidth`) |
| fix-bug | fix (seeded `cart.js`: `total()` skips the last item, off-by-one) | `import { total } from './cart.js'` → `total([{p:2,q:1},{p:3,q:2}]) === 8` |
| fix-crash | fix (seeded game with a TDZ `ReferenceError` at load) | page loads with no errors; canvas non-blank |
| add-feature | extend (seeded todo app) | "add a Clear completed button": after two adds and one toggle, clicking it leaves 1 item; the old features still pass |
| css | style (seeded page) | "make the header sticky and all buttons 8px rounded": `getComputedStyle(header).position === 'sticky'`, buttons' `borderRadius === '8px'`; page otherwise unchanged (same text content) |
| refactor | multi-file (seeded single `app.js` with helpers) | "move the helpers into `utils.js` and import them": `utils.js` exists and exports `formatTime` and `slugify`; `app.js` imports from it and defines neither; the page still renders the same text |
| logic | pure JS + tests | "implement `wordFreq(text)` in `lib.js` (lower-case, ignore punctuation, sorted by count then word)"; the check imports it and asserts 4 cases; the agent is told a test file exists so it is nudged to use `run_js` |

File-state checks (a file exists, a function is not defined in `app.js`) read the workspace
directly in `suite.js`; behaviour checks run in the probe.

### C.3 Runner

`suite.js` builds, per task: a `MemoryWorkspace` seeded with `files`, `watch`, a `PreviewServer`,
`codingTools` + `previewTools` + `run_js`, and an `Agent` with `CODE_SYSTEM`, auto-approve and
`maxSteps`. The model is pluggable:

- `mock`: `scripted(task.mock)` from `tests/scripted-model.js`. Proves the tasks, the checks and
  the runner; CI-safe, no GPU. Each task also has a unit test that a deliberately wrong final
  file fails its check (so no check passes vacuously).
- `engine`: `engineModel` (one local WebGPU engine, `harness/engine-model.js`) with the model
  from `?model=` and the tools for the constraint. The single-device baseline.
- `room`: the room's `roomModel(api)`, when `suite.js` is loaded in a room page (Code mode reads
  `?eval=` in `room/code.js` and lazily imports `tests/eval/suite.js`; no change to `p2p.html`).
  This is the split-room variant (F2).

Per task it records `{ id, model, ok, reason (done|limit|stuck|context|error), check (message),
steps, calls, cards: {id: n}, forced, prompt tokens prefilled, tokens reused, tokens generated,
wall ms, first-answer ms }` from the agent's `usage` / `card` events, and the trajectory
(`agent.toJSON()`, mini-swe style) for reading failures. `run.mjs` prints a table and a summary
line: `success 10/12 (83 %) · steps 7.4 · prefilled 41k · generated 18k · 612 s`, and exits
non-zero in mock mode if anything fails. `--repeat N` runs each task N times (sampling noise);
`--tasks a,b` selects.

Commands:

```
# now, no GPU (headless Chromium, no WebGPU, the scripted model):
NODE_PATH=<np>:/home/nehanth/bello/node_modules node tests/eval/run.mjs --model mock

# manual, on the GPU machine when nothing else uses it (Chrome with WebGPU, one device):
E2E_GPU=real node tests/eval/run.mjs --model engine --weights qwen36-moe --headed [--tasks tetris,calculator] [--repeat 3]

# manual, in a room (host + a second device), from the Code mode page:
http://localhost:8080/p2p.html?eval=all   (after "cluster online"; results download as .jsonl)
```

`tests/manual/harness_real_model.md` gets a section pointing at these, and the before / after
table for each port in section 3 is filled from `--model engine` runs (the only honest way to
rank rows 1, 7 and 9).

## 7. Order of work

1. C with the mock model (and B.2, which C's checks use): the measuring stick.
2. B.1 grammar + mask cache + `forced` diagnostics (zero tokens; also triages F2).
3. A.3 edit robustness, A.1 cards (format, parts, edit, errors), A.4 quality checks.
4. A.2 prompt / description trim + `run_js` in the tool block; prompt-size test updated.
5. GPU runs (`--model engine`, then the room) when the GPU is free; keep or drop rows 7 and 9 on
   the numbers.

Each step keeps `deno test --allow-read tests/unit`, `tests/e2e/preview_browser.mjs` and
`tests/e2e/harness_tetris.mjs` green.
