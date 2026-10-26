# 28 · Code mode: agent robustness

**Phase:** code mode · **Status:** in progress

## Why
Code mode is only as good as its agent loop. Open models do not always call tools the way the prompt asks. Qwen 3.6 sometimes writes a tool as bare tags without `<tool_call>`, a model can loop on the same failing edit, and an error in the preview only helps if the agent reads it and acts on it. Every one of these turns "build me a game" into a stuck room.

## Design
- **Tool-call parsing.** Accept the formats models actually produce (Qwen XML, JSON, bare known-tool tags) and remind the model of the call format when it slips. `harness/agent.js` already runs known tools written as bare tags.
- **Loops.** A repeated identical tool call is already short-circuited (docs/design/harness-light.md). Extend it to the same error coming back several times in a row: tell the model what it is repeating, and stop with a clear message if it keeps going.
- **Reading errors.** After each change, the preview's console errors go back to the agent with the file and line. The agent should fix the first real error, not the noise after it.
- **Tests.** Each failure mode gets a scripted-model case (`tests/scripted-model.js`) in `tests/e2e/harness_tetris.mjs` or `tests/e2e/agent_synth.mjs`, so it runs with no GPU.

## Done when
- The Tetris acceptance test (`npm run e2e:code`) passes with the model writing tool calls in each supported format.
- A scripted model that repeats a failing edit is stopped with a clear message instead of running out of steps.
- On a real room with Qwen 3.6 35B MoE, "build a tetris game" ends with a working preview in most runs, and the misses are filed as issues.
