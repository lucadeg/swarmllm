// A scripted stand-in for the model, for the agent's unit tests (Deno) and the Code-mode e2e test
// (in the page, as window.__pooledMock.model). It has the shape Agent expects:
//   generate({ system, turns, signal }) -> async iterable of text deltas
// Each call answers with the next reply, streamed in small pieces. A reply is a string, or a
// function (req = { system, turns, call }) -> string that can look at what the agent sent (the
// e2e test checks there that the serve result carried the preview's error).
//   scripted(replies, seen?)                 seen: array that gets a copy of every request
//   scripted(replies, { seen, piece = 5, delay = 0, fallback = "done" })
export function scripted(replies, opts = {}) {
  const { seen = [], piece = 5, delay = 0, fallback = "done" } = Array.isArray(opts) ? { seen: opts } : opts;
  let i = 0;
  return async function* ({ system, turns, signal } = {}) {
    const req = { system, turns: turns.map((t) => ({ ...t })), call: i };
    seen.push(req);
    const r = replies[i++] ?? fallback;
    const text = typeof r === "function" ? String(await r(req)) : r;
    for (let k = 0; k < text.length; k += piece) {
      if (signal?.aborted) return;
      if (delay) await new Promise((res) => setTimeout(res, delay));
      yield text.slice(k, k + piece);
    }
  };
}

// an xml-style tool call (the format harness/tools.js asks Qwen 3.5+ for)
export function xmlCall(name, args = {}) {
  const ps = Object.entries(args).map(([k, v]) => `<parameter=${k}>\n${typeof v === "string" ? v : JSON.stringify(v)}\n</parameter>\n`).join("");
  return `<tool_call>\n<function=${name}>\n${ps}</function>\n</tool_call>`;
}
