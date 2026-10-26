// Tool cards (docs/design/harness-light.md A.1): short recovery notes that cost prompt tokens only
// when a call goes wrong. At most one per step, appended to that call's own <tool_response> as
// "\nhint: <card>", so the system prompt (and the prefix every device caches) never changes.
// The agent skips a card whose last copy is still in the prompt, and sends each at most MAX_PER
// times per request. DOM-free.

export const CARDS = {
  format: "A call is exactly:\n<tool_call>\n<function=read_file>\n<parameter=path>\nindex.html\n</parameter>\n</function>\n</tool_call>\nOne call per <tool_call> block. Nothing after </tool_call>.",
  parts: "Write long files in parts of at most ~100 lines: write_file with the first part, then write_file with append: true for each next part, starting right after the last saved line. Never rewrite what is already saved.",
  edit: "old must be copied exactly from the file as it is now: read_file the lines around the change, copy 2-5 whole lines with their indentation, and change only those. Do not rewrite the file with write_file to fix one spot.",
  runjs: "Fix the code the error points at, then run_js again.",
  scratch: "run_js ran this once and kept nothing: the page does not have this code. Put code the page needs in a file (write_file app.js), load it from the page with edit_file (<script src=\"app.js\"></script> before </body>), then serve.",
  errors: "Fix one error at a time: read_file around the reported line, fix it with edit_file. The preview reloads by itself: check preview_logs, do not serve again.",
  loop: "You already made this exact call and nothing changed since. Change something first, or finish.",
  rewrite: "Most lines were unchanged: use edit_file for small changes.",
};
// most urgent first: when several calls in one step earn a card, the first of these wins
export const PRIORITY = ["loop", "format", "parts", "edit", "errors", "runjs", "scratch", "rewrite"];
export const MAX_PER = { errors: 1 };   // default 2 per request
// JSON-style models (Qwen3 1.7B) get their own format card: the XML one would teach them a
// format they were not trained on
export const CARDS_JSON = {
  format: "A call is exactly:\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"index.html\", \"content\": \"...\"}}\n</tool_call>\nOne call per <tool_call> block, valid JSON, newlines in strings written as \\n.",
};
export const hint = (id, style = "xml") => "\nhint: " + ((style === "json" && CARDS_JSON[id]) || CARDS[id]);

const lines = (s) => (typeof s === "string" ? s.split("\n").length : 0);

// The card one finished call earns, or null.
//   call: { name, arguments, error?, open?, salvage? }   result: the text the model will read
//   repeat: the call was short-circuited as a repeat
export function pickCard({ call, result = "", repeat = false }) {
  if (repeat) return "loop";
  if (call?.garbage) return null;   // the engine's fault, not the model's
  const name = call?.name, err = /^error/.test(result);
  if (call?.salvage || (call?.open && /cut at \d+ tokens/.test(result))) return "parts";
  if (call?.error || /^error: there is no tool called/.test(result)) return "format";
  if (name === "edit_file" && err && /not found in|appears \d+ times/.test(result)) return "edit";
  if (name === "write_file" && !err) {
    if (/old lines unchanged/.test(result)) return "rewrite";
    if (!call.arguments?.append && lines(call.arguments?.content) > 150) return "parts";
  }
  if (name === "run_js" && err) return "runjs";
  // app code (DOM wiring, a game loop) run with run_js instead of written into the page (Qwen3 1.7B)
  if (name === "run_js" && !err && lines(call.arguments?.code) >= 15 && /\bdocument\.|getContext\(|addEventListener\(|requestAnimationFrame\(|setInterval\(/.test(call.arguments?.code)) return "scratch";
  if ((name === "serve" || name === "preview_logs") && (/· \d+ errors?\b/.test(result) || /\] error /.test(result))) return "errors";
  return null;
}
