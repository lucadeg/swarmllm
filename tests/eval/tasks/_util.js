// Shared by the task modules: the golden scripted runs (the mock model's replies).
import { xmlCall } from "../../scripted-model.js";
export { xmlCall };

// one write_file per file (a file given as an array is written in parts, append: true after the
// first), then serve, then the answer
export function build(files, answer, { pre = [], serve = true } = {}) {
  const out = [...pre];
  for (const [path, v] of Object.entries(files)) {
    const parts = Array.isArray(v) ? v : [v];
    parts.forEach((content, i) => out.push(xmlCall("write_file", i ? { path, content, append: true } : { path, content })));
  }
  if (serve) out.push(xmlCall("serve", { port: 5173 }));
  out.push(answer);
  return out;
}
// the files of build()'s input as they end up on disk
export const joined = (files) => Object.fromEntries(Object.entries(files).map(([p, v]) => [p, Array.isArray(v) ? v.join("") : v]));
