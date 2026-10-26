import { xmlCall } from "./_util.js";

const test = `import { wordFreq } from "./lib.js";
const eq = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(m + ": got " + JSON.stringify(a)); };
eq(wordFreq("the cat and the hat"), [["the", 2], ["and", 1], ["cat", 1], ["hat", 1]], "count, then word");
console.log("lib.test.js: all tests passed");
`;
const lib = `// word -> count pairs: lower-cased, punctuation ignored, most frequent first, ties alphabetical
export function wordFreq(text) {
  const counts = new Map();
  for (const w of text.toLowerCase().match(/[a-z0-9]+/g) || []) counts.set(w, (counts.get(w) || 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}
`;
const runner = `<!doctype html>
<html><head><meta charset="utf-8"><title>tests</title></head>
<body><script type="module" src="lib.test.js"></script></body></html>
`;

export default {
  id: "logic",
  kind: "pure JS",
  prompt: "Implement and export wordFreq(text) in lib.js: lower-case the words, ignore punctuation, and return [word, count] pairs sorted by count (highest first), then by word (a to z). lib.test.js has a test for it.",
  files: { "lib.test.js": test },
  maxSteps: 12,
  check: `
import { wordFreq } from "./lib.js";
const eq = (a, b, m) => ok(JSON.stringify(a) === JSON.stringify(b), m + ": got " + JSON.stringify(a));
eq(wordFreq("The cat and the hat."), [["the", 2], ["and", 1], ["cat", 1], ["hat", 1]], "case and punctuation");
eq(wordFreq(""), [], "empty text");
eq(wordFreq("B a b, A!"), [["a", 2], ["b", 2]], "ties sorted by word");
eq(wordFreq("one two  three\\ntwo three three"), [["three", 3], ["two", 2], ["one", 1]], "counts");
`,
  mock: [
    xmlCall("read_file", { path: "lib.test.js" }),
    xmlCall("write_file", { path: "lib.js", content: lib }),
    "Let me run the test in a page.\n" + xmlCall("write_file", { path: "index.html", content: runner }),
    (req) => xmlCall("serve", { port: 5173 }),
    "wordFreq is in lib.js and lib.test.js passes.",
  ],
  bad: { "lib.js": lib.replace(" || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)", "") },
};
