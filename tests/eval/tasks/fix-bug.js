import { xmlCall } from "./_util.js";

const BUG = "  for (let i = 0; i < items.length - 1; i++) {";
const FIX = "  for (let i = 0; i < items.length; i++) {";
const cart = `// the cart's total price: items are { p: price, q: quantity }
export function total(items) {
  let sum = 0;
${BUG}
    sum += items[i].p * items[i].q;
  }
  return sum;
}
`;
const app = `import { total } from "./cart.js";
const items = [{ name: "Beans", p: 12, q: 2 }, { name: "Filter papers", p: 3, q: 1 }, { name: "Mug", p: 9, q: 1 }];
const ul = document.getElementById("items");
for (const it of items) {
  const li = document.createElement("li");
  li.textContent = it.name + " x" + it.q + ": " + it.p * it.q + " €";
  ul.append(li);
}
document.getElementById("total").textContent = "Total: " + total(items) + " €";
`;
const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>Cart</title></head>
<body><h1>Cart</h1><ul id="items"></ul><p id="total"></p><script type="module" src="app.js"></script></body></html>
`;

export default {
  id: "fix-bug",
  kind: "fix",
  prompt: "The cart total on the page is wrong: it should be 36 € but it shows 27 €. Find the bug and fix it.",
  files: { "index.html": html, "app.js": app, "cart.js": cart },
  maxSteps: 12,
  check: `
import { total } from "./cart.js";
ok(total([{ p: 2, q: 1 }, { p: 3, q: 2 }]) === 8, "total([{p:2,q:1},{p:3,q:2}]) is " + total([{ p: 2, q: 1 }, { p: 3, q: 2 }]) + ", not 8");
ok(total([]) === 0, "total([]) is not 0");
ok(total([{ p: 5, q: 3 }]) === 15, "total of one item is wrong");
ok(/36/.test(text($("#total"))), "the page shows '" + text($("#total")) + "'");
`,
  mock: [
    "Let me look at how the total is computed.\n" + xmlCall("read_file", { path: "cart.js" }),
    "The loop stops one item early.\n" + xmlCall("edit_file", { path: "cart.js", old: BUG, new: FIX }),
    xmlCall("serve", { port: 5173 }),
    "Fixed: total() skipped the last item (the loop ran to length - 1).",
  ],
  bad: {},
};
