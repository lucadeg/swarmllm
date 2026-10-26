import { build } from "./_util.js";

export const TODO_HTML = `<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Todo</title>
<style>
body { font: 16px system-ui, sans-serif; max-width: 420px; margin: 40px auto; padding: 0 16px; }
#new { width: 100%; padding: 8px; box-sizing: border-box; }
li { display: flex; gap: 8px; align-items: center; padding: 6px 0; }
li.done span { text-decoration: line-through; opacity: 0.6; }
li span { flex: 1; }
</style></head>
<body>
  <h1>Todo</h1>
  <input id="new" placeholder="What needs doing?" autofocus>
  <ul id="list"></ul>
  <script type="module" src="app.js"></script>
</body>
</html>
`;
export const TODO_JS = `const input = document.getElementById("new"), list = document.getElementById("list");
function add(text) {
  const li = document.createElement("li");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.addEventListener("change", () => li.classList.toggle("done", box.checked));
  const span = document.createElement("span");
  span.textContent = text;
  const del = document.createElement("button");
  del.textContent = "Delete";
  del.addEventListener("click", () => li.remove());
  li.append(box, span, del);
  list.append(li);
}
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && input.value.trim()) { add(input.value.trim()); input.value = ""; }
});
`;
// shared with add-feature: add two, toggle one (it must look different), delete it
export const TODO_CHECK = `
const input = $("#new"), list = $("#list");
ok(input && list, "needs an input #new and a list #list");
const addItem = async (t) => {
  input.focus(); input.value = t; input.dispatchEvent(new Event("input", { bubbles: true }));
  if (input.form) input.form.requestSubmit(); else press("Enter", input);
  await sleep(50);
};
await addItem("alpha"); await addItem("beta");
let lis = $$("#list li");
ok(lis.length === 2, "2 items after two adds, got " + lis.length);
const a = lis.find((l) => l.textContent.includes("alpha"));
ok(a, "no item with the text alpha");
const box = a.querySelector("input[type=checkbox]");
ok(box, "an item has no checkbox");
const look = () => [a, ...a.querySelectorAll("*")].map((e) => { const s = getComputedStyle(e); return e.className + "|" + s.textDecorationLine + "|" + s.opacity + "|" + s.color; }).join(";");
const was = look();
box.click(); await sleep(50);
ok(box.checked && look() !== was, "checking an item does not mark it done (no class or style change)");
`;
export const TODO_DELETE = `
const del = [...a.querySelectorAll("button")].find((b) => /delete/i.test(b.textContent));
ok(del, "an item has no Delete button");
del.click(); await sleep(50);
lis = $$("#list li");
ok(lis.length === 1 && lis[0].textContent.includes("beta"), "after Delete: expected only beta, got " + lis.length + " items");
`;
const files = { "index.html": TODO_HTML, "app.js": TODO_JS };

export default {
  id: "todo",
  kind: "build, DOM",
  prompt: "Build a todo list app: an input with id 'new' where Enter adds an item; items are <li> elements in <ul id=\"list\">, each with a checkbox that marks it done (strike it through) and a 'Delete' button that removes it.",
  maxSteps: 16,
  check: TODO_CHECK + TODO_DELETE,
  mock: build(files, "Built the todo app, served on :5173."),
  bad: { "app.js": TODO_JS.replace(`del.addEventListener("click", () => li.remove());`, "") },
};
