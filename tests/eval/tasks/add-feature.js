import { xmlCall } from "./_util.js";
import { TODO_HTML, TODO_JS, TODO_CHECK, TODO_DELETE } from "./todo.js";

const OLD = `  <ul id="list"></ul>\n`;
const NEW = `  <ul id="list"></ul>\n  <button id="clear">Clear completed</button>\n`;
const ADD = `document.getElementById("clear").addEventListener("click", () => {
  for (const li of list.querySelectorAll("li")) if (li.querySelector("input[type=checkbox]").checked) li.remove();
});
`;

export default {
  id: "add-feature",
  kind: "extend",
  prompt: "Add a 'Clear completed' button below the list that removes every checked item. Keep everything else working.",
  files: { "index.html": TODO_HTML, "app.js": TODO_JS },
  maxSteps: 14,
  check: TODO_CHECK + `
const clear = button("Clear completed");
ok(clear, "no 'Clear completed' button");
clear.click(); await sleep(50);
lis = $$("#list li");
ok(lis.length === 1 && lis[0].textContent.includes("beta"), "Clear completed should leave only beta, got " + lis.length + " items");
await addItem("gamma");
const g = $$("#list li").find((l) => l.textContent.includes("gamma"));
ok(g && g.querySelector("input[type=checkbox]"), "a new item after Clear completed has no checkbox");
` + TODO_DELETE.replace("a.querySelectorAll", "g.querySelectorAll").replace(`lis[0].textContent.includes("beta")`, `!lis.some((l) => l.textContent.includes("gamma"))`).replace("only beta", "gamma gone"),
  mock: [
    xmlCall("read_file", { path: "app.js" }),
    xmlCall("edit_file", { path: "index.html", old: OLD, new: NEW }),
    xmlCall("write_file", { path: "app.js", content: ADD, append: true }),
    xmlCall("serve", { port: 5173 }),
    "Added a 'Clear completed' button that removes the checked items.",
  ],
  bad: {},   // the seed: no such button
};
