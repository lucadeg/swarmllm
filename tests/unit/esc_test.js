// room/markdown.js esc(): peer-chosen text must be safe inside quoted attributes as well as text.
import { esc } from "../../room/markdown.js";

Deno.test("esc neutralises quotes, so a name cannot leave title=\"...\" or data-name='...'", () => {
  const name = `x" onmouseover="alert(1)' <img src=x onerror=alert(1)>&`;
  const out = esc(name);
  if (/["'<>]/.test(out)) throw new Error(out);
  if (out !== "x&quot; onmouseover=&quot;alert(1)&#39; &lt;img src=x onerror=alert(1)&gt;&amp;") throw new Error(out);
});
