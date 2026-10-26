import { build } from "./_util.js";

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ember Roasters</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="hero">
    <h1>Ember Roasters</h1>
    <p>Small-batch coffee, roasted on Mondays, at your door by Friday.</p>
    <a class="cta" href="#subscribe">Start a subscription</a>
  </header>
  <section class="features">
    <article class="card"><h2>Fresh</h2><p>Every bag is roasted to order, never more than five days old.</p></article>
    <article class="card"><h2>Traceable</h2><p>Each coffee names its farm, its altitude and its harvest.</p></article>
    <article class="card"><h2>Flexible</h2><p>Pause, skip or switch beans whenever you like.</p></article>
  </section>
  <footer id="subscribe"><p>From 14 € a bag. Cancel any time.</p></footer>
</body>
</html>
`;
const css = `* { box-sizing: border-box; }
body { margin: 0; font: 17px/1.5 Georgia, serif; color: #2a1d15; background: #fbf6ef; }
.hero { padding: 64px 20px; text-align: center; background: #3b2418; color: #fbf6ef; }
.hero h1 { margin: 0 0 8px; font-size: clamp(32px, 8vw, 56px); }
.cta { display: inline-block; margin-top: 16px; padding: 12px 22px; border-radius: 999px; background: #e0823d; color: #fff; text-decoration: none; }
.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; max-width: 960px; margin: 40px auto; padding: 0 20px; }
.card { padding: 20px; border-radius: 12px; background: #fff; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08); }
footer { text-align: center; padding: 32px 20px; }
`;
const files = { "index.html": html, "style.css": css };

export default {
  id: "landing",
  kind: "build, static",
  prompt: "Build a landing page for a small coffee roaster: one h1, a features section with at least three feature cards, a call-to-action button or link, and it must work on a phone (viewport meta, no horizontal scroll at 400 px wide).",
  maxSteps: 12,
  viewport: [400, 800],
  check: `
ok($$("h1").length === 1, "expected one h1, found " + $$("h1").length);
ok($('meta[name="viewport"]'), "no viewport meta");
const cards = $$("body *").some((p) => {
  const n = new Map();
  for (const c of p.children) if (c.querySelector("h2, h3, h4, p")) { const k = c.tagName + "." + c.className; n.set(k, (n.get(k) || 0) + 1); }
  return [...n.values()].some((v) => v >= 3);
});
ok(cards, "no group of three or more feature cards (repeated siblings with a heading or text)");
ok($$("a, button").some((e) => text(e).length > 0), "no call-to-action link or button");
ok(document.documentElement.scrollWidth <= innerWidth + 1, "horizontal scroll at " + innerWidth + " px: the page is " + document.documentElement.scrollWidth + " px wide");
`,
  mock: build(files, "Built the landing page, served on :5173."),
  bad: { "style.css": css.replace("minmax(220px, 1fr)", "minmax(520px, 1fr)") },
};
