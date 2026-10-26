/* The Star button: the repo's live star count, cached for ten minutes. If GitHub can't be
   reached (offline, rate-limited), the button just says "Star". Any link with [data-stars] gets it. */
(() => {
  "use strict";
  const links = document.querySelectorAll("[data-stars]");
  if (!links.length) return;
  const KEY = "pooled:stars", TTL = 10 * 60 * 1000;
  const fmt = n => n >= 1000 ? (Math.round(n / 100) / 10).toFixed(1).replace(/\.0$/, "") + "k" : String(n);
  const show = n => links.forEach(a => {
    const s = a.querySelector(".st-n"); if (!s) return;
    s.textContent = fmt(n); s.hidden = false; a.classList.add("has-n");
    a.setAttribute("aria-label", `Star pooled on GitHub, ${n.toLocaleString("en-US")} stars`);
  });
  let c = null;
  try { c = JSON.parse(localStorage.getItem(KEY) || "null"); } catch { c = null; }
  if (c && typeof c.n === "number") show(c.n);
  if (c && typeof c.t === "number" && Date.now() - c.t < TTL) return;
  fetch("https://api.github.com/repos/Nehanth/pooled")
    .then(r => r.ok ? r.json() : Promise.reject(new Error(String(r.status))))
    .then(j => {
      const n = j && j.stargazers_count;
      if (typeof n !== "number") return;
      show(n);
      try { localStorage.setItem(KEY, JSON.stringify({ n, t: Date.now() })); } catch { /* private mode: no cache */ }
    })
    .catch(() => { /* no count: the button says "Star" */ });
})();
