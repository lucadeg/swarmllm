// Line diff for the approval card and the timeline: what an edit_file / write_file changes.
//   lineDiff(a, b, { max, context }) -> [{ op: " " | "+" | "-", text, skip? }] | null
// Unchanged runs longer than 2*context lines fold into one { op: " ", text: "…", skip: n } row,
// so a one-line edit in a 2,000-line file is a few rows. null when the result would still be
// over `max` rows (the UI then says "new file, N lines" or "N lines changed").
// Common prefix and suffix are cut first; the middle is an LCS table, which is O(n*m), so a
// middle over ~4M cells counts as "everything changed" instead (still a correct diff).

export function lineDiff(a, b, { max = 400, context = 3 } = {}) {
  const A = split(a), B = split(b);
  let p = 0;
  while (p < A.length && p < B.length && A[p] === B[p]) p++;
  let s = 0;
  while (s < A.length - p && s < B.length - p && A[A.length - 1 - s] === B[B.length - 1 - s]) s++;
  if (p === A.length && p === B.length) return [];
  const am = A.slice(p, A.length - s), bm = B.slice(p, B.length - s);
  const rows = [];
  for (let i = 0; i < p; i++) rows.push({ op: " ", text: A[i] });
  rows.push(...middle(am, bm));
  for (let i = A.length - s; i < A.length; i++) rows.push({ op: " ", text: A[i] });
  const out = fold(rows, context);
  return out.length > max ? null : out;
}

const split = (t) => (t == null || t === "" ? [] : String(t).replace(/\n$/, "").split("\n"));

function middle(a, b) {
  const n = a.length, m = b.length;
  if (!n || !m || n * m > 4e6) return [...a.map((text) => ({ op: "-", text })), ...b.map((text) => ({ op: "+", text }))];
  // L[i][j] = LCS length of a[i..], b[j..]
  const L = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ op: " ", text: a[i] }); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) out.push({ op: "-", text: a[i++] });
    else out.push({ op: "+", text: b[j++] });
  }
  while (i < n) out.push({ op: "-", text: a[i++] });
  while (j < m) out.push({ op: "+", text: b[j++] });
  return out;
}

function fold(rows, c) {
  const out = [];
  for (let i = 0; i < rows.length;) {
    if (rows[i].op !== " ") { out.push(rows[i++]); continue; }
    let j = i;
    while (j < rows.length && rows[j].op === " ") j++;
    const head = i === 0 ? 0 : c, tail = j === rows.length ? 0 : c;   // context only next to a change
    if (j - i > head + tail + 1) {
      out.push(...rows.slice(i, i + head), { op: " ", text: "…", skip: j - i - head - tail }, ...rows.slice(j - tail, j));
    } else out.push(...rows.slice(i, j));
    i = j;
  }
  return out;
}
