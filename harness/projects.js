// Code-mode projects (docs/design/harness-app.md D). Two kinds:
//   "opfs"   - a scratch folder in this browser's origin private file system,
//              pooled-projects/<slug>/; nothing leaves the browser
//   "folder" - a folder on disk the user picked (showDirectoryPicker, Chrome/Edge desktop)
// Metadata (name, kind, updated, the folder's FileSystemDirectoryHandle) lives in IndexedDB,
// since handles are structured-cloneable and localStorage cannot hold them. The agent session of
// each project (Agent.toJSON()) is kept next to it so reopening a project restores the
// conversation. Every ws returned is watch()ed, and writes bump the project's `updated`.
import { DirWorkspace, watch } from "./workspace.js";

export const PROJECTS_DIR = "pooled-projects";
const DB = "pooled-projects", META = "projects", SESS = "sessions";

let dbp = null;
function db() {
  return dbp ||= new Promise((res, rej) => {
    const q = indexedDB.open(DB, 1);
    q.onupgradeneeded = () => { q.result.createObjectStore(META, { keyPath: "id" }); q.result.createObjectStore(SESS); };
    q.onsuccess = () => res(q.result);
    q.onerror = () => { dbp = null; rej(q.error); };
  });
}
async function tx(store, mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(store, mode), q = fn(t.objectStore(store));
    t.oncomplete = () => res(q?.result);
    t.onerror = t.onabort = () => rej(t.error);
  });
}
const getMeta = (id) => tx(META, "readonly", (s) => s.get(id));
const putMeta = (m) => tx(META, "readwrite", (s) => s.put(m));

async function projectsRoot() { return (await navigator.storage.getDirectory()).getDirectoryHandle(PROJECTS_DIR, { create: true }); }

export const slugify = (name) => String(name || "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
  .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/, "") || "project";

export const canOpenFolder = () => typeof window !== "undefined" && "showDirectoryPicker" in window;

export async function listProjects() {
  const all = (await tx(META, "readonly", (s) => s.getAll())) || [];
  return all.map(({ id, name, kind, updated }) => ({ id, name, kind, updated })).sort((a, b) => b.updated - a.updated);
}

// the watched workspace for a project; writes mark it updated (at most every 2 s)
function open(meta, handle) {
  const ws = watch(new DirWorkspace(handle, { private: meta.kind === "folder" }));
  let last = 0;
  ws.onChange(() => {
    const now = Date.now();
    if (now - last < 2000) return;
    last = now;
    getMeta(meta.id).then((m) => m && putMeta({ ...m, updated: now })).catch(() => {});
  });
  return { id: meta.id, name: meta.name, kind: meta.kind, ws };
}

// A new scratch project: "Tetris" -> pooled-projects/tetris/, then tetris-2, tetris-3...
export async function createProject(name) {
  const root = await projectsRoot(), base = slugify(name);
  const taken = new Set();
  for await (const k of root.keys()) taken.add(k);
  for (const p of await listProjects()) taken.add(p.id.replace(/^opfs:/, ""));
  let slug = base;
  for (let i = 2; taken.has(slug); i++) slug = `${base}-${i}`;
  const handle = await root.getDirectoryHandle(slug, { create: true });
  const meta = { id: "opfs:" + slug, name: String(name || slug).trim() || slug, kind: "opfs", updated: Date.now() };
  await putMeta(meta);
  return open(meta, handle);
}

export async function openProject(id) {
  const meta = await getMeta(id);
  if (!meta) throw new Error(`no project ${id}`);
  let handle;
  if (meta.kind === "opfs") handle = await (await projectsRoot()).getDirectoryHandle(id.replace(/^opfs:/, ""), { create: true });
  else {
    handle = meta.handle;
    // permission lapses when the tab closes; asking again needs a user gesture (the open click)
    if ((await handle.queryPermission?.({ mode: "readwrite" })) !== "granted" && (await handle.requestPermission?.({ mode: "readwrite" })) !== "granted") {
      throw new Error(`no permission to edit the folder ${handle.name}`);
    }
  }
  await putMeta({ ...meta, updated: Date.now() });
  return open(meta, handle);
}

// Pick a folder on disk. null when unsupported or cancelled. The same folder picked twice is one project.
export async function openFolder() {
  if (!canOpenFolder()) return null;
  let handle;
  try { handle = await window.showDirectoryPicker({ mode: "readwrite" }); }
  catch (e) { if (e.name === "AbortError") return null; throw e; }
  for (const p of await listProjects()) {
    if (p.kind !== "folder") continue;
    const m = await getMeta(p.id);
    if (await m.handle.isSameEntry?.(handle)) { await putMeta({ ...m, handle, updated: Date.now() }); return open(m, handle); }
  }
  const meta = { id: "folder:" + crypto.getRandomValues(new Uint32Array(2)).join(""), name: handle.name, kind: "folder", updated: Date.now(), handle };
  await putMeta(meta);
  return open(meta, handle);
}

// OPFS projects lose their files; a folder on disk is only forgotten.
export async function deleteProject(id) {
  const meta = await getMeta(id);
  if (meta?.kind === "opfs") await (await projectsRoot()).removeEntry(id.replace(/^opfs:/, ""), { recursive: true }).catch(() => {});
  await tx(META, "readwrite", (s) => s.delete(id));
  await tx(SESS, "readwrite", (s) => s.delete(id));
}

export const saveSession = (id, json) => tx(SESS, "readwrite", (s) => s.put(json, id));
export const loadSession = async (id) => (await tx(SESS, "readonly", (s) => s.get(id))) ?? null;
