const FORBIDDEN_KEY = /(api[-_]?key|access[-_]?token|refresh[-_]?token|secret|password|passwd|private[-_]?key|authorization|cookie|session[-_]?token)/i;
const SECRET_VALUE = /(-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._~+\/-]+=*)/i;
const SAFE_SENSITIVITY = new Set(["public", "internal_nonsecret"]);
const SAFE_ACTIONS = new Set(["observe", "recommend", "prepare"]);

function inspect(value, path = "$") {
  if (value == null) return;
  if (typeof value === "string") {
    if (SECRET_VALUE.test(value)) throw new Error(`AutoDev job contains a credential-like value at ${path}`);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => inspect(item, `${path}[${index}]`));
    return;
  }
  if (typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_KEY.test(key)) throw new Error(`AutoDev job contains forbidden secret field ${path}.${key}`);
    inspect(child, `${path}.${key}`);
  }
}

export function normalizeAutoDevJob(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("AutoDev job must be an object");
  inspect(raw);
  const jobId = String(raw.jobId || "").trim();
  const prompt = String(raw.prompt || "").trim();
  const label = String(raw.label || "AutoDev").trim().slice(0, 80) || "AutoDev";
  const sensitivity = String(raw.sensitivity || "").trim().toLowerCase();
  const actionClass = String(raw.actionClass || "").trim().toLowerCase();

  if (!/^[A-Za-z0-9._:-]{6,128}$/.test(jobId)) throw new Error("AutoDev jobId must be 6-128 safe characters");
  if (!prompt) throw new Error("AutoDev prompt is required");
  if (prompt.length > 20000) throw new Error("AutoDev prompt exceeds 20,000 characters");
  if (!SAFE_SENSITIVITY.has(sensitivity)) throw new Error("AutoDev browser compute only accepts public or internal_nonsecret jobs");
  if (!SAFE_ACTIONS.has(actionClass)) throw new Error("AutoDev browser compute only accepts observe, recommend, or prepare jobs");
  if (raw.containsSecrets === true || raw.requiresCredentials === true) throw new Error("Credentialed AutoDev jobs cannot run in the browser swarm");

  return Object.freeze({ jobId, prompt, label, sensitivity, actionClass });
}

export function autoDevStatusSnapshot(state) {
  return Object.freeze({
    schema: "swarmllm.autodev.status.v1",
    backend: "swarmllm-browser",
    availability: state.modelReady && state.isHost && !state.busy ? "ready" : "unavailable",
    persistent: false,
    requiresBrowserPresence: true,
    trustedPeersOnly: true,
    remoteComputeVerified: false,
    roomCode: state.roomCode || null,
    peerId: state.peerId || null,
    model: state.model || null,
    modelReady: Boolean(state.modelReady),
    busy: state.busy || null,
    connectedPeers: Number(state.connectedPeers || 0),
    webgpu: Boolean(state.local?.webgpu),
    gpu: state.local?.gpu || null,
  });
}
