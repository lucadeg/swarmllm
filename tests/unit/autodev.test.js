import { autoDevStatusSnapshot, normalizeAutoDevJob, parseAutoJoinParams } from "../../room/autodev.js";

const eq = (actual, expected, message = "mismatch") => {
  const left = JSON.stringify(actual);
  const right = JSON.stringify(expected);
  if (left !== right) throw new Error(`${message}: ${left} != ${right}`);
};
const throws = (fn, message = "expected function to throw") => {
  let didThrow = false;
  try { fn(); } catch { didThrow = true; }
  if (!didThrow) throw new Error(message);
};

Deno.test("accepts bounded non-sensitive planning jobs", () => {
  const job = normalizeAutoDevJob({
    jobId: "JOB-AUTODEV-001",
    prompt: "Review this public architecture and identify test gaps.",
    label: "verification",
    sensitivity: "public",
    actionClass: "recommend",
  });
  eq(job.actionClass, "recommend");
  eq(job.sensitivity, "public");
});

Deno.test("rejects secret-bearing fields and credential-looking values", () => {
  throws(() => normalizeAutoDevJob({
    jobId: "JOB-SECRET-001",
    prompt: "inspect",
    sensitivity: "public",
    actionClass: "observe",
    api_key: "never",
  }));
  throws(() => normalizeAutoDevJob({
    jobId: "JOB-SECRET-002",
    prompt: "Use Bearer abcdefghijklmnopqrstuvwxyz012345",
    sensitivity: "public",
    actionClass: "observe",
  }));
});

Deno.test("rejects irreversible execution classes", () => {
  throws(() => normalizeAutoDevJob({
    jobId: "JOB-DEPLOY-001",
    prompt: "Deploy production.",
    sensitivity: "public",
    actionClass: "deploy",
  }));
});

Deno.test("status is honest about opportunistic compute", () => {
  const status = autoDevStatusSnapshot({
    roomCode: "ABCD",
    isHost: true,
    peerId: "p1",
    modelReady: true,
    busy: null,
    model: "qwen3-4b",
    connectedPeers: 2,
    local: { webgpu: true, gpu: "GPU" },
  });
  eq(status.availability, "ready");
  eq(status.persistent, false);
  eq(status.requiresBrowserPresence, true);
  eq(status.remoteComputeVerified, false);
});

Deno.test("parses explicit worker autojoin without hidden defaults", () => {
  const cfg = parseAutoJoinParams("?code=abcd&autojoin=1&name=tablet&gb=6");
  eq(cfg.code, "ABCD");
  eq(cfg.autojoin, true);
  eq(cfg.autocreate, false);
  eq(cfg.name, "tablet");
  eq(cfg.gb, 6);
});

Deno.test("rejects ambiguous auto-start modes", () => {
  throws(() => parseAutoJoinParams("?code=ABCD&autojoin=1&autocreate=1"));
});
