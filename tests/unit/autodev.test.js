import { assertEquals, assertThrows } from "jsr:@std/assert";
import { autoDevStatusSnapshot, normalizeAutoDevJob } from "../../room/autodev.js";

Deno.test("accepts bounded non-sensitive planning jobs", () => {
  const job = normalizeAutoDevJob({
    jobId: "JOB-AUTODEV-001",
    prompt: "Review this public architecture and identify test gaps.",
    label: "verification",
    sensitivity: "public",
    actionClass: "recommend",
  });
  assertEquals(job.actionClass, "recommend");
  assertEquals(job.sensitivity, "public");
});

Deno.test("rejects secret-bearing fields and credential-looking values", () => {
  assertThrows(() => normalizeAutoDevJob({
    jobId: "JOB-SECRET-001",
    prompt: "inspect",
    sensitivity: "public",
    actionClass: "observe",
    api_key: "never",
  }));
  assertThrows(() => normalizeAutoDevJob({
    jobId: "JOB-SECRET-002",
    prompt: "Use Bearer abcdefghijklmnopqrstuvwxyz012345",
    sensitivity: "public",
    actionClass: "observe",
  }));
});

Deno.test("rejects irreversible execution classes", () => {
  assertThrows(() => normalizeAutoDevJob({
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
  assertEquals(status.availability, "ready");
  assertEquals(status.persistent, false);
  assertEquals(status.requiresBrowserPresence, true);
  assertEquals(status.remoteComputeVerified, false);
});
