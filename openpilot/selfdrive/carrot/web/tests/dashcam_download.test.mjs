import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";

const root = path.resolve(import.meta.dirname, "..");
const read = (relativePath) => fs.readFileSync(path.join(root, relativePath), "utf8");

const dashcamSource = read("src/features/logs/dashcam.js");
const runtimeSource = read("src/features/logs/runtime.js");

const START_MARK = "// --- dashcam downloads ---";
const END_MARK = "// --- end dashcam downloads ---";

function sliceWithMarkers(source) {
  const start = source.indexOf(START_MARK);
  const end = source.indexOf(END_MARK);
  assert.ok(start >= 0, "download block start marker is missing");
  assert.ok(end > start, "download block end marker is missing");
  return source.slice(start, end);
}

const blockSource = sliceWithMarkers(dashcamSource);

function loadBlock(overrides = {}) {
  const clicks = [];
  const gaps = [];
  const toasts = [];
  const jsonCalls = [];
  const dialogCalls = [];

  const filesBySegment = overrides.filesBySegment ?? {};

  const context = vm.createContext({
    getJson: async (href) => {
      jsonCalls.push(href);
      const segment = decodeURIComponent(String(href).split("/").pop());
      if (overrides.getJsonThrows) throw new Error("boom");
      return { files: filesBySegment[segment] ?? [] };
    },
    openAppDialog: async (options) => {
      dialogCalls.push(options);
      return overrides.dialogResult ?? null;
    },
    showAppToast: (message, options) => toasts.push({ message, options }),
    getUIText: (key, fallback, params) => (params && params.count !== undefined ? fallback.replace("{count}", params.count) : fallback),
    formatLogBytes: () => "1 MB",
    dashcamApiPath: (kind, segment) => `/api/dashcam/${kind}/${encodeURIComponent(segment)}`,
    document: {
      createElement: () => {
        const link = { href: "", style: {}, remove: () => {}, click: () => clicks.push(link.href) };
        return link;
      },
      body: { appendChild: () => {} },
    },
    setTimeout: (fn, ms) => {
      gaps.push(ms);
      fn();
    },
  });

  const body = blockSource + "\n;__exports = { downloadDashcamSegments, dashcamDownloadLabel, startDashcamDownload };";
  const wrapped = new vm.Script(body);
  const sandbox = context;
  sandbox.__exports = undefined;
  wrapped.runInContext(context);
  return { ...context.__exports, clicks, gaps, toasts, jsonCalls, dialogCalls };
}

test("downloadDashcamSegments skips segments without the chosen kind", async () => {
  const harness = loadBlock({
    dialogResult: "fcamera",
    filesBySegment: {
      a: [{ kind: "fcamera" }],
      b: [{ kind: "qcamera" }],
      c: [{ kind: "fcamera" }],
    },
  });

  await harness.downloadDashcamSegments(["a", "b", "c"]);

  assert.deepEqual(harness.clicks, ["/api/dashcam/download/a/fcamera", "/api/dashcam/download/c/fcamera"]);
  assert.deepEqual(harness.gaps, [2000]);
  assert.equal(harness.toasts.length, 1);
  assert.match(harness.toasts[0].message, /2 downloads started/);
  assert.match(harness.toasts[0].message, /1 without this file/);
});

test("downloadDashcamSegments aborts without touching the network when the dialog is dismissed", async () => {
  const harness = loadBlock({ dialogResult: null });

  await harness.downloadDashcamSegments(["a", "b"]);

  assert.deepEqual(harness.jsonCalls, []);
  assert.deepEqual(harness.clicks, []);
  assert.deepEqual(harness.toasts, []);
});

test("downloadDashcamSegments rejects an empty selection before opening the dialog", async () => {
  const harness = loadBlock({ dialogResult: "fcamera" });

  await harness.downloadDashcamSegments([]);

  assert.deepEqual(harness.dialogCalls, []);
  assert.equal(harness.toasts.length, 1);
  assert.equal(harness.toasts[0].message, "No segments selected.");
  assert.equal(harness.toasts[0].options?.tone, "error");
});

test("downloadDashcamSegments falls back to every kind when the file listing fails", async () => {
  const harness = loadBlock({ dialogResult: "fcamera", getJsonThrows: true });

  await harness.downloadDashcamSegments(["a", "b", "c"]);

  assert.deepEqual(harness.clicks, [
    "/api/dashcam/download/a/fcamera",
    "/api/dashcam/download/b/fcamera",
    "/api/dashcam/download/c/fcamera",
  ]);
});

test("dashcamDownloadLabel shows the size only when the server reported one", () => {
  const harness = loadBlock();

  assert.equal(harness.dashcamDownloadLabel({ kind: "fcamera", size: 1048576 }), "fcamera Download · 1 MB");
  assert.equal(harness.dashcamDownloadLabel({ kind: "fcamera", size: 0 }), "fcamera Download");
});

test("dashcam upload entry points are gone from the source", () => {
  assert.doesNotMatch(dashcamSource, /value: "upload"/);
  assert.doesNotMatch(dashcamSource, /data-action="upload-selected"/);
  assert.doesNotMatch(dashcamSource, /onSegmentSend:/);
  assert.match(dashcamSource, /data-action="download-selected"/);

  assert.doesNotMatch(runtimeSource, /resumeDashcamUploadJobIfNeeded\(\)/);
  assert.doesNotMatch(runtimeSource, /LOGS_MENU_UPLOAD/);
  assert.doesNotMatch(runtimeSource, /uploadRecentDashcamSegments/);
  assert.match(runtimeSource, /downloadDashcamSegments\(targets\)/);
});
