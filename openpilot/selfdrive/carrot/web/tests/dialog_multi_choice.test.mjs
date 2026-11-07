import assert from "node:assert/strict";
import test from "node:test";
import { createDialogController } from "../src/ui/components/dialog/dialog.js";

function makeNode() {
  const classes = new Set();
  const attributes = new Map();
  const listeners = new Map();
  const node = {
    hidden: true,
    value: "",
    disabled: false,
    textContent: "",
    style: {
      removeProperty() {},
      setProperty() {},
    },
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      remove(...names) { names.forEach((name) => classes.delete(name)); },
      contains(name) { return classes.has(name); },
      toggle(name, force) {
        const on = force === undefined ? !classes.has(name) : Boolean(force);
        if (on) classes.add(name);
        else classes.delete(name);
        return on;
      },
    },
    setAttribute(name, value) { attributes.set(name, String(value)); },
    getAttribute(name) { return attributes.has(name) ? attributes.get(name) : null; },
    removeAttribute(name) { attributes.delete(name); },
    addEventListener(type, callback) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(callback);
    },
    click() {
      for (const callback of listeners.get("click") || []) callback();
    },
    focus() {},
    select() {},
    scrollIntoView() {},
    querySelector() { return null; },
    appendChild(child) { return child; },
  };
  Object.defineProperty(node, "className", {
    get() { return [...classes].join(" "); },
    set(value) {
      classes.clear();
      for (const name of String(value).split(/\s+/)) if (name) classes.add(name);
    },
  });
  return node;
}

function harness() {
  const timers = new Map();
  const frames = new Map();
  let serial = 0;
  const nodes = new Map();
  const make = (id) => {
    const node = makeNode();
    nodes.set(id, node);
    return node;
  };
  for (const id of [
    "appDialog", "appDialogBackdrop", "appDialogTitle", "appDialogBody", "appDialogChoices",
    "appDialogInputWrap", "appDialogInput", "appDialogInputError", "appDialogCopy",
    "appDialogDefault", "appDialogCancel", "appDialogConfirm",
  ]) make(id);

  const choiceButtons = [];
  const container = nodes.get("appDialogChoices");
  Object.defineProperty(container, "innerHTML", {
    get() { return ""; },
    set() { choiceButtons.length = 0; },
  });
  container.appendChild = (child) => { choiceButtons.push(child); return child; };
  container.querySelector = (selector) => (selector === ".is-current"
    ? choiceButtons.find((button) => button.classList.contains("is-current")) || null
    : choiceButtons[0] || null);

  const document = {
    getElementById: (id) => nodes.get(id) || null,
    createElement: () => makeNode(),
    activeElement: makeNode(),
    body: makeNode(),
    addEventListener() {},
  };
  const target = {
    history: {
      state: {},
      pushState() {},
      replaceState() {},
      back() {},
      forward() {},
    },
    addEventListener() {},
  };
  const dialog = createDialogController({
    target,
    document,
    setTimeout: (callback) => { timers.set(++serial, callback); return serial; },
    requestAnimationFrame: (callback) => { frames.set(++serial, callback); return serial; },
    cancelAnimationFrame: (id) => frames.delete(id),
    createFocusTrap: () => ({ activate() {}, deactivate() {} }),
  });
  function flush() {
    for (const [id, callback] of [...frames]) { frames.delete(id); callback(); }
    for (const [id, callback] of [...timers]) { timers.delete(id); callback(); }
  }
  return { dialog, nodes, choiceButtons, flush };
}

test("multiple choice keeps confirm visible and resolves every ticked value in choice order", async () => {
  const h = harness();
  const completion = h.dialog.openAppDialog({
    mode: "choice",
    multiple: true,
    choices: [
      { label: "a", value: "a" },
      { label: "b", value: "b", selected: true },
      { label: "c", value: "c" },
    ],
  });
  h.flush();

  const confirm = h.nodes.get("appDialogConfirm");
  assert.equal(confirm.hidden, false);
  assert.equal(confirm.disabled, false);
  assert.equal(h.nodes.get("appDialogConfirm").getAttribute("aria-hidden"), "false");
  assert.equal(h.choiceButtons.length, 3);

  h.choiceButtons[0].click();
  assert.equal(confirm.disabled, false);
  await h.dialog.confirmAppDialog();
  h.flush();

  assert.deepEqual(await completion, ["a", "b"]);
  assert.equal(h.nodes.get("appDialog").hidden, true);
});

test("deselecting the last ticked choice disables confirm", async () => {
  const h = harness();
  const completion = h.dialog.openAppDialog({
    mode: "choice",
    multiple: true,
    choices: [
      { label: "a", value: "a" },
      { label: "b", value: "b", selected: true },
      { label: "c", value: "c" },
    ],
  });
  h.flush();

  const confirm = h.nodes.get("appDialogConfirm");
  const second = h.choiceButtons[1];
  assert.equal(second.classList.contains("is-current"), true);
  assert.equal(second.getAttribute("aria-pressed"), "true");

  second.click();
  assert.equal(confirm.disabled, true);
  assert.equal(second.getAttribute("aria-pressed"), "false");
  assert.equal(second.classList.contains("is-current"), false);

  second.click();
  assert.equal(confirm.disabled, false);
  assert.equal(second.getAttribute("aria-pressed"), "true");

  await h.dialog.cancelAppDialog();
  h.flush();
  assert.equal(await completion, null);
});

test("cancel resolves null for a multiple choice dialog", async () => {
  const h = harness();
  const completion = h.dialog.openAppDialog({
    mode: "choice",
    multiple: true,
    choices: [{ label: "a", value: "a" }],
  });
  h.flush();

  await h.dialog.cancelAppDialog();
  h.flush();
  assert.equal(await completion, null);
});

test("a single-choice dialog still resolves on click and hides confirm", async () => {
  const h = harness();
  const completion = h.dialog.openAppDialog({
    mode: "choice",
    choices: [
      { label: "a", value: "a" },
      { label: "b", value: "b", selected: true },
    ],
  });
  h.flush();

  assert.equal(h.nodes.get("appDialogConfirm").hidden, true);
  assert.equal(h.choiceButtons.length, 2);
  assert.equal(h.choiceButtons[1].getAttribute("aria-current"), "true");
  assert.equal(h.choiceButtons[1].getAttribute("aria-pressed"), null);

  h.choiceButtons[1].click();
  h.flush();
  assert.equal(await completion, "b");
});
