const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const app = fs.readFileSync(path.join(__dirname, '../../src/helix/ui/static/app.js'), 'utf8');
const token = 'dashboard-test-token';
const cid = 'g2-s4';

class Element {
  constructor() {
    this.style = {};
    this.children = [];
    this.textContent = '';
  }
  setAttribute() {}
  addEventListener() {}
  replaceChildren() { this.children = []; }
  append(child) { this.children.push(child); }
  text() {
    return this.textContent + this.children.map((child) => child instanceof Element ? child.text() : child).join('');
  }
}

function dashboard(fetch) {
  const location = {
    pathname: '/', search: '', hash: '#token=' + token, reloads: 0,
    reload() { this.reloads++; },
  };
  const storage = new Map();
  const candidate = new Element();
  const context = {
    fetch, location, URLSearchParams, TextDecoder, Node: Element,
    window: { ReadableStream, TextDecoder },
    document: { getElementById: () => candidate, createElement: () => new Element() },
    history: { replaceState: (_state, _title, url) => { assert.equal(url, '/'); location.hash = ''; } },
    sessionStorage: { setItem: (key, value) => storage.set(key, value), getItem: (key) => storage.get(key) },
    matchMedia: () => ({ matches: false }),
    setTimeout: () => {}, setInterval: () => {},
  };
  // Load the real script with a minimal DOM and replace auto-boot so tests can drive each update.
  vm.runInNewContext(app.replace('  boot();', `
    globalThis.ui = { state: S, authorize: () => { token = loadToken(); }, connect, fromHash, get, index, renderView };
  `), context);
  return { ...context.ui, location, candidate };
}

test('fragment login sends only an explicit header on same-origin requests', async () => {
  const ui = dashboard(async (url, options) => {
    assert.equal(url, '/api/run');
    assert.equal(options.headers.Authorization, 'Bearer ' + token);
    assert.equal(options.mode, 'same-origin');
    assert.equal(options.credentials, 'omit');
    assert.equal(options.redirect, 'error');
    return { ok: true, json: async () => ({ ready: true }) };
  });
  ui.authorize();
  assert.equal(ui.location.hash, '');
  ui.authorize(); // Refreshing the tab reuses its origin-scoped session storage.
  assert.equal((await ui.get('/api/run')).ready, true);
});

test('opening a fresh launch link in the same tab reloads with the new token', async () => {
  const ui = dashboard(async (_, options) => {
    assert.equal(options.headers.Authorization, 'Bearer fresh-token');
    return { ok: true, json: async () => ({ ready: true }) };
  });
  ui.authorize();
  ui.location.hash = '#token=fresh-token';
  ui.fromHash();
  assert.equal(ui.location.reloads, 1);
  assert.equal(ui.location.hash, '#token=fresh-token');
  ui.authorize();
  assert.equal(ui.location.hash, '');
  assert.equal((await ui.get('/api/run')).ready, true);
});

test('the authenticated event stream decodes split frames and skips keepalives', async () => {
  const bytes = Buffer.from('event: run\ndata: {"label":"caf\u00e9"}\n\n: keepalive\n\nevent: run\ndata: {"step":2}\n\n');
  const split = bytes.indexOf(0xc3) + 1;
  let called = false;
  const ui = dashboard(async (url, options) => {
    assert.equal(url, '/api/events');
    assert.equal(options.headers.Authorization, 'Bearer ' + token);
    called = true;
    return {
      ok: true,
      body: new ReadableStream({
        start(stream) {
          stream.enqueue(bytes.subarray(0, split));
          stream.enqueue(bytes.subarray(split, bytes.indexOf(': keepalive')));
          stream.enqueue(bytes.subarray(bytes.indexOf(': keepalive')));
          stream.close();
        },
      }),
    };
  });
  const frames = [];
  Object.defineProperty(ui.state, 'run', { set: (frame) => frames.push(frame) });
  ui.authorize();
  await ui.connect();
  assert.equal(called, true);
  assert.equal(frames.length, 2);
  assert.equal(frames[0].label, 'caf\u00e9');
  assert.equal(frames[1].step, 2);
});

test('the candidate view refetches an open proposal when its outcome changes', async () => {
  const inProgress = { id: cid, accepted: false, worktree_exists: true };
  const accepted = { id: cid, accepted: true, mean: 0.75, diff: '+updated code', worktree_exists: true };
  const responses = [inProgress, accepted];
  const ui = dashboard(async (url) => {
    assert.equal(url, '/api/candidate/' + cid + '?vs=seed');
    assert.ok(responses.length);
    const detail = responses.shift();
    return { ok: true, json: async () => detail };
  });
  ui.authorize();
  Object.assign(ui.state, { view: 'candidate', cand: cid, candVs: 'seed' });
  ui.state.run = { status: { state: 'running' }, now: { lanes: [{ id: cid, phase: 'agent' }] } };
  ui.renderView();
  await ui.state.candRequest;
  assert.match(ui.candidate.text(), /in progress/);

  ui.state.run.now.lanes[0].phase = 'validation';
  ui.renderView();
  assert.equal(responses.length, 1);

  ui.state.run.candidates = [accepted];
  ui.index();
  ui.renderView();
  await ui.state.candRequest;
  assert.equal(responses.length, 0);
  assert.match(ui.candidate.text(), /0\.750/);
  assert.match(ui.candidate.text(), /\+updated code/);
  assert.doesNotMatch(ui.candidate.text(), /in progress/);
});
