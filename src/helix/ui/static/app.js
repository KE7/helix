// helix ui: a read-only live view of one HELIX project. Vanilla JS, no build step, no external requests.
// Every string from the API reaches the page as a text node or an attribute value, never as markup.
'use strict';
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const ID_RE = /^g\d+-[sm]\d+$/;
  const VIEWS = ['map', 'frontier', 'candidate', 'mutation', 'validation', 'lineage', 'proposals'];
  const MENU = [['map', 'Loop'], ['frontier', 'Frontier'], ['lineage', 'Lineage'], ['proposals', 'Proposals']];
  const OUTCOMES = ['frontier', 'dominated', 'rejected', 'skipped', 'failed', 'running'];
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const dark = matchMedia('(prefers-color-scheme: dark)');
  const $ = (id) => document.getElementById(id);

  const S = {
    static: null, run: null, recvAt: 0, byId: {}, front: [], rank: {},
    frontier: null, frontierKey: null, frontierLoading: null, viewKey: null,
    view: 'map', from: 'map', cand: null, candData: null,
    lane: null, tx: null, mapKey: null, laneKey: null, popKey: null, chart: null, phase: 0, error: null,
    lin: { data: null, key: null, loading: null, byId: {}, showRej: null, pts: [], pin: null, hov: null, g: null, ix: null },
    prop: { data: null, key: null, loading: null, byId: {}, filter: 'all', sort: 'gen', dir: 1, sel: null, rows: [], bodyKey: null },
    candVs: 'parent', kids: {}, inst: null, ci: null, pobj: null, pobjKey: null, opop: { pin: false, timer: 0 },
  };

  // ---------------- DOM helpers (no innerHTML anywhere) ----------------
  function attrs(e, a) {
    if (!a) return;
    for (const k of Object.keys(a)) {
      const v = a[k];
      if (v == null || v === false) continue;
      if (k === 'style') {
        for (const p of Object.keys(v)) {
          if (v[p] == null) continue;
          if (p.startsWith('--')) e.style.setProperty(p, String(v[p]));
          else e.style[p] = String(v[p]);
        }
      } else if (k === 'text') e.textContent = String(v);
      else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : String(v));
    }
  }
  function kids(e, list) {
    for (const k of list.flat(Infinity)) {
      if (k == null || k === false) continue;
      e.append(k instanceof Node ? k : String(k));
    }
  }
  function h(tag, a, ...c) { const e = document.createElement(tag); attrs(e, a); kids(e, c); return e; }
  function s(tag, a, ...c) { const e = document.createElementNS(NS, tag); attrs(e, a); kids(e, c); return e; }
  function fill(e, ...c) { e.replaceChildren(); kids(e, c); return e; }
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // ---------------- formatting ----------------
  const num = (v) => typeof v === 'number' && Number.isFinite(v);
  const nowSec = () => Date.now() / 1000;
  const med = (a) => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : null; };
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  function compact(n) {
    if (!num(n)) return '–';
    const a = Math.abs(n);
    const f = (v, u) => (v < 10 ? v.toFixed(1).replace(/\.0$/, '') : String(Math.round(v))) + u;
    if (a < 1000) return String(Math.round(n));
    if (a < 1e6) return f(n / 1e3, 'k');
    if (a < 1e9) return f(n / 1e6, 'M');
    return f(n / 1e9, 'B');
  }
  function dur(sec) {
    if (!num(sec)) return '–';
    if (sec < 90) return Math.round(sec) + ' s';
    if (sec < 90 * 60) return Math.round(sec / 60) + ' min';
    const hrs = sec / 3600;
    if (hrs >= 10) return Math.round(hrs) + ' h';
    const H = Math.floor(hrs), M = Math.round((sec - H * 3600) / 60);
    return M ? `${H} h ${M} m` : `${H} h`;
  }
  function clock(sec) {
    if (!num(sec)) return '–';
    sec = Math.max(0, Math.floor(sec));
    const H = Math.floor(sec / 3600), M = Math.floor(sec / 60) % 60, X = sec % 60, p = (v) => String(v).padStart(2, '0');
    return H ? `${H}:${p(M)}:${p(X)}` : `${p(M)}:${p(X)}`;
  }
  function money(v) {
    if (!num(v)) return '–';
    if (v < 10) return '$' + v.toFixed(2);
    if (v < 10000) return '$' + Math.round(v).toLocaleString('en-US');
    return '$' + compact(v);
  }
  const pct = (v) => (num(v) ? Math.round(100 * v) + '%' : '–');
  const fixed = (v, d) => (num(v) ? v.toFixed(d) : '–');
  const binary = () => !!(S.run && S.run.binary_scores);
  function score(c) {
    if (!c || !num(c.mean)) return '–';
    if (binary() && num(c.n)) return `${num(c.successes) ? c.successes : Math.round(c.mean * c.n)}/${c.n}`;
    return c.mean.toFixed(3);
  }
  const plural = (n, one, many) => (n === 1 ? one : many);
  // objective values come in the evaluator's units: keep them short whatever their scale
  const ofmt = (v) => (!num(v) ? '–' : Math.abs(v) >= 1000 ? compact(v) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* per-viewer nicety only */ } } };
  // proposals and lineage nodes: lineage carries gate_score/gate_basis, proposals score/basis; a gate score is never a mean
  const pScore = (x) => (num(x.score) ? x.score : num(x.gate_score) ? x.gate_score : null);
  const pBasis = (x) => x.basis || x.gate_basis || null;
  function propScore(x) {
    const v = pScore(x);
    if (v == null) return '–';
    if (binary() && num(x.successes) && num(x.n)) return `${x.successes}/${x.n}`;
    return v.toFixed(3);
  }
  const basisTag = (x) => (pScore(x) == null ? '' : pBasis(x) === 'val' ? 'val' : 'gate');
  const basisWord = (x) => (pScore(x) == null ? '' : pBasis(x) === 'val' ? (binary() ? 'succeeded' : 'mean') : 'gate');
  const outcomeOf = (x) => x.outcome || (x.accepted ? (x.front ? 'frontier' : 'dominated') : 'rejected');
  const REASONS = {
    minibatch_gate: 'gate', train_gate: 'train gate', val_stage: 'val stage', duplicate_child: 'duplicate', proposal_selection: 'not selected',
    merge_no_change: 'no change', merge_below_parents: 'below parents', perfect_subsample: 'perfect parent', 'failed mutation': 'mutation failed',
    'tamper-rejected mutation candidate': 'tampered', 'child whose minibatch eval raised': 'gate raised',
  };
  const why = (x) => (x.reason ? REASONS[x.reason] || String(x.reason).replace(/_/g, ' ') : '');
  const frontierType = () => (S.run && S.run.frontier && S.run.frontier.type) || S.static.config.frontier_type;

  // ---------------- API ----------------
  class HttpError extends Error { constructor(status) { super('HTTP ' + status); this.status = status; } }
  async function get(path) {
    const r = await fetch(path, { credentials: 'same-origin', cache: 'no-store' });
    if (!r.ok) throw new HttpError(r.status);
    return r.json();
  }

  // ---------------- state indexing ----------------
  function index() {
    const r = S.run;
    S.byId = {};
    for (const c of (r && r.candidates) || []) S.byId[c.id] = c;
    const ids = (r && r.frontier && r.frontier.ids) || [];
    S.front = ids.map((id) => S.byId[id] || { id, mean: null }).sort((a, b) => (b.mean ?? -Infinity) - (a.mean ?? -Infinity));
    S.rank = {};
    S.front.forEach((c, i) => { S.rank[c.id] = i + 1; });
  }
  const running = () => !!(S.run && S.run.status && S.run.status.state === 'running');
  const animate = () => running() && !reduceMotion.matches;
  function staleLimit() {
    const sb = S.static.config.sandbox || {};
    const typ = S.run && S.run.now && S.run.now.typical_seconds;
    return Math.max(3600, 2 * (sb.timeout_seconds || 2700), 2 * (typ || 0));
  }

  // ---------------- header ----------------
  function stateEl() {
    const r = S.run;
    if (!r) return h('span', { class: 'live none' }, h('i'), 'Not started');
    const st = r.status || {}, detail = st.detail || null;
    const stale = num(st.stale_seconds) ? st.stale_seconds + (nowSec() - S.recvAt) : null;
    if (st.state === 'running') {
      if (stale != null && stale > staleLimit()) {
        return h('span', { class: 'live idle', title: `no timing record for ${dur(stale)}` }, h('i'), 'Running?');
      }
      return h('span', { class: 'live', title: stale != null ? `last timing record ${dur(stale)} ago` : null }, h('i'), 'Running');
    }
    if (st.state === 'finished') return h('span', { class: 'live fin' }, h('i'), 'Finished');
    if (st.state === 'crashed') return h('span', { class: 'live bad', title: detail }, h('i'), 'Crashed');
    return h('span', { class: 'live idle', title: detail }, h('i'), 'Stopped');
  }
  function stat(v, label, title) { return h('span', { title: title || null }, h('b', { text: v }), label); }
  function headerStats() {
    const c = S.static.config, r = S.run, out = [stateEl()];
    if (!r) {
      out.push(stat('–', 'best'), stat(`0/${c.max_generations}`, 'generations'));
      if (num(c.max_evaluations)) out.push(stat(`0/${compact(c.max_evaluations)}`, 'evaluations'));
      out.push(stat('–', 'spend'));
      return out;
    }
    const ob = frontierType() !== 'instance' ? (r.objective_bests || []).filter((o) => o && o.name != null) : [];
    if (ob.length) {
      // per-objective bests in place of the single mean; all of them, with holders and the seed, in the popover
      out.push(h('button', { class: 'objs', id: 'objs', type: 'button', 'aria-expanded': String($('opop').classList.contains('on')), 'aria-controls': 'opop' },
        ob.slice(0, 3).map((o) => h('span', { title: o.name }, h('b', { text: ofmt(o.mean) }), h('span', { class: 'on', text: o.name }))),
        ob.length > 3 ? h('span', { class: 'more', text: `+${ob.length - 3}` }) : null));
    } else {
      const sd = r.seed, seed = sd && num(sd.mean) && !(r.best && r.best.id === sd.id) ? score(sd) : null;
      out.push(h('span', { title: r.best && r.best.id ? r.best.id + (seed ? ' · seed ' + sd.id : '') : null }, h('b', { text: score(r.best) }), 'best',
        seed ? h('span', { class: 'sd' }, ' · seed ', h('b', { text: seed })) : null));
    }
    const g = r.generation || {};
    out.push(stat(`${num(g.current) ? g.current : '–'}${num(g.max) ? '/' + g.max : ''}`, 'generations'));
    const ev = r.evaluations || {};
    out.push(stat(num(ev.max) ? `${compact(ev.used)}/${compact(ev.max)}` : compact(ev.used), 'evaluations'));
    if (num(r.time_left_seconds) && running()) out.push(stat('~' + dur(r.time_left_seconds), 'left'));
    else if (num(r.elapsed_seconds)) out.push(stat(dur(r.elapsed_seconds + (running() ? nowSec() - S.recvAt : 0)), 'elapsed'));
    const sp = r.spend || {};
    const tokens = (sp.input_tokens || 0) + (sp.output_tokens || 0);
    if (sp.has_cost) out.push(stat(money(sp.cost_usd) + (num(sp.projected_cost_usd) ? ` → ~${money(sp.projected_cost_usd)}` : ''), 'spend'));
    else if (tokens > 0) out.push(stat(`${compact(sp.input_tokens)} in · ${compact(sp.output_tokens)} out`, 'tokens'));
    else out.push(stat('–', 'spend'));
    return out;
  }
  function renderHeader() {
    const p = S.static.project;
    $('name').textContent = p.name;
    document.title = `${p.name} · HELIX`;
    if (S.error) { fill($('stats'), h('span', { class: 'note-err', text: S.error })); return; }
    fill($('stats'), headerStats());
    const repro = JSON.stringify(S.static.repro || []);
    if (repro !== S.popKey) { S.popKey = repro; fill($('pop'), h('dl', null, (S.static.repro || []).map(([k, v]) => [h('dt', { text: k }), h('dd', { text: v, title: v })]))); }
  }
  function objPop(on, pin) {
    const pop = $('opop'), st = S.opop, btn = $('objs');
    clearTimeout(st.timer);
    if (pin != null) st.pin = pin;
    on = on && !!btn;
    if (on) {
      const r = S.run, sm = (r && r.seed && r.seed.objective_means) || {};
      fill(pop, h('table', { class: 'ob' }, h('thead', null, h('tr', null, h('th', { text: 'objective' }), h('th', { text: 'best' }), h('th', { text: 'candidate' }), h('th', { text: 'seed' }))),
        h('tbody', null, ((r && r.objective_bests) || []).map((o) => h('tr', null, h('td', { text: o.name }), h('td', { class: 'n', text: ofmt(o.mean) }),
          h('td', null, o.id ? h('button', { class: 'lnk mono', type: 'button', 'data-open': o.id, text: o.id }) : '–'), h('td', { class: 'n', text: ofmt(sm[o.name]) }))))));
    }
    pop.classList.toggle('on', on);
    if (on) pop.style.left = Math.max(8, Math.min(btn.offsetLeft, pop.parentElement.clientWidth - pop.offsetWidth - 8)) + 'px';
    if (!on) st.pin = false;
    if (btn) btn.setAttribute('aria-expanded', String(on));
  }
  function togglePop(on) {
    const pop = $('pop'), name = $('name');
    on = on ?? !pop.classList.contains('on');
    pop.style.left = name.offsetLeft + 'px';
    pop.classList.toggle('on', on);
    name.setAttribute('aria-expanded', String(on));
  }

  // ---------------- progress: a helix around the best score ----------------
  function chartDomain() {
    const r = S.run, c = S.static.config, pts = (r && r.progress && r.progress.points) || [];
    const cur = (r && r.generation && num(r.generation.current)) ? r.generation.current : 0;
    const G = Math.max(1, c.max_generations || 1, cur, ...pts.map((p) => p.generation || 0));
    return { G, cur };
  }
  function ticks(G) {
    if (G <= 20) return [...Array(G + 1).keys()];
    const step = [5, 10, 25, 50, 100, 250, 500, 1000, 2500].find((t) => G / t <= 8) || Math.ceil(G / 8);
    const out = [];
    for (let g = 0; g <= G; g += step) out.push(g);
    return out;
  }
  function jitter(id) { let x = 0; for (const ch of id) x = (x * 31 + ch.charCodeAt(0)) >>> 0; return (x % 1000) / 1000 * 0.8 - 0.4; }
  function renderChart() {
    syncObjSel();
    const S0 = $('pc'), W = S0.clientWidth || 800, Hh = 112, pl = 40, pt = 12, pb = 20, amp = 6, per = 56;
    const pr = $('psel').hidden ? 8 : $('psel').offsetWidth + 16;
    S0.replaceChildren();
    S.chart = null;
    const { G, cur } = chartDomain(), r = S.run;
    const src = progressSource(), obj = src !== (r && r.progress);
    const points = (src && src.points) || [];
    const steps = ((src && src.steps) || []).filter((p) => num(p.mean)).sort((a, b) => a.generation - b.generation);
    const ms = [...points, ...steps].map((p) => p.mean).filter(num);
    let lo = ms.length ? Math.min(...ms) : 0, hi = ms.length ? Math.max(...ms) : 1;
    if (hi - lo < 0.02) { lo -= 0.05; hi += 0.05; }
    const x = (g) => pl + g / G * (W - pl - pr);
    const y = (v) => pt + amp + (1 - (v - lo) / (hi - lo)) * (Hh - pt - pb - 2 * amp);
    const C = { cyan: css('--cyan'), mag: css('--mag'), paper: css('--paper'), ink: css('--ink'), ink2: css('--ink2'), ink3: css('--ink3'), line: css('--line'), line2: css('--line2') };
    S0.appendChild(s('line', { x1: pl, x2: W - pr, y1: Hh - pb, y2: Hh - pb, stroke: C.line }));
    for (const g of ticks(G)) S0.appendChild(s('text', { x: x(g), y: Hh - 4, class: 'ax', 'text-anchor': 'middle', text: g }));
    if (ms.length) {
      const yt = hi - lo > 0.06 ? [lo, (lo + hi) / 2, hi] : [(lo + hi) / 2];
      for (const v of yt) {
        S0.appendChild(s('text', { x: pl - 6, y: y(v) + 4, class: 'ax', 'text-anchor': 'end', text: obj ? ofmt(v) : binary() ? Math.round(v * 100) + '%' : v.toFixed(2) }));
        S0.appendChild(s('line', { x1: pl, x2: W - pr, y1: y(v), y2: y(v), stroke: C.line2 }));
      }
    }
    const big = G > 20;
    for (const p of points) {
      if (!num(p.mean)) continue;
      S0.appendChild(s('circle', { cx: x(p.generation + (p.generation ? jitter(p.id) : 0)), cy: y(p.mean), r: big ? 2 : 3.5, fill: C.ink3, opacity: 0.3, 'data-c': p.id, class: 'cd' }));
    }
    if (!steps.length) return;
    const xc = x(Math.max(cur, steps[steps.length - 1].generation)), x0 = x(steps[0].generation);
    const ramp = Math.max(5, Math.min(14, (x(1) - x(0)) * 1.5));
    const centre = (X) => {
      let v = steps[0].mean;
      for (const c of steps) {
        const xs = x(c.generation);
        if (X >= xs) v = c.mean;
        else { const t = (X - (xs - ramp)) / ramp; if (t > 0) v += (c.mean - v) * (t * t * (3 - 2 * t)); break; }
      }
      return y(v);
    };
    // the strands are drawing only: clicks reach the dots under them, and the background opens the lineage
    const back = S0.appendChild(s('path', { fill: 'none', stroke: C.cyan, 'stroke-width': 2.2, 'stroke-linecap': 'round', 'pointer-events': 'none' }));
    const halo = S0.appendChild(s('path', { fill: 'none', stroke: C.paper, 'stroke-width': 5, 'stroke-linecap': 'round', 'pointer-events': 'none' }));
    const front = S0.appendChild(s('path', { fill: 'none', stroke: C.mag, 'stroke-width': 2.2, 'stroke-linecap': 'round', 'pointer-events': 'none' }));
    const best = steps[steps.length - 1].mean;
    if (x(G) - xc > 1) S0.appendChild(s('path', { d: `M${xc} ${y(best)}H${x(G)}`, fill: 'none', stroke: C.ink3, 'stroke-width': 1.2, 'stroke-dasharray': '2 4', opacity: 0.6 }));
    const rungs = steps.slice(1).map((c) => [S0.appendChild(s('line', { stroke: C.ink2, 'stroke-width': 2, 'stroke-linecap': 'round', 'data-c': c.id, class: 'cd' })), c]);
    const head = S0.appendChild(s('circle', { r: 3.5, fill: C.ink, 'pointer-events': 'none' }));
    function draw() {
      let A = '', B = '';
      for (let X = x0; X <= xc + 0.01; X += 2) {
        const c = centre(X), f = Math.sin((X - pl) / per * 2 * Math.PI + S.phase);
        A += (A ? 'L' : 'M') + X.toFixed(1) + ' ' + (c + amp * f).toFixed(2);
        B += (B ? 'L' : 'M') + X.toFixed(1) + ' ' + (c - amp * f).toFixed(2);
      }
      back.setAttribute('d', A); halo.setAttribute('d', B); front.setAttribute('d', B);
      for (const [ln, c] of rungs) {
        const X = x(c.generation), f = Math.sin((X - pl) / per * 2 * Math.PI + S.phase), cy = centre(X), a = amp * Math.max(0.4, Math.abs(f));
        ln.setAttribute('x1', X); ln.setAttribute('x2', X); ln.setAttribute('y1', cy - a); ln.setAttribute('y2', cy + a);
      }
      head.setAttribute('cx', xc); head.setAttribute('cy', centre(xc));
    }
    draw();
    // crosshair: generation, best so far, and the candidate holding it
    const cross = S0.appendChild(s('line', { y1: pt, y2: Hh - pb, stroke: C.ink3, 'stroke-width': 1, opacity: 0, 'pointer-events': 'none' }));
    S.chart = { draw, cross, x, pl, pr, W, G, xc, steps, obj: obj ? src.name : null };
  }
  // on runs with objectives the centre line can follow one objective's best so far instead of the aggregate
  function progressSource() {
    const r = S.run, objs = r && r.progress && r.progress.objectives;
    const o = S.pobj != null && objs ? objs.find((x) => x.name === S.pobj) : null;
    return o || (r && r.progress) || null;
  }
  function syncObjSel() {
    const sel = $('psel'), r = S.run, objs = (r && r.progress && r.progress.objectives) || [];
    const names = frontierType() !== 'instance' ? objs.map((o) => o.name) : [];
    const key = JSON.stringify(names);
    if (key !== S.pobjKey) {
      S.pobjKey = key;
      if (S.pobj == null) S.pobj = store.get('helix-ui-progress');
      if (!names.includes(S.pobj)) S.pobj = null;
      fill(sel, h('option', { value: '', text: 'aggregate' }), names.map((n) => h('option', { value: n, text: n })));
      sel.hidden = !names.length;
    }
    sel.value = S.pobj ?? '';
  }
  function chartMove(ev) {
    const ch = S.chart, xh = $('xh');
    if (!ch) return;
    const X = ev.clientX - $('pc').getBoundingClientRect().left;
    if (X < ch.pl || X > ch.xc + 2) { ch.cross.setAttribute('opacity', 0); xh.style.display = 'none'; return; }
    const g = Math.max(0, Math.round((X - ch.pl) / (ch.W - ch.pl - ch.pr) * ch.G));
    const hold = [...ch.steps].reverse().find((c) => c.generation <= g) || ch.steps[0];
    const holder = S.byId[hold.id] || hold;
    ch.cross.setAttribute('x1', ch.x(g)); ch.cross.setAttribute('x2', ch.x(g)); ch.cross.setAttribute('opacity', 1);
    xh.style.display = 'block';
    if (ch.obj) fill(xh, 'generation ', h('b', { text: g }), ' · best ', h('b', { text: ofmt(hold.mean) }), ` ${ch.obj} · ${hold.id}`);
    else fill(xh, 'generation ', h('b', { text: g }), ' · best ', h('b', { text: score(holder) }), ' · ' + hold.id);
    // the readout sits right of the crosshair, or left of it when it would run into the selector or the edge
    const X0 = 28 + ch.x(g), wx = xh.offsetWidth;
    xh.style.left = (X0 + 8 + wx <= 28 + ch.W - ch.pr ? X0 + 8 : Math.max(28, X0 - 8 - wx)) + 'px';
  }
  function chartLeave() { if (S.chart) S.chart.cross.setAttribute('opacity', 0); $('xh').style.display = 'none'; }
  function frame() {
    if (S.chart && animate() && document.visibilityState === 'visible') { S.phase -= 0.012; S.chart.draw(); }
    requestAnimationFrame(frame);
  }

  // ---------------- the loop, with live numbers ----------------
  const speed = (sec) => Math.min(12, Math.max(0.6, 0.35 * Math.sqrt(sec)));
  function warnings() {
    const w = [...((S.static.isolation && S.static.isolation.warnings) || [])];
    if (S.run && S.run.config_changed && !w.some((x) => x.id === 'config-changed')) {
      w.push({ id: 'config-changed', text: 'config changed since the run started', detail: 'The configuration differs from the one the run recorded.', keys: [] });
    }
    return w;
  }
  function mapModel() {
    const st = S.static, c = st.config, r = S.run, iso = st.isolation || {}, L = (r && r.loop) || null;
    const planned = (st.loop && st.loop.planned) || {};
    const type = frontierType();
    const nobj = (S.frontier && S.frontier.objectives && S.frontier.objectives.length) || (r && r.objective_bests && r.objective_bests.length) || 0;
    const typeSub = type === 'hybrid' ? `instances + ${nobj || '?'} objectives` : type === 'objective' ? `${nobj || '?'} objectives` : type === 'cartesian' ? `instances × ${nobj || '?'} objectives` : null;
    const m = {
      run: !!r, anim: animate(), agent: iso.agent || (c.sandbox && c.sandbox.enabled ? 'sandbox' : 'host'), evaluator: iso.evaluator || 'host',
      merge: !!c.merge_enabled, warn: warnings().map((x) => [x.id, x.text, x.detail || '']), dark: dark.matches,
      front: S.front.map((x) => [x.id, x.mean]),
    };
    if (!L) {
      m.f = ['–', '', typeSub];
      m.m = ['–', '', planned.mutation || null, false];
      m.g = ['–', '', planned.gate || null];
      m.v = ['–', '', planned.validation || null, false];
      m.mg = ['–', '', null];
      m.secs = [900, 120, 300];
      return m;
    }
    const mu = L.mutation || {}, gt = L.gate || {}, va = L.validation || {}, fr = L.frontier || {};
    m.f = [String(num(fr.size) ? fr.size : S.front.length), `of ${num(fr.accepted) ? fr.accepted : (r.candidates || []).length} accepted`, typeSub];
    m.m = [String(mu.running ?? 0), 'running', num(mu.median_seconds) ? `median ${dur(mu.median_seconds)}` : (mu.finished ? `${mu.finished} finished` : 'none finished yet'), (mu.running || 0) > 0];
    m.g = gt.tried ? [pct(gt.passed / gt.tried), 'pass', `${gt.passed} of ${gt.tried} · ${gt.ids_each ?? c.minibatch_size} ids each`] : ['–', '', `${gt.ids_each ?? c.minibatch_size} ids each`];
    m.v = [num(va.median_seconds) ? dur(va.median_seconds) : '–', num(va.median_seconds) ? 'median' : '', `${va.count ?? 0} validated · ${va.ids ?? '–'} ids`, (va.running || 0) > 0];
    const merges = (r.candidates || []).filter((x) => x.operation === 'merge').length;
    m.mg = [String(merges), 'accepted', null];
    const split = Object.fromEntries(((r.health && r.health.time_split) || []).map((t) => [t.kind, t.seconds]));
    m.secs = [mu.median_seconds || 900, gt.tried && split.gate ? split.gate / gt.tried : 120, va.median_seconds || 300];
    return m;
  }
  function renderMap() {
    const m = mapModel(), key = JSON.stringify(m);
    if (key === S.mapKey) return;
    S.mapKey = key;
    const O = $('map'), C = css('--cyan'), M = css('--mag'), P = css('--pur'), A = css('--amber'), I3 = css('--ink3');
    O.replaceChildren();
    const add = (e) => O.appendChild(e);
    const node = (x, y, w, hh, cls, go, title, [val, unit, sub, busy]) => {
      const g = add(s('g', { class: `nd ${cls}${busy && m.anim ? ' busy' : ''}`, transform: `translate(${x},${y})`, 'data-go': go, tabindex: go ? 0 : null, role: go ? 'button' : null, 'aria-label': title }));
      g.appendChild(s('rect', { width: w, height: hh, rx: 10 }));
      g.appendChild(s('text', { x: 16, y: 26, class: 't', text: title }));
      const v = g.appendChild(s('text', { x: 16, y: 64, class: 'v', text: val }));
      if (unit) v.appendChild(s('tspan', { class: 'u', dx: 6, text: unit }));
      if (sub) g.appendChild(s('text', { x: 16, y: 88, class: 's', text: sub }));
      return g;
    };
    const edge = (d, secs, kind, color) => {
      if (kind === 'data') {
        add(s('path', { d, class: 'ed', stroke: color }));
        if (!m.anim) return;
        const du = speed(secs);
        for (let i = 0; i < 2; i++) {
          const c = add(s('circle', { r: 2.8, fill: color }));
          c.appendChild(s('animateMotion', { dur: du + 's', begin: -i * du / 2 + 's', repeatCount: 'indefinite', path: d }));
        }
      } else add(s('path', { d, class: 'ed dec' + (m.anim ? '' : ' still'), stroke: color, style: { '--d': speed(secs) * 1.6 + 's' } }));
    };
    const sandboxAll = m.agent === 'sandbox' && m.evaluator === 'sandbox';
    let box = null;
    if (m.agent === 'sandbox') {
      box = sandboxAll ? { x: 262, y: 70, w: 582, h: 332 } : { x: 262, y: 70, w: 250, h: 150 };
      add(s('rect', { x: box.x, y: box.y, width: box.w, height: box.h, rx: 16, class: 'sb' }));
      add(s('text', { x: 276, y: 62, class: 'zl', fill: M, text: 'sandbox' }));
    }
    if (!sandboxAll) add(s('text', { x: 592, y: 62, class: 'zl', fill: I3, text: m.evaluator === 'sidecar' ? 'sidecar' : 'host' }));
    node(20, 70, 190, m.merge ? 250 : 332, 'f', 'frontier', 'Frontier', m.f);
    node(282, 90, 210, 110, 'm', 'mutation', 'Mutation', m.m);
    node(592, 90, 232, 110, 'e', 'proposals/rejected', 'Minibatch gate', m.g);
    node(592, 272, 232, 110, 'e', 'validation', 'Validation', m.v);
    // frontier members as dots inside the frontier node
    const n = m.front.length;
    if (n) {
      const bw = 150, bh = m.merge ? 130 : 190, cyc = m.merge ? 225 : 275, sp = Math.min(24, Math.sqrt(bw * bh / n)), cols = Math.max(1, Math.min(n, Math.floor(bw / sp))), rows = Math.ceil(n / cols), rr = Math.min(8, sp * 0.36);
      const means = m.front.map((f) => f[1]).filter(num), lo = Math.min(...means), hi = Math.max(...means);
      m.front.forEach(([id, mean], i) => add(s('circle', {
        cx: 115 - (Math.min(n, cols) - 1) * sp / 2 + (i % cols) * sp, cy: cyc - (rows - 1) * sp / 2 + Math.floor(i / cols) * sp, r: rr, fill: C,
        opacity: hi > lo && num(mean) ? 0.3 + 0.7 * (mean - lo) / (hi - lo) : 1, 'data-c': id, class: 'cd',
      })));
    }
    const [aSec, gSec, vSec] = m.secs;
    edge('M210 145 H282', aSec, 'dec', C);
    edge('M492 145 H592', aSec, 'data', M);
    edge('M708 200 V272', gSec, 'dec', P);
    const back = m.merge ? 300 : 327;
    edge(`M592 ${back} H210`, vSec, 'data', P);
    if (m.merge) {
      node(20, 342, 190, 100, 'm', null, 'Merge', m.mg);
      edge('M115 320 V342', vSec, 'dec', C);
      edge('M210 392 H708 V382', vSec, 'data', P);
    }
    if (box) {
      add(s('circle', { cx: 262, cy: 145, r: 4, class: 'port' }));
      if (sandboxAll) { add(s('circle', { cx: 262, cy: back, r: 4, class: 'port' })); if (m.merge) add(s('circle', { cx: 262, cy: 392, r: 4, class: 'port' })); }
      else add(s('circle', { cx: 512, cy: 145, r: 4, class: 'port' }));
    }
    // amber only for likely mistakes, on the boundary they concern
    const top = box ? box.y : 90, bottom = box ? box.y + box.h : 200;
    const anchors = {
      'web-access-open-network': [387, top, -1], 'omit-without-sandbox': [387, top, -1], 'shared-auth-volume': [387, bottom, 1],
      'minibatch-repeats': [708, 90, -1], 'config-changed': [115, 70, -1],
    };
    const used = {};
    for (const [id, text, detail] of m.warn) {
      const [ax, ay, dir] = anchors[id] || [115, 70, -1], k = `${ax},${ay}`, nth = used[k] = (used[k] ?? -1) + 1;
      const ly = dir < 0 ? ay - 30 - nth * 18 : ay + 32 + nth * 18;
      add(s('path', { d: `M${ax} ${ay + dir * 5} V${ly + (dir < 0 ? 6 : -12)}`, stroke: A, 'stroke-dasharray': '3 3', fill: 'none' }));
      const p = add(s('circle', { cx: ax, cy: ay, r: 4.5, class: 'port w' }));
      const l = add(s('text', { x: ax, y: ly, class: 'wl', 'text-anchor': 'middle', text }));
      for (const e of [p, l]) e.appendChild(s('title', { text: detail || text }));
    }
  }

  // ---------------- run health ----------------
  function renderBand() {
    const r = S.run, c = S.static.config, hl = (r && r.health) || null;
    const sections = [];
    // momentum
    const steps = (r && r.progress && r.progress.steps) || [];
    const since = hl && num(hl.since_best) ? hl.since_best : null;
    const spark = s('svg', { class: 'spark' });
    sections.push(h('section', null, h('h4', { text: 'Momentum' }),
      steps.length > 1 && since != null
        ? h('div', { class: 'big' }, String(since), h('small', { text: plural(since, 'generation', 'generations') + ' since a new best' }))
        : h('div', { class: 'big' }, '–', h('small', { text: r ? 'no new best yet' : 'not started' })),
      h('div', { class: 'sub', text: hl && num(hl.pass_all) ? `gate pass ${pct(hl.pass_recent ?? hl.pass_all)} over the last 30 · ${pct(hl.pass_all)} overall` : 'gate has not run yet' }),
      spark));
    // where the time goes
    const colours = { mutation: ['--mag', 1], gate: ['--pur', 0.55], validation: ['--pur', 1], other: ['--ink3', 1] };
    const split = ((hl && hl.time_split) || []).filter((t) => num(t.seconds) && t.seconds > 0);
    const tot = split.reduce((a, t) => a + t.seconds, 0);
    sections.push(h('section', null, h('h4', { text: 'Where the time goes' }),
      h('div', { class: 'tb' }, split.map((t) => h('i', { style: { flex: String(t.seconds), background: `var(${(colours[t.kind] || colours.other)[0]})`, opacity: String((colours[t.kind] || colours.other)[1]) } }))),
      tot ? h('div', { class: 'tl' }, split.map((t) => [h('b', { text: Math.round(100 * t.seconds / tot) + '%' }), h('span', { text: `${t.kind} · ${dur(t.seconds)}` })]))
        : h('div', { class: 'sub', text: '–' })));
    // spend, or budget used when there is no spend to show; with usage records, the share on rejected work and a bar per generation
    const sp = (r && r.spend) || {}, tokens = (sp.input_tokens || 0) + (sp.output_tokens || 0);
    const gens = sp.per_record && Array.isArray(sp.by_generation) ? sp.by_generation.filter((g) => num(g.generation) && uVal(g, sp.has_cost) > 0) : [];
    const share = sp.per_record ? wasteShare(sp.by_outcome, sp.has_cost) : null;
    const shareRow = share ? [[pct(share.v), 'on rejected', share.title]] : [];
    const bars = gens.length > 1 ? s('svg', { class: 'spk' }) : null;
    let title = 'Spend', rows;
    if (sp.has_cost) {
      // a projection only while the run is going; after it, the cost of a generation
      rows = [[money(sp.cost_usd), running() ? 'so far' : 'in all', num(sp.cost_per_accepted) ? `${money(sp.cost_per_accepted)} per accepted` : null],
        running() ? [num(sp.projected_cost_usd) ? '~' + money(sp.projected_cost_usd) : '–', 'at finish', num(sp.cost_per_generation) ? `${money(sp.cost_per_generation)} per generation` : null]
          : [money(sp.cost_per_generation), 'per generation']];
      rows.push(...(shareRow.length ? shareRow : [[money(sp.cost_per_accepted), 'per accepted']]));
    } else if (tokens > 0) {
      rows = [[compact(sp.input_tokens), 'tokens in', num(sp.cached_input_tokens) ? `${compact(sp.cached_input_tokens)} cached` : null], [compact(sp.output_tokens), 'tokens out']];
      rows.push(...(shareRow.length ? shareRow : [[compact(sp.cached_input_tokens), 'cached in']]));
    } else {
      title = 'Budget used';
      const ev = (r && r.evaluations) || {}, g = (r && r.generation) || {};
      rows = [[`${num(g.current) ? g.current : 0}${num(g.max) ? '/' + g.max : ''}`, 'generations'],
        [num(ev.max ?? c.max_evaluations) ? `${compact(ev.used || 0)}/${compact(ev.max ?? c.max_evaluations)}` : compact(ev.used || 0), 'evaluations'],
        [r ? dur(r.elapsed_seconds) : '–', 'elapsed']];
    }
    sections.push(h('section', null, h('h4', { text: title }),
      h('div', { class: 'kv' }, rows.map(([v, l, t]) => [h('b', { text: v, title: t || null }), h('span', { text: l, title: t || null })])), bars));
    // failures
    const f = (hl && hl.failures) || {}, types = Object.entries(f.evaluator_error_types || {});
    const go = r ? 'proposals/failed' : null;
    const frow = (v, l, t) => [h('b', { class: num(v) && v > 0 ? 'bad' : null, text: r ? String(v ?? 0) : '–', title: t || null, 'data-go': go, role: go && 'button', tabindex: go && 0 }),
      h('span', { text: l, title: t || null, 'data-go': go })];
    sections.push(h('section', null, h('h4', { text: 'Failures' }), h('div', { class: 'kv' + (go ? ' go' : '') },
      frow(f.agent_timeouts, 'agent timeouts'), frow(f.agent_errors, 'agent errors'),
      frow(f.evaluator_errors, 'evaluator errors', types.map(([k, v]) => `${k} × ${v}`).join(', ')))));
    fill($('band'), sections);
    const pass = ((hl && hl.pass_rates) || []).filter(num);
    if (pass.length > 2) {
      const W = spark.clientWidth || 200, Hh = 30, n = pass.length, k = 8;
      const sm = pass.map((_, i) => avg(pass.slice(Math.max(0, i - k), i + 1)));
      const x = (i) => i / (n - 1) * W, y = (v) => Hh - 2 - v * (Hh - 6);
      spark.appendChild(s('rect', { x: x(Math.max(0, n - 30)), y: 0, width: W - x(Math.max(0, n - 30)), height: Hh, fill: css('--ink'), opacity: 0.04 }));
      spark.appendChild(s('path', { d: sm.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1)).join(''), fill: 'none', stroke: css('--pur'), 'stroke-width': 1.5 }));
    }
    if (bars) spendBars(bars, gens, sp.has_cost);
  }
  // usage: dollars when there is a cost, else tokens (Codex reports cost 0 with tokens)
  const uVal = (u, cost) => (!u ? 0 : cost ? (num(u.cost_usd) ? u.cost_usd : 0) : (u.input_tokens || 0) + (u.output_tokens || 0));
  const uText = (u, cost) => (cost ? money(u.cost_usd) : `${compact(u.input_tokens)} in · ${compact(u.output_tokens)} out`);
  function wasteShare(bo, cost) {
    if (!bo) return null;
    const v = (k) => uVal(bo[k], cost), all = v('accepted') + v('rejected') + v('failed') + v('running');
    if (!(all > 0) || !(all - v('running') > 0)) return null;  // nothing finished yet: no share to show
    const rej = v('rejected') / all, fail = v('failed') / all;
    return { v: rej + fail, title: `rejected ${pct(rej)} · failed ${pct(fail)}` };
  }
  // one bar per generation with usage records; a generation with none is an empty slot
  function spendBars(el, gens, cost) {
    const W = el.clientWidth || 160, Hh = 22, g0 = gens[0].generation;
    const cur = S.run && S.run.generation && num(S.run.generation.current) ? S.run.generation.current : 0;
    const g1 = Math.max(cur, gens[gens.length - 1].generation), n = g1 - g0 + 1, slot = Math.min(8, W / n);
    const top = Math.max(...gens.map((g) => uVal(g, cost)));
    const bw = slot >= 3 ? slot - 1 : slot, fillc = css('--mag');
    for (const g of gens) {
      const v = uVal(g, cost), bh = Math.max(1, v / top * Hh);
      el.appendChild(s('rect', { x: ((g.generation - g0) * slot).toFixed(2), y: (Hh - bh).toFixed(2), width: Math.max(0.6, bw).toFixed(2), height: bh.toFixed(2), fill: fillc, opacity: 0.55 },
        s('title', { text: `generation ${g.generation} · ${uText(g, cost)}` })));
    }
  }

  // ---------------- now: lanes and the transcript ----------------
  function resultText(res) {
    const sc = num(res.successes) && num(res.n) ? `${res.successes}/${res.n}` : num(res.score) ? res.score.toFixed(2) : '';
    if (res.status === 'accepted') return ['accepted ' + sc, 'ac'];
    if (res.status === 'failed') return ['failed' + (res.reason ? ' · ' + String(res.reason).replace(/_/g, ' ') : ''), 'no'];
    const why = String(res.reason || res.stage || '');
    const where = /gate|minibatch/.test(why) ? 'rejected at gate' : /val/.test(why) ? 'rejected at validation' : why ? 'rejected · ' + why.replace(/_/g, ' ') : 'rejected';
    return [where + (sc ? ' · ' + sc : ''), 'no'];
  }
  function laneStatus(l, t) {
    const since = num(l.phase_started_at) ? t - l.phase_started_at : null;
    switch (l.phase) {
      case 'agent': return [clock(since), 'go'];
      case 'parent_eval': return ['parent eval ' + clock(since), 'ev'];
      case 'child_eval': return ['gate ' + clock(since), 'ev'];
      case 'waiting': return ['waiting', 'no'];
      case 'validation': return ['validating', 'ev'];
      default: return l.result ? resultText(l.result) : ['done', 'no'];
    }
  }
  // set text only when it differs, so a selection inside it survives the 1 s tick
  function setText(e, v) { if (e.textContent !== v) e.textContent = v; }
  function renderRail() {
    const r = S.run, now = r && r.now, t = running() ? nowSec() : (S.recvAt || nowSec());
    if (!now) {
      setText($('gt'), r ? 'Generation –' : 'Not started');
      setText($('gs'), '');
    } else {
      setText($('gt'), `Generation ${now.generation ?? '–'}`);
      const el = num(now.started_at) && running() ? dur(t - now.started_at) : (r.status && r.status.state) || '';
      setText($('gs'), el + (num(now.typical_seconds) && running() ? ` · usually ${dur(now.typical_seconds)}` : ''));
    }
    const lanes = (now && now.lanes) || [];
    if (!lanes.length) {
      if (S.laneKey !== '') { S.laneKey = ''; fill($('lanes'), h('div', { class: 'empty', text: 'No agent running' })); }
      $('txh').style.display = 'none'; $('tx').style.display = 'none';
      stopTx();
      return;
    }
    $('txh').style.display = ''; $('tx').style.display = '';
    if (!S.lane || !lanes.some((l) => l.id === S.lane)) S.lane = (lanes.find((l) => l.phase === 'agent') || lanes[0]).id;
    const t0 = num(now.started_at) ? now.started_at : Math.min(...lanes.flatMap((l) => (l.segments || []).map((g) => g.start)).filter(num), t);
    const ends = lanes.flatMap((l) => (l.segments || []).map((g) => (num(g.end) ? g.end : t)));
    const t1 = Math.max(t, t0 + (now.typical_seconds || 0), ...ends);
    const span = Math.max(1, t1 - t0);
    const rows = lanes.map((l) => {
      const [txt, cls] = laneStatus(l, t);
      const segs = (l.segments || []).filter((g) => num(g.start)).map((g) => {
        const open = !num(g.end), e = open ? t : g.end;
        return [(g.kind === 'agent' ? 'a' : 'e') + (open && running() ? ' run' : ''), (100 * (g.start - t0) / span).toFixed(2) + '%', Math.max(0.6, 100 * (e - g.start) / span).toFixed(2) + '%'];
      });
      return { l, txt, cls: 'st ' + cls, segs };
    });
    // rebuild the rows only when the lanes change; the 1 s tick updates times and bars in place, so clicks and selections land
    const key = JSON.stringify([S.lane, rows.map((w) => [w.l.id, w.l.parent, w.segs.length])]), box = $('lanes');
    if (key !== S.laneKey) {
      S.laneKey = key;
      fill(box, rows.map(({ l, txt, cls, segs }) => h('div', { class: 'lane' + (l.id === S.lane ? ' sel' : ''), 'data-lane': l.id },
        h('div', null, h('button', { class: 'lnk', type: 'button', 'data-open': l.id, text: l.id }), h('small', { text: l.parent ? 'from ' + l.parent : '' })),
        h('div', { class: 'trk' }, segs.map(([c, left, width]) => h('i', { class: c, style: { left, width } }))),
        h('div', { class: cls, text: txt, title: txt }))));
    } else {
      rows.forEach(({ txt, cls, segs }, i) => {
        const row = box.children[i], st = row.querySelector('.st'), bars = row.querySelector('.trk').children;
        setText(st, txt);
        if (st.className !== cls) st.className = cls;
        if (st.title !== txt) st.title = txt;
        segs.forEach(([c, left, width], j) => {
          const b = bars[j];
          if (b.className !== c) b.className = c;
          b.style.left = left; b.style.width = width;
        });
      });
    }
    if (!S.tx || S.tx.id !== S.lane) openTx(S.lane);
  }

  function stopTx() { if (S.tx && S.tx.timer) clearTimeout(S.tx.timer); S.tx = null; }
  function openTx(id) {
    stopTx();
    S.tx = { id, cursor: 0, events: [], source: null, note: null, timer: null, t0: null, seen: 0, shown: null };
    renderTx();
    fetchTx(S.tx);
  }
  async function fetchTx(t) {
    if (S.tx !== t) return;
    if (document.visibilityState !== 'visible') { t.timer = setTimeout(() => fetchTx(t), 4000); return; }
    let d;
    try { d = await get(`/api/transcript/${encodeURIComponent(t.id)}?cursor=${t.cursor}`); } catch (e) { d = null; }
    if (S.tx !== t) return;
    if (d) {
      if (t.source && d.source !== t.source && t.cursor) { t.events = []; t.cursor = 0; t.seen = 0; t.source = d.source; t.timer = setTimeout(() => fetchTx(t), 0); return; }
      t.source = d.source; t.note = d.note || null; t.backend = d.backend || null;
      for (const e of d.events || []) { t.events.push(e); t.seen++; }
      if (t.events.length > 600) t.events.splice(0, t.events.length - 600);
      if (num(d.cursor)) t.cursor = d.cursor;
      const key = `${t.source}|${t.note}|${t.seen}`;
      if (key !== t.shown) { t.shown = key; renderTx(); }
    }
    if (!d || d.source === 'live') t.timer = setTimeout(() => fetchTx(t), 4000);
  }
  function renderTx() {
    const t = S.tx, box = $('tx');
    if (!t) return;
    const label = t.source === 'live' ? 'live' : t.source === 'finished' ? 'finished' : t.source === 'none' ? 'no transcript' : '…';
    $('txw').textContent = `${t.id} · ${label}`;
    $('txw').title = t.note || '';
    const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    const first = t.events.find((e) => num(e.t));
    const t0 = first ? first.t : null;
    if (!t.events.length && t.note) { fill(box, h('div', { class: 'ev muted', text: t.note })); return; }
    fill(box, t.events.map((e) => {
      const w = h('div', { class: 'w', text: num(e.t) && t0 != null ? clock(e.t - t0) : '' });
      const k = e.kind || 'message', text = e.text == null ? '' : String(e.text);
      if (k === 'message') return h('div', { class: 'ev' }, w, text);
      const bad = num(e.exit_code) && e.exit_code !== 0;
      const line = k === 'command' ? '$ ' + text + (bad ? `  (exit ${e.exit_code})` : '') : text;
      return h('div', { class: 'ev cmd' }, w, h('code', { class: bad ? 'x' : null, text: line, title: text.length > 80 ? text.slice(0, 2000) : null }));
    }));
    if (atEnd) box.scrollTop = box.scrollHeight;
  }

  // ---------------- the hover card ----------------
  let hoverEl = null;
  function showCard(target) {
    const id = target.getAttribute('data-c'), c = S.byId[id], k = $('card');
    document.querySelectorAll('.hot').forEach((x) => x.classList.remove('hot'));
    document.querySelectorAll(`[data-c="${CSS.escape(id)}"]`).forEach((x) => x.classList.add('hot'));
    if (S.view === 'frontier') strip(id);
    if (!c) {
      const x = otherOf(id);
      if (!x) { k.style.display = 'none'; return; }
      fill(k, attemptCard(x));
      placeCard(k, target);
      return;
    }
    const seed = S.static.config.seedless ? 'seed, generated' : 'seed';
    const rank = S.rank[c.id];
    const held = S.frontier && S.frontier.held && S.frontier.held[c.id];
    let bars = null;
    if (held && S.frontier.instances && S.frontier.instances.length) {
      const N = S.frontier.instances.length, B = Math.min(120, N), mark = new Uint8Array(N);
      for (const i of held) if (i >= 0 && i < N) mark[i] = 1;
      bars = h('div', { class: 'bars' }, [...Array(B).keys()].map((b) => {
        const a = Math.floor(b * N / B), z = Math.max(a + 1, Math.floor((b + 1) * N / B));
        let on = 0; for (let i = a; i < z; i++) on += mark[i];
        return h('i', { style: { opacity: String(on ? 0.3 + 0.7 * on / (z - a) : 0.08) } });
      }));
    }
    const times = [num(c.agent_seconds) ? `mutation ${dur(c.agent_seconds)}` : null, num(c.validation_seconds) ? `validation ${dur(c.validation_seconds)}` : null].filter(Boolean);
    fill(k,
      h('div', { class: 'h' }, h('b', { text: c.id }), h('span', { text: c.parent ? 'from ' + c.parent : (c.parents && c.parents.length ? 'merge of ' + c.parents.join(' + ') : seed) })),
      h('div', { class: 'k' }, h('span', null, h('b', { text: score(c) }), binary() ? 'succeeded' : 'mean'),
        c.front || rank ? [h('span', null, h('b', { text: rank ? '#' + rank : '–' }), '/' + S.front.length), h('span', null, h('b', { text: num(c.keys) ? String(c.keys) : '–' }), 'keys')]
          : h('span', { text: 'not on the frontier' })),
      bars,
      h('div', { class: 'note', text: c.note || (c.parent || (c.parents && c.parents.length) ? '–' : S.static.config.seedless ? 'Generated from the objective.' : 'Your repository as given.') }),
      h('div', { class: 't', text: [...times, clickHint()].join(' · ') }));
    placeCard(k, target);
  }
  function placeCard(k, target) {
    const r = target.getBoundingClientRect(), w = 290;
    k.style.display = 'block';
    k.style.left = (r.right + w + 16 < innerWidth ? r.right + 12 : Math.max(8, r.left - w - 4)) + 'px';
    k.style.top = Math.max(8, Math.min(r.top - 30, innerHeight - k.offsetHeight - 8)) + 'px';
  }
  // a proposal that is not an accepted candidate (rejected, skipped, failed, running): its score is a gate score, labelled so
  function otherOf(id) { return S.prop.byId[id] || S.lin.byId[id] || S.kids[id] || null; }
  function attemptCard(x) {
    const pars = x.parents || [], o = outcomeOf(x);
    return [
      h('div', { class: 'h' }, h('b', { text: x.id }), h('span', { text: pars.length > 1 ? 'merge of ' + pars.join(' + ') : pars.length ? 'from ' + pars[0] : '' })),
      h('div', { class: 'k' }, h('span', null, h('b', { text: propScore(x) }), basisWord(x)), h('span', { class: 'o-' + o, text: [o, why(x)].filter(Boolean).join(' · ') })),
      x.summary ? h('div', { class: 'note', text: x.summary }) : null,
      h('div', { class: 't', text: [num(x.agent_seconds) ? 'mutation ' + dur(x.agent_seconds) : null,
        x.usage && uVal(x.usage, false) + uVal(x.usage, true) > 0 ? uText(x.usage, x.usage.cost_usd > 0) : null, ID_RE.test(x.id) ? clickHint() : null].filter(Boolean).join(' · ') || '–' })];
  }
  // in the lineage a click pins the node and shows its panel; a double click opens it
  const clickHint = () => (S.view === 'lineage' ? 'click for details' : 'click to open');
  function hideCard() {
    $('card').style.display = 'none';
    document.querySelectorAll('.hot').forEach((x) => x.classList.remove('hot'));
    if (S.view === 'frontier') strip(null);
  }

  // ---------------- views ----------------
  function crumbs(...parts) {
    return h('div', { class: 'crumb' }, parts.map(([label, view], i) => i < parts.length - 1
      ? [h('button', { class: 'lnk', type: 'button', 'data-go': view, text: label }), ' › ']
      : h('b', { text: label })));
  }
  const propRoute = () => 'proposals' + (S.prop.filter !== 'all' ? '/' + S.prop.filter : '');
  const linRoute = () => 'lineage' + (S.lin.pin ? '/' + S.lin.pin : '');
  function setHash() {
    const want = S.view === 'map' ? '' : S.view === 'candidate' ? '#candidate/' + S.cand : S.view === 'proposals' ? '#' + propRoute()
      : S.view === 'lineage' ? '#' + linRoute() : '#' + S.view;
    if (location.hash !== want) history.replaceState(null, '', location.pathname + location.search + want);
  }
  function go(v) {
    v = String(v || 'map');
    if (v.startsWith('proposals')) {
      const f = v.split('/')[1];
      S.prop.filter = OUTCOMES.includes(f) ? f : 'all';
      v = 'proposals';
    }
    // lineage/<id> pins that node; the plain lineage route opens it unpinned
    if (v.startsWith('lineage')) {
      const id = v.split('/')[1];
      S.lin.pin = id && ID_RE.test(id) ? id : null;
      S.lin.hov = null;
      v = 'lineage';
    }
    if (!VIEWS.includes(v)) v = 'map';
    if (v !== 'candidate') S.cand = null;
    S.view = v;
    for (const n of VIEWS) $('v-' + n).classList.toggle('hide', n !== v);
    hideCard();
    closeInst();
    toggleMenu(false);
    setHash();
    renderView(true);
  }
  function renderView(force) {
    // redraw a view only when its inputs changed, so hover, scroll and selects survive live updates
    const r = S.run;
    const key = S.view === 'lineage' ? ['lineage', lineageKey(), dark.matches, innerWidth, innerHeight].join('|')
      : S.view === 'proposals' ? ['proposals', proposalsKey(), dark.matches].join('|')
        : [S.view, frontierKey(), dark.matches, innerWidth, r ? JSON.stringify([r.loop, r.health && r.health.failures, r.details, (r.candidates || []).length]) : ''].join('|');
    if (!force && key === S.viewKey) return;
    S.viewKey = key;
    if (S.view === 'frontier') drawFrontier();
    else if (S.view === 'mutation') drawMutation();
    else if (S.view === 'validation') drawValidation();
    else if (S.view === 'lineage') drawLineage();
    else if (S.view === 'proposals') drawProposals();
  }

  // the views menu
  function renderMenu() {
    fill($('menu'), MENU.map(([v, label]) => h('button', {
      type: 'button', role: 'menuitem', class: 'mi' + (S.view === v ? ' cur' : ''), 'data-go': v, 'aria-current': S.view === v ? 'page' : null, text: label,
    })));
  }
  function toggleMenu(on, refocus) {
    const m = $('menu'), b = $('vbtn'), was = m.classList.contains('on');
    on = on ?? !was;
    if (on === was) return;
    if (on) {
      renderMenu();
      m.style.right = Math.max(8, b.parentElement.clientWidth - b.offsetLeft - b.offsetWidth) + 'px';
      m.style.top = b.offsetTop + b.offsetHeight + 6 + 'px';
    }
    m.classList.toggle('on', on);
    b.setAttribute('aria-expanded', String(on));
    if (on) (m.querySelector('.cur') || m.firstElementChild).focus();
    else if (refocus) b.focus();
  }

  // frontier
  function frontierKey() {
    const r = S.run;
    return r ? `${(r.frontier && r.frontier.ids || []).join(',')}|${(r.candidates || []).length}` : 'none';
  }
  function loadFrontier() {
    const key = frontierKey();
    if (!S.run) { S.frontier = null; S.frontierKey = key; return Promise.resolve(null); }
    if (S.frontierKey === key) return Promise.resolve(S.frontier);
    if (S.frontierLoading && S.frontierLoading.key === key) return S.frontierLoading.p;
    const p = get('/api/frontier').then((d) => { S.frontier = d; S.frontierKey = key; return d; }).catch(() => S.frontier)
      .finally(() => { if (S.frontierLoading && S.frontierLoading.key === key) S.frontierLoading = null; });
    S.frontierLoading = { key, p };
    return p;
  }
  async function drawFrontier() {
    const V = $('v-frontier');
    if (!V.firstChild) fill(V, crumbs(['Loop', 'map'], ['Frontier', '']));
    const fr = await loadFrontier();
    if (S.view !== 'frontier') return;
    const type = frontierType(), all = S.run ? S.run.candidates || [] : [];
    const stats = h('div', { class: 'fs' },
      h('span', null, h('b', { text: String(S.front.length) }), 'on the frontier'),
      h('span', null, h('b', { text: String(all.length) }), 'accepted'),
      fr && num(fr.below_threshold) ? h('span', { title: `no frontier member reaches ${fr.threshold}` }, h('b', { text: String(fr.below_threshold) }), 'not reached by any frontier member') : null);
    const out = [crumbs(['Loop', 'map'], ['Frontier', '']), stats];
    const objs = (fr && fr.objectives) || [];
    const sc = s('svg', { class: 'sc', id: 'sc' });
    if (type !== 'instance' && objs.length) {
      const prev = [$('ox') && $('ox').value, $('oy') && $('oy').value];
      const sel = (id, d) => {
        const was = id === 'ox' ? prev[0] : prev[1], pick = was != null && +was < objs.length ? +was : d;
        return h('select', { id, 'aria-label': id === 'ox' ? 'x objective' : 'y objective', onchange: () => scatter() },
          objs.map((o, j) => h('option', { value: j, selected: j === pick, text: o })));
      };
      stats.appendChild(h('span', { class: 'sel' }, 'x ', sel('ox', 0), ' y ', sel('oy', Math.min(1, objs.length - 1))));
      const om = fr.objective_means || {};
      const bestCol = objs.map((_, j) => Math.max(...S.front.map((c) => (om[c.id] || [])[j]).filter(num)));
      const table = h('table', { class: 'ot' },
        h('thead', null, h('tr', null, h('th', { text: 'candidate' }), objs.map((o) => h('th', { text: o })), h('th', { text: 'zero-score' }))),
        h('tbody', null, S.front.map((c) => h('tr', { 'data-c': c.id, class: 'cd' }, h('td', { text: c.id }),
          objs.map((_, j) => { const v = (om[c.id] || [])[j]; return h('td', { class: num(v) && v >= bestCol[j] - 1e-9 ? 'best' : null, text: fixed(v, 2) }); }),
          h('td', { class: num(c.zero) && c.zero > 0 ? 'bad' : null, text: num(c.zero) ? String(c.zero) : '–' })))));
      out.push(h('div', { class: 'fgrid' }, sc, h('div', { class: 'otw' }, table)));
    } else out.push(sc);
    if (fr && fr.instances && fr.instances.length) out.push(h('canvas', { class: 'is', id: 'is' }), h('div', { class: 'isl', id: 'isl' }));
    const keep = S.inst && S.inst.where === 'frontier' ? S.inst.iid : null;
    closeInst();
    fill(V, out);
    scatter();
    strip(null);
    if (keep && $('isl')) openInst(keep, $('isl'), 'frontier');
  }
  function scatter() {
    const S0 = $('sc');
    if (!S0) return;
    S0.replaceChildren();
    const fr = S.frontier, W = S0.clientWidth || 600, Hh = 260, pl = 48, pb = 36, pt = 8;
    const type = frontierType(), objs = (fr && fr.objectives) || [];
    let pts, xv, yv, xl, yl, logy = false;
    if (type !== 'instance' && objs.length && $('ox')) {
      const a = +$('ox').value, b = +$('oy').value, allm = fr.all_objective_means || fr.objective_means || {}, fm = fr.objective_means || {};
      const ids = new Set([...Object.keys(allm), ...S.front.map((c) => c.id)]);
      pts = [...ids].map((id) => ({ id, front: S.rank[id] != null, v: fm[id] || allm[id] || [] })).filter((p) => num(p.v[a]) && num(p.v[b]));
      xv = (p) => p.v[a]; yv = (p) => p.v[b]; xl = objs[a]; yl = objs[b];
    } else {
      pts = S.front.filter((c) => num(c.mean)).map((c) => ({ id: c.id, front: true, mean: c.mean, keys: num(c.keys) ? c.keys : 1 }));
      xv = (p) => p.mean; yv = (p) => p.keys; xl = binary() ? 'success rate' : 'mean score'; yl = 'instances held'; logy = true;
    }
    const L = css('--line');
    S0.appendChild(s('line', { x1: pl, x2: W, y1: Hh - pb, y2: Hh - pb, stroke: L }));
    S0.appendChild(s('line', { x1: pl, x2: pl, y1: pt, y2: Hh - pb, stroke: L }));
    const ym = (pt + Hh - pb) / 2;
    S0.appendChild(s('text', { x: (W + pl) / 2, y: Hh - 4, class: 'ax', 'text-anchor': 'middle', text: xl }));
    S0.appendChild(s('text', { x: 12, y: ym, class: 'ax', 'text-anchor': 'middle', transform: `rotate(-90 12 ${ym})`, text: yl }));
    if (!pts.length) return;
    // a collapsed domain (tied members) gets a small pad so the dots sit mid-axis with room around them
    const pad = (lo, hi) => (hi - lo > 1e-6 ? [lo, hi] : [lo - Math.max(0.01, Math.abs(lo) * 0.02), hi + Math.max(0.01, Math.abs(hi) * 0.02)]);
    const xs = pts.map(xv), ys = pts.map(yv), dx0 = Math.min(...xs), dx1 = Math.max(...xs), dy0 = Math.min(...ys), dy1 = Math.max(...ys);
    const [x0, x1] = pad(dx0, dx1), [y0, y1] = logy ? [dy0, dy1] : pad(dy0, dy1);
    const x = (v) => pl + 10 + (v - x0) / (x1 - x0) * (W - pl - 24);
    const y = (v) => (logy
      ? Hh - pb - 8 - Math.log(Math.max(1, v)) / Math.log(Math.max(2, y1)) * (Hh - pb - pt - 16)
      : Hh - pb - 8 - (v - y0) / (y1 - y0) * (Hh - pb - pt - 16));
    const xf = (v) => (logy ? v.toFixed(binary() ? 2 : 3) : v.toFixed(2));
    for (const v of dx1 - dx0 > 1e-6 ? [dx0, dx1] : [dx0]) S0.appendChild(s('text', { x: x(v), y: Hh - pb + 14, class: 'ax', 'text-anchor': 'middle', text: xf(v) }));
    const yticks = [];
    const cand = logy ? [y1, ...[1, 10, 100, 1000, 10000, 100000].filter((v) => v < y1)] : dy1 - dy0 > 1e-6 ? [dy1, dy0] : [dy0];
    for (const v of cand) if (!yticks.some((u) => Math.abs(y(u) - y(v)) < 14)) yticks.push(v);
    for (const v of yticks) S0.appendChild(s('text', { x: pl - 6, y: y(v) + 4, class: 'ax', 'text-anchor': 'end', text: logy ? String(v) : v.toFixed(2) }));
    const nf = S.front.length, r = nf > 60 ? 4.5 : nf > 20 ? 6 : 7, C = css('--cyan'), I3 = css('--ink3');
    for (const p of pts.filter((q) => !q.front)) S0.appendChild(s('circle', { cx: x(xv(p)), cy: y(yv(p)), r: 2.4, fill: I3, opacity: 0.35, 'data-c': p.id, class: 'cd' }));
    // members that land on (nearly) the same spot are laid out in a small centred row, so each can be hovered and opened
    const spots = [];
    for (const p of pts.filter((q) => q.front)) {
      const cx = x(xv(p)), cy = y(yv(p)), g = spots.find((q) => Math.abs(q.cx - cx) < r && Math.abs(q.cy - cy) < r);
      if (g) g.ps.push(p); else spots.push({ cx, cy, ps: [p] });
    }
    const step = 2 * r + 2;
    for (const g of spots) g.ps.forEach((p, i) => {
      const k = g.ps.length, cols = Math.min(k, 8), row = Math.floor(i / cols), n = Math.min(cols, k - row * cols);
      const cx = Math.max(pl + r, Math.min(W - r, g.cx + (i % cols - (n - 1) / 2) * step)), cy = Math.max(pt + r, Math.min(Hh - pb - r, g.cy + row * step));
      S0.appendChild(s('circle', { cx, cy, r, fill: C, opacity: 0.9, 'data-c': p.id, class: 'cd' }));
    });
  }
  function groupsOf(fr) {
    const ids = fr.instances, groups = (fr.groups || []).filter((g) => g !== '');
    if (!groups.length) return { names: [''], of: ids.map(() => '') };
    const sorted = [...groups].sort((a, b) => b.length - a.length);
    const of = ids.map((id) => sorted.find((g) => id.startsWith(g)) ?? '');
    const names = [...groups]; if (of.includes('') && !names.includes('')) names.push('');
    return { names: names.filter((g) => of.includes(g)), of };
  }
  function strip(cid) {
    const cv = $('is'), fr = S.frontier;
    if (!cv || !cv.offsetWidth || !fr || !fr.instances) return;
    const dpr = window.devicePixelRatio || 1, W = cv.offsetWidth, Hh = 20;
    cv.width = W * dpr; cv.height = Hh * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr);
    const ids = fr.instances, best = fr.best || [], th = num(fr.threshold) ? fr.threshold : 1;
    const bv = best.filter(num), blo = Math.min(0, ...bv), bhi = Math.max(th, ...bv);
    const norm = (v) => (bhi > blo ? (v - blo) / (bhi - blo) : 1);
    const { names, of } = groupsOf(fr), gap = names.length > 1 ? 4 : 0;
    const usable = W - names.length * gap, nb = Math.max(1, Math.min(ids.length, Math.floor(usable / 3)));
    const own = cid && fr.held && fr.held[cid] ? new Set(fr.held[cid]) : null;
    const C = { line2: css('--line2'), cyan: css('--cyan'), amber: css('--amber') };
    let X = 0;
    const labs = [], cells = [];
    for (const gn of names) {
      const ix = []; of.forEach((o, i) => { if (o === gn) ix.push(i); });
      const bins = Math.max(1, Math.round(nb * ix.length / ids.length)), bw = usable * ix.length / ids.length / bins;
      labs.push([gn, X]);
      for (let b = 0; b < bins; b++) {
        const part = ix.slice(Math.floor(b * ix.length / bins), Math.floor((b + 1) * ix.length / bins));
        if (!part.length) continue;
        const vals = part.map((i) => best[i]).filter(num), v = vals.length ? avg(vals.map(norm)) : 0;
        cells.push([X, bw, part]);
        g.fillStyle = C.line2; g.fillRect(X, 0, Math.max(0.5, bw - 0.6), Hh);
        if (own) { const o = part.filter((i) => own.has(i)).length / part.length; g.globalAlpha = o ? 0.3 + 0.7 * o : 0.06; g.fillStyle = C.cyan; }
        else if (part.length === 1 && best[part[0]] === 0) { g.globalAlpha = 0.85; g.fillStyle = C.amber; }
        else { g.globalAlpha = 0.1 + 0.9 * v; g.fillStyle = C.cyan; }
        g.fillRect(X, 0, Math.max(0.5, bw - 0.6), Hh);
        g.globalAlpha = 1;
        X += bw;
      }
      X += gap;
    }
    const lab = $('isl');
    let lastEnd = -Infinity;
    fill(lab, labs.filter(([gn]) => gn).map(([gn, x]) => {
      if (x < lastEnd + 8) return null;
      lastEnd = x + gn.length * 6.2;
      return h('span', { style: { left: x + 'px' }, text: gn });
    }));
    cv.onclick = (ev) => {
      const mx = ev.clientX - cv.getBoundingClientRect().left, cell = cells.find(([x, w]) => mx >= x && mx < x + w);
      if (!cell) return;
      // a binned cell holds several instances: the pointer's position inside the cell picks one
      const ix = cell[2], i = ix[Math.min(ix.length - 1, Math.max(0, Math.floor((mx - cell[0]) / cell[1] * ix.length)))];
      openInst(ids[i], $('isl'), 'frontier');
    };
    cv.onmousemove = (ev) => {
      const mx = ev.clientX - cv.getBoundingClientRect().left, cell = cells.find(([x, w]) => mx >= x && mx < x + w);
      if (!cell) { cv.title = ''; return; }
      const ix = cell[2];
      if (ix.length === 1) {
        const i = ix[0], holders = Object.entries(fr.held || {}).filter(([, l]) => l.includes(i)).map(([id]) => id);
        cv.title = `${ids[i]} · best ${fixed(best[i], 2)}${holders.length ? ' · ' + holders.slice(0, 3).join(', ') + (holders.length > 3 ? ` +${holders.length - 3}` : '') : ''}`;
      } else {
        const vals = ix.map((i) => best[i]).filter(num);
        cv.title = `${ix.length} instances · ${ids[ix[0]]} … ${ids[ix[ix.length - 1]]} · mean best ${vals.length ? avg(vals).toFixed(2) : '–'}`;
      }
    };
  }

  // candidate
  // vs: 'seed' asks for the diff against the seed's worktree instead of the parent's (only the switch passes it)
  async function openCand(id, vs) {
    if (!ID_RE.test(id)) return;
    const same = S.view === 'candidate' && S.cand === id;
    if (S.view !== 'candidate') S.from = S.view;
    S.cand = id;
    S.candVs = vs === 'seed' ? 'seed' : 'parent';
    const V = $('v-candidate'), top = V.scrollTop;
    if (!same) fill(V, candCrumbs(id), h('div', { class: 'ch' }, h('h2', { text: id })));
    go('candidate');
    const want = S.candVs;
    let d = null, err = null;
    try { d = await get('/api/candidate/' + encodeURIComponent(id) + (want === 'seed' ? '?vs=seed' : '')); } catch (e) { err = e; }
    if (S.cand !== id || S.candVs !== want) return;
    S.candData = d;
    if (!d) { fill(V, candCrumbs(id), h('div', { class: 'ch' }, h('h2', { text: id }), h('span', { text: err && err.status === 404 ? 'not found' : 'could not load' }))); return; }
    fill(V, candidateView(d));
    if (same) V.scrollTop = top;
  }
  function candCrumbs(id) {
    const via = { frontier: ['Frontier', 'frontier'], mutation: ['Mutation', 'mutation'], validation: ['Validation', 'validation'], lineage: ['Lineage', linRoute()], proposals: ['Proposals', propRoute()] }[S.from];
    return crumbs(['Loop', 'map'], ...(via ? [via] : []), [id, '']);
  }
  function copyBtn(text) {
    return h('button', { class: 'copy', type: 'button', disabled: text ? null : true, onclick: (ev) => {
      const b = ev.currentTarget;
      navigator.clipboard.writeText(text).then(() => { b.textContent = 'copied'; setTimeout(() => { b.textContent = 'copy'; }, 1200); }, () => { b.textContent = 'failed'; });
    }, text: 'copy' });
  }
  function diffView(text) {
    return h('pre', { class: 'diff' }, text.split('\n').map((ln) => {
      const cls = ln.startsWith('+++') || ln.startsWith('---') || ln.startsWith('diff ') || ln.startsWith('index ') ? 'h' : ln.startsWith('@@') ? 'h' : ln.startsWith('+') ? 'a' : ln.startsWith('-') ? 'r' : null;
      return [h('span', { class: cls, text: ln }), '\n'];
    }));
  }
  function candidateView(d) {
    const run = S.byId[d.id] || {}, bin = binary();
    // a candidate still in the current generation has no lineage entry yet: take its parent and generation from the lane
    const lane = ((S.run && S.run.now && S.run.now.lanes) || []).find((l) => l.id === d.id) || null;
    const parent = d.parent || (lane && lane.parent) || null;
    const gen = d.generation ?? (lane && S.run.now.generation) ?? +d.id.slice(1, d.id.indexOf('-'));
    const what = d.note || (d.seed === 'generated' ? 'Generated from the objective.' : d.seed === 'repository' ? 'Your repository as given.' : '–');
    const link = (p) => h('button', { class: 'lnk', type: 'button', 'data-open': p, text: p });
    const sub = d.parent || !(d.parents && d.parents.length) && parent ? ['generation ' + gen + (/-m/.test(d.id) ? ' · merge from ' : ' · from '), link(parent)]
      : d.parents && d.parents.length ? ['generation ' + gen + ' · merge of ', d.parents.map((p, i) => [i ? ' + ' : '', link(p)])]
      : d.seed || d.id.startsWith('g0-') ? (d.seed === 'generated' ? 'seed · generated from the objective' : 'seed · your repository') : 'generation ' + gen;
    const status = d.accepted !== false ? 'not on the frontier' : d.attempt ? 'rejected'
      : lane ? (lane.phase === 'done' && lane.result ? resultText(lane.result)[0] : 'in progress') : 'not on the frontier';
    const om = d.objective_means || run.objective_means || null;
    const zero = num(run.zero) ? run.zero : null;
    const fs = h('div', { class: 'fs' },
      h('span', null, h('b', { text: score(d) }), bin ? 'instances succeeded' : 'mean score'),
      d.front ? [h('span', null, h('b', { text: '#' + (S.rank[d.id] ?? '–') }), `of ${S.front.length} on the frontier`),
        h('span', null, h('b', { text: num(d.keys) ? String(d.keys) : '–' }), 'frontier keys'),
        num(d.sole) ? h('span', null, h('b', { text: String(d.sole) }), 'held alone') : null]
        : h('span', null, h('b', { text: '–' }), status),
      om ? Object.entries(om).map(([k, v]) => h('span', null, h('b', { text: fixed(v, 2) }), k)) : null,
      zero != null ? h('span', null, h('b', { class: zero > 0 ? 'bad' : null, text: String(zero) }), 'zero-score') : null);
    fs.style.marginTop = '8px';
    const vp = d.vs_parent, list = (a, n) => { a = a || []; return a.length ? a.slice(0, 40).join(', ') + (n > 40 ? ` … +${n - 40}` : '') : null; };
    const rows = Array.isArray(d.instances) ? d.instances : null, canJump = !!(rows && d.compare_to && d.instances_basis === 'val');
    // with every instance on the page the counts open the Changed tab in that direction; otherwise their ids are in the title
    const vpBit = (cls, dir, n, text, ids) => (canJump && n
      ? h('button', { class: 'vl ' + cls, type: 'button', onclick: () => ciJump(dir), text })
      : h('span', { class: cls, title: canJump ? null : list(ids, n), text }));
    const vpl = vp && num(vp.improved) && num(vp.regressed) ? h('div', { class: 'vp' },
      vpBit('a', 'up', vp.improved, `+${vp.improved} ${bin ? 'solved' : 'improved'}`, vp.improved_ids), ' · ',
      vpBit('r', 'down', vp.regressed, `−${vp.regressed} regressed`, vp.regressed_ids), ' vs parent') : null;
    const pars = d.parents && d.parents.length ? d.parents : parent ? [parent] : [];
    const seedLike = !!d.seed || d.operation === 'seed' || d.id.startsWith('g0-');
    const canVs = !!d.diff_base && !seedLike && d.worktree_exists !== false && !(pars.length === 1 && pars[0] === 'g0-s0');
    const seg = canVs ? h('div', { class: 'seg', role: 'group', 'aria-label': 'diff against' }, [['parent', 'vs parent'], ['seed', 'vs seed']].map(([v, label]) =>
      h('button', { type: 'button', 'aria-pressed': String(d.diff_base === v), onclick: () => { if (d.diff_base !== v) openCand(d.id, v); }, text: label }))) : null;
    const kids = d.children || [];
    S.kids = {};
    for (const k of kids) S.kids[k.id] = { ...k, parents: [d.id] };
    const tm = d.timing || {};
    const left = h('div', null,
      h('section', null, h('h4', { text: 'What changed' }), h('div', { text: what }),
        h('div', { class: 'path' }, 'worktree ', h('code', { text: d.worktree || '–', title: d.worktree_exists === false ? 'removed' : null }), d.worktree_exists === false ? h('span', { text: 'removed' }) : copyBtn(d.worktree)),
        h('div', { class: 'path' }, 'commit ', h('code', { text: d.commit || '–' }), d.commit ? copyBtn(d.commit) : null),
        seg,
        seg && !(d.files && d.files.length) && !d.diff ? h('div', { class: 'muted', text: '–' }) : null,
        d.files && d.files.length ? h('div', { class: 'files' }, d.files.map((f) => h('div', null, h('span', { class: 'mono', text: f.path, title: f.path }),
          h('span', null, h('span', { class: 'a', text: num(f.added) ? '+' + f.added : '' }), ' ', h('span', { class: 'r', text: num(f.removed) ? '−' + f.removed : '' }))))) : null,
        d.diff ? diffView(d.diff + (d.diff_truncated ? '\n…' : '')) : null,
        h('div', { class: 'path tml' }, [['mutation', tm.agent_seconds], ['gate', tm.gate_seconds], ['validation', tm.validation_seconds]]
          .filter(([, v]) => num(v)).map(([k, v]) => `${k} ${dur(v)}`).join(' · '), candUse(d))),
      d.attempt ? h('section', null, h('h4', { text: 'Attempt' }), h('div', { class: 'muted', text: [
        String(d.attempt.reason || '').replace(/_/g, ' '), d.attempt.stage ? 'at ' + String(d.attempt.stage).replace(/_/g, ' ') : null,
        num(d.attempt.score) ? `${d.attempt.score.toFixed(3)}${num(d.attempt.n) ? ` on ${d.attempt.n} ids` : ''}` : null].filter(Boolean).join(' · ') })) : null,
      kids.length ? h('section', null, h('h4', { text: 'Children' }),
        h('table', { class: 'pt mini' }, h('tbody', null, kids.slice(0, 12).map((k) => h('tr', { class: ID_RE.test(k.id) ? 'cd' : null, 'data-c': ID_RE.test(k.id) ? k.id : null },
          h('td', { class: 'mono', text: k.id }), outcomeCell(k), scoreCell(k))))),
        kids.length > 12 ? h('div', { class: 'muted', text: `+${kids.length - 12}` }) : null) : null,
      d.prompt ? h('section', null, h('details', { class: 'tp' }, h('summary', { text: 'Task prompt' }),
        h('pre', { class: 'prompt', text: d.prompt + (d.prompt_truncated ? '\n…' : '') }))) : null);
    const low = d.lowest || [];
    const valIds = d.accepted || (d.attempt && d.attempt.stage === 'val_stage');
    if (rows) {
      ciState(d);
      return [candCrumbs(d.id), h('div', { class: 'ch' }, h('h2', { text: d.id }), h('span', null, sub)), fs, vpl,
        h('div', { class: 'cgrid' }, left, h('div', null, famSection(d), h('section', { class: 'cis', id: 'cis' }, ciBody(d))))];
    }
    const right = h('div', null, h('h4', { text: 'Lowest instances' + (valIds ? '' : ' (gate)') + (bin ? ' (failed trials)' : '') }),
      low.length ? h('div', { class: 'low' }, low.map((x) => h('div', null,
        valIds ? h('button', { class: 'iid mono', type: 'button', 'data-inst': x.instance, text: x.instance, title: x.instance })
          : h('span', { class: 'iid mono', text: x.instance, title: x.instance }),
        h('span', { class: x.score === 0 ? 'sc0' : null, text: bin ? (x.score ? 'ok' : 'fail') : fixed(x.score, 2) }),
        h('span', { class: 'si', text: x.side_info || '', title: x.side_info || null })))) : h('div', { class: 'muted', text: '–' }));
    return [candCrumbs(d.id), h('div', { class: 'ch' }, h('h2', { text: d.id }), h('span', null, sub)), fs, vpl, h('div', { class: 'cgrid' }, left, right)];
  }

  // the candidate's recorded usage: dollars, or tokens when the backend reports no cost
  function candUse(d) {
    const u = d.usage;
    if (!u || !(uVal(u, true) > 0 || uVal(u, false) > 0)) return null;
    const cost = num(u.cost_usd) && u.cost_usd > 0, tm = d.timing || {};
    const any = ['agent_seconds', 'gate_seconds', 'validation_seconds'].some((k) => num(tm[k]));
    const tip = `${compact(u.input_tokens)} in · ${compact(u.output_tokens)} out` + (u.cached_input_tokens ? ` · ${compact(u.cached_input_tokens)} cached` : '')
      + (d.usage_source === 'worktree' ? ' · from the worktree' : '');
    return [any ? ' · ' : '', h('b', { class: 'use', title: tip, text: cost ? money(u.cost_usd) : `${compact(u.input_tokens)} in · ${compact(u.output_tokens)} out` }),
      cost ? null : ' tokens'];
  }

  // candidate instances: every scored instance against the parent, in three tabs, paged; a row opens its side_info
  const CI_PAGE = 100;
  function famOf(iid) {
    const c = S.static.config, sep = c.group_key_separator == null ? '__' : c.group_key_separator, k = sep ? iid.indexOf(sep) : -1;
    return k >= 0 ? iid.slice(0, k) : '';
  }
  const ciChanged = (r) => num(r.parent_score) && num(r.score) && r.score !== r.parent_score;
  function ciState(d) {
    if (S.ci && S.ci.id === d.id) return S.ci;
    const any = d.compare_to && d.instances_basis === 'val' && d.instances.some(ciChanged);
    S.ci = { id: d.id, tab: any ? 'changed' : 'lowest', dir: null, fam: null, page: 0, open: new Set(), side: new Map(), hist: new Map(), ix: null };
    return S.ci;
  }
  function ciRows(d) {
    const st = S.ci;
    if (!st.ix || st.ix.d !== d) st.ix = { d, rows: d.instances.map((r, i) => ({ ...r, i, fam: famOf(String(r.id)) })) };
    return st.fam == null ? st.ix.rows : st.ix.rows.filter((r) => r.fam === st.fam);
  }
  function ciJump(dir) {
    const st = S.ci;
    if (!st) return;
    Object.assign(st, { tab: 'changed', dir, fam: null, page: 0 });
    ciPaint();
    const el = $('cis');
    if (el) el.scrollIntoView({ block: 'start', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
  }
  function ciPaint() {
    const d = S.candData;
    if (!d || !S.ci || S.ci.id !== d.id) return;
    const sec = $('cis'), fam = $('cfam');
    if (sec) fill(sec, ciBody(d));
    if (fam) fam.replaceWith(famSection(d));
  }
  const civ = (v) => (!num(v) ? '–' : binary() ? (v ? 'ok' : 'fail') : v.toFixed(3));
  const sgn = (v, f) => (v > 0 ? '+' : v < 0 ? '−' : '±') + f(Math.abs(v));
  function ciBody(d) {
    const st = S.ci, bin = binary(), gate = !!d.instances_basis && d.instances_basis !== 'val';
    const base = ciRows(d), canChange = !!d.compare_to && !gate;
    const changed = canChange ? base.filter(ciChanged) : [];
    const up = changed.filter((r) => r.score > r.parent_score);
    if (!canChange && st.tab === 'changed') st.tab = 'lowest';
    const set = (o) => { Object.assign(st, o); ciPaint(); };
    const tab = (k, label, n) => h('button', {
      class: 'chip' + (st.tab === k ? ' on' : ''), type: 'button', role: 'tab', 'aria-selected': String(st.tab === k),
      onclick: () => set({ tab: k, page: 0, dir: k === 'changed' ? st.dir : null }),
    }, label, n != null ? h('span', { class: 'n', text: String(n) }) : null);
    const head = h('div', { class: 'cih' }, h('h4', null, 'Instances', gate ? h('span', { class: 'bt', text: 'gate' }) : null),
      d.instances.length ? h('div', { class: 'chips', role: 'tablist', 'aria-label': 'instances' },
        canChange ? tab('changed', 'Changed', changed.length) : null, tab('all', 'All', base.length), tab('lowest', 'Lowest', null)) : null,
      st.tab === 'changed' && changed.length ? h('div', { class: 'seg', role: 'group', 'aria-label': 'direction' },
        [[null, 'both'], ['up', `+${up.length}`], ['down', `−${changed.length - up.length}`]].map(([v, label]) =>
          h('button', { type: 'button', 'aria-pressed': String(st.dir === v), onclick: () => set({ dir: v, page: 0 }), text: label }))) : null,
      st.fam != null ? h('button', { class: 'chip on fx', type: 'button', 'aria-label': 'clear family filter', onclick: () => set({ fam: null, page: 0 }) },
        st.fam || '–', h('span', { class: 'n', text: '×' })) : null);
    if (!d.instances.length) return [head, h('div', { class: 'muted', text: '–' })];
    let list;
    if (st.tab === 'changed') {
      list = changed.filter((r) => !st.dir || (st.dir === 'up') === (r.score > r.parent_score));
      // improved first, then the biggest moves, then evaluation order
      list.sort((a, b) => (b.score > b.parent_score) - (a.score > a.parent_score)
        || Math.abs(b.score - b.parent_score) - Math.abs(a.score - a.parent_score) || a.i - b.i);
    } else if (st.tab === 'lowest') list = [...base].sort((a, b) => (num(a.score) ? a.score : Infinity) - (num(b.score) ? b.score : Infinity) || a.i - b.i);
    else list = base;
    const pages = Math.max(1, Math.ceil(list.length / CI_PAGE));
    st.page = Math.min(st.page, pages - 1);
    const a = st.page * CI_PAGE, z = Math.min(list.length, a + CI_PAGE);
    const pager = list.length > CI_PAGE ? h('div', { class: 'pager' },
      h('button', { class: 'pg', type: 'button', 'aria-label': 'previous', disabled: st.page === 0, onclick: () => set({ page: st.page - 1 }), text: '‹' }),
      h('span', { text: `${a + 1}–${z} of ${list.length}` }),
      h('button', { class: 'pg', type: 'button', 'aria-label': 'next', disabled: st.page >= pages - 1, onclick: () => set({ page: st.page + 1 }), text: '›' })) : null;
    return [head, h('div', { class: 'irows' + (bin || !canChange ? ' nod' : '') }, list.length ? list.slice(a, z).map((r) => ciRow(d, r)) : h('div', { class: 'muted', text: '–' })), pager];
  }
  function ciRow(d, r) {
    const st = S.ci, open = st.open.has(r.id), hasP = num(r.parent_score) && num(r.score), dv = hasP ? r.score - r.parent_score : null;
    const dir = dv > 0 ? 'a' : dv < 0 ? 'r' : null;
    const row = h('div', { class: 'irow' + (open ? ' open' : '') });
    const btn = h('button', { class: 'irb', type: 'button', 'aria-expanded': String(open), onclick: () => {
      if (st.open.has(r.id)) st.open.delete(r.id); else st.open.add(r.id);
      row.replaceWith(ciRow(d, r));
    } },
    h('span', { class: 'mono iname', text: r.id, title: r.id }),
    h('span', { class: 'isc' }, hasP ? [h('span', { class: 'pv', text: civ(r.parent_score) }), h('span', { class: 'ar', text: '→' })] : null,
      h('span', { class: dir || (r.score === 0 ? 'z' : null), text: civ(r.score) })),
    binary() || !d.compare_to || d.instances_basis !== 'val' ? null : h('span', { class: 'idl ' + (dir || ''), text: dv ? sgn(dv, (v) => v.toFixed(3)) : '' }),
    // the Lowest tab keeps the compacted side_info the API already sends for the lowest ids inline; a click opens the full one
    !open && st.tab === 'lowest' && lowSide(d).has(r.id) ? h('span', { class: 'isi', text: lowSide(d).get(r.id), title: lowSide(d).get(r.id) }) : null);
    kids(row, [btn, open ? ciDetail(d, r) : null]);
    return row;
  }
  function lowSide(d) {
    const st = S.ci;
    if (!st.low || st.low.d !== d) st.low = { d, m: new Map((d.lowest || []).filter((x) => x && x.side_info).map((x) => [String(x.instance), String(x.side_info)])) };
    return st.low.m;
  }
  function ciDetail(d, r) {
    const st = S.ci, box = h('div', { class: 'ixp' }), gate = !!d.instances_basis && d.instances_basis !== 'val';
    const paint = (doc) => {
      if (doc === null) { fill(box, h('div', { class: 'muted', text: 'could not load' })); return; }
      const block = (who, sc, info, tag) => [h('div', { class: 'ixh' }, who, h('b', { text: civ(sc) }), tag ? h('span', { class: 'bt', text: tag }) : null),
        h('pre', { class: 'six' + (info ? '' : ' muted'), text: info || '–' })];
      const p = doc.parent;
      const hist = gate ? null : h('div', { class: 'ixs' });
      fill(box, block(h('span', { class: 'mono', text: d.id }), doc.score, doc.side_info, gate ? 'gate' : null),
        p ? block(h('button', { class: 'lnk mono', type: 'button', 'data-open': p.id, text: p.id }), p.score, p.side_info, null) : null, hist);
      if (hist) ciHist(hist, r.id);
    };
    const have = st.side.get(r.id);
    if (have !== undefined) paint(have);
    else {
      box.append(h('div', { class: 'muted', text: '…' }));
      get('/api/candidate/' + encodeURIComponent(d.id) + '/instance/' + encodeURIComponent(r.id)).then((doc) => doc, () => null).then((doc) => {
        if (S.ci !== st) return;
        st.side.set(r.id, doc);
        if (box.isConnected) paint(doc);
      });
    }
    return box;
  }
  // the instance's score across accepted candidates, as in the frontier strip's panel
  function ciHist(el, iid) {
    const st = S.ci, have = st.hist.get(iid);
    const paint = (doc) => {
      if (!doc) { el.remove(); return; }
      const hold = doc.holders || [], sv = s('svg', { class: 'isp' });
      fill(el, h('div', { class: 'iph' }, h('span', null, 'best ', h('b', { text: instVal(doc.best) })),
        hold.length ? h('span', { class: 'hl' }, 'held by ', hold.slice(0, 4).map((x, i) => [i ? ', ' : '', h('button', { class: 'lnk', type: 'button', 'data-open': x, text: x })]),
          hold.length > 4 ? ` +${hold.length - 4}` : '') : null), sv);
      instSpark(sv, doc);
    };
    if (have !== undefined) { paint(have); return; }
    get('/api/instance/' + encodeURIComponent(iid)).then((doc) => doc, () => null).then((doc) => {
      if (S.ci !== st) return;
      st.hist.set(iid, doc);
      if (el.isConnected) paint(doc);
    });
  }
  // per family (instance id prefix, as the frontier strip groups them): this candidate against its parent; a click filters the instances
  function famSection(d) {
    const fams = d.families;
    if (!fams || !fams.length || !S.ci) return h('div', { id: 'cfam', hidden: true });
    const st = S.ci, bin = binary(), gate = !!d.instances_basis && d.instances_basis !== 'val', par = !!d.compare_to && !gate;
    const val = (f, p) => {
      const m = p ? f.parent_mean : f.mean, k = p ? f.parent_successes : f.successes;
      return !num(m) ? '–' : bin && num(k) ? `${k}/${f.n}` : m.toFixed(3);
    };
    const delta = (f) => {
      if (bin && num(f.successes) && num(f.parent_successes)) { const v = f.successes - f.parent_successes; return [v, v ? sgn(v, String) : '0']; }
      return num(f.delta) ? [f.delta, Math.abs(f.delta) < 5e-4 ? '0' : sgn(f.delta, (v) => v.toFixed(3))] : [0, '–'];
    };
    return h('section', { class: 'cfam', id: 'cfam' }, h('h4', null, 'Families', gate ? h('span', { class: 'bt', text: 'gate' }) : null),
      h('div', { class: 'otw fw' }, h('table', { class: 'ot fam' },
        h('thead', null, h('tr', null, h('th', { text: 'family' }), h('th', { text: 'candidate' }), par ? [h('th', { text: 'parent', title: d.compare_to }), h('th', { text: 'Δ' })] : null)),
        h('tbody', null, fams.map((f) => {
          const [dv, dt] = delta(f), on = st.fam === f.name;
          return h('tr', { class: on ? 'sel' : null, tabindex: 0, role: 'button', 'aria-pressed': String(on),
            onclick: () => { Object.assign(st, { fam: on ? null : f.name, page: 0 }); ciPaint(); },
            onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.currentTarget.click(); } } },
          h('td', { class: 'mono', text: f.name || '–', title: f.name || null }), h('td', { text: val(f, false) }),
          par ? [h('td', { class: 'pv', text: val(f, true) }), h('td', { class: dv > 0 ? 'a' : dv < 0 ? 'r' : 'pv', text: dt })] : null);
        })))));
  }

  // mutation
  function dl(rows) {
    return h('dl', { class: 'dl' }, rows.filter((r) => r && r[1] != null && r[1] !== '').map(([k, v, warn]) => [h('dt', { text: k }), h('dd', { class: warn ? 'w' : null, text: String(v), title: String(v) })]));
  }
  function details(within, source) {
    return ((S.run && S.run.details) || []).filter((d) => d.within === within && (!source || d.source === source) && num(d.seconds));
  }
  function barRows(rows, colour) {
    const max = Math.max(...rows.map((r) => r[2] || 0), 1e-9);
    return h('div', { class: 'mods' }, rows.map(([name, src, sec]) => h('div', null,
      h('span', null, h('div', { text: name }), src ? h('div', { class: 'src', text: src, title: src }) : null),
      h('span', { class: 'num', text: num(sec) ? dur(sec) : '–' }),
      h('div', { class: 'hb' }, num(sec) ? h('i', { style: { width: (100 * sec / max).toFixed(1) + '%', background: `var(${colour})` } }) : null))));
  }
  function drawMutation() {
    const c = S.static.config, a = c.agent || {}, sb = c.sandbox || {}, iso = S.static.isolation || {}, r = S.run;
    const mu = (r && r.loop && r.loop.mutation) || {}, f = (r && r.health && r.health.failures) || {};
    const warnKeys = new Set(warnings().flatMap((w) => w.keys || []));
    const fs = h('div', { class: 'fs' },
      h('span', null, h('b', { text: r ? String(mu.running ?? 0) : '–' }), 'running'),
      h('span', null, h('b', { text: r ? String(mu.finished ?? 0) : '–' }), 'finished'),
      h('span', null, h('b', { text: dur(mu.median_seconds) }), 'median'),
      h('span', null, h('b', { class: f.agent_timeouts ? 'bad' : null, text: r ? String(f.agent_timeouts ?? 0) : '–' }), 'timeouts'),
      h('span', null, h('b', { class: f.agent_errors ? 'bad' : null, text: r ? String(f.agent_errors ?? 0) : '–' }), 'agent errors'));
    fs.style.marginTop = '8px';
    const facts = dl([
      ['backend', a.backend], ['model', a.model ?? 'default'], ['effort', a.effort ?? 'default'], ['max turns', a.max_turns],
      ['web access', a.web_access == null ? null : a.web_access ? 'on' : 'off', warnKeys.has('agent.web_access')],
      ['runs in', iso.agent === 'sandbox' ? 'sandbox' : 'host'],
      ...(iso.agent === 'sandbox' ? [['network', sb.network, warnKeys.has('sandbox.network')], ['host gateway', sb.add_host_gateway ? 'on' : 'off'],
        ['timeout', num(sb.timeout_seconds) ? dur(sb.timeout_seconds) : null], ['auth volume', sb.auth_volume ?? 'shared', warnKeys.has('sandbox.auth_volume')],
        ['image', sb.image], ['cpus', sb.cpus], ['memory', sb.memory], ['hidden from agent', num(sb.omit_from_agent) ? `${sb.omit_from_agent} paths` : null]] : []),
      ['proposals', c.proposals_per_generation != null ? `${c.proposals_per_generation} per generation` : null],
      ['mutations per parent', c.mutations_per_parent], ['sibling history', c.sibling_history == null ? null : c.sibling_history ? 'on' : 'off'],
      ['env', (c.env_names || []).join(', ') || null],
    ]);
    const breakdown = details('agent').reduce((m, d) => { m[d.name] = (m[d.name] || 0) + d.seconds; return m; }, {});
    const brows = Object.entries(breakdown).sort((x, y) => y[1] - x[1]).map(([n, sec]) => [n, null, sec]);
    const right = h('div', null,
      h('section', null, h('h4', { text: 'Agent time per candidate' }), s('svg', { class: 'as', id: 'as' })),
      brows.length ? h('section', null, h('h4', { text: 'Where agent time goes' }), barRows(brows, '--mag')) : null);
    fill($('v-mutation'), crumbs(['Loop', 'map'], ['Mutation', '']),
      h('div', { class: 'ch' }, h('h2', { text: 'Mutation' }), h('span', { text: [a.backend, a.model, a.effort].filter(Boolean).join(' · ') })),
      fs, h('div', { class: 'cgrid' }, h('div', null, facts), right));
    agentScatter(num(sb.timeout_seconds) && iso.agent === 'sandbox' ? sb.timeout_seconds : null);
  }
  function agentScatter(timeout) {
    const S0 = $('as');
    if (!S0) return;
    const pts = ((S.run && S.run.candidates) || []).filter((c) => num(c.agent_seconds));
    const W = S0.clientWidth || 400, Hh = 150, pl = 40, pb = 18, pt = 8;
    const L = css('--line');
    S0.appendChild(s('line', { x1: pl, x2: W, y1: Hh - pb, y2: Hh - pb, stroke: L }));
    if (!pts.length) { S0.appendChild(s('text', { x: pl + 4, y: Hh / 2, class: 'ax', text: 'no finished agent call yet' })); return; }
    const G = Math.max(1, ...pts.map((c) => c.generation || 0)), top = Math.max(timeout || 0, ...pts.map((c) => c.agent_seconds));
    const x = (g) => pl + 8 + g / G * (W - pl - 16), y = (v) => Hh - pb - 4 - v / top * (Hh - pb - pt - 8);
    for (const v of [0, top]) S0.appendChild(s('text', { x: pl - 6, y: y(v) + 4, class: 'ax', 'text-anchor': 'end', text: dur(v) }));
    S0.appendChild(s('text', { x: (W + pl) / 2, y: Hh - 3, class: 'ax', 'text-anchor': 'middle', text: 'generation' }));
    for (const g of [0, G]) S0.appendChild(s('text', { x: x(g), y: Hh - 3, class: 'ax', 'text-anchor': 'middle', text: g }));
    if (timeout) S0.appendChild(s('line', { x1: pl, x2: W, y1: y(timeout), y2: y(timeout), stroke: css('--amber'), 'stroke-dasharray': '3 4', opacity: 0.7 }, s('title', { text: 'agent timeout ' + dur(timeout) })));
    const M = css('--mag');
    for (const c of pts) S0.appendChild(s('circle', { cx: x((c.generation || 0) + jitter(c.id) * 0.5), cy: y(c.agent_seconds), r: pts.length > 200 ? 2.5 : 3.5, fill: M, opacity: 0.55, 'data-c': c.id, class: 'cd' }));
  }

  // validation
  function drawValidation() {
    const st = S.static, c = st.config, ev = c.evaluator || {}, iso = st.isolation || {}, r = S.run, sizes = st.sizes || {};
    const va = (r && r.loop && r.loop.validation) || {}, gt = (r && r.loop && r.loop.gate) || {}, f = (r && r.health && r.health.failures) || {};
    const types = Object.entries(f.evaluator_error_types || {});
    const where = iso.evaluator === 'sandbox' ? 'sandbox' : iso.evaluator === 'sidecar' ? 'sidecar' : 'host';
    const fs = h('div', { class: 'fs' },
      h('span', null, h('b', { text: dur(va.median_seconds) }), 'median validation'),
      h('span', null, h('b', { text: r ? String(va.count ?? 0) : '–' }), 'validated'),
      h('span', null, h('b', { text: String(va.ids ?? sizes.val ?? '–') }), 'validation ids'),
      h('span', null, h('b', { text: String(gt.ids_each ?? c.minibatch_size ?? '–') }), 'gate ids'),
      h('span', null, h('b', { class: f.evaluator_errors ? 'bad' : null, text: r ? String(f.evaluator_errors ?? 0) : '–' }), 'evaluator errors'));
    fs.style.marginTop = '8px';
    const facts = dl([
      ['command', ev.command], ['runs on', where], ['train ids', sizes.train], ['validation ids', sizes.val],
      ['minibatch', c.minibatch_size], ['sampler', c.batch_sampler], ['acceptance', c.acceptance_criterion ? String(c.acceptance_criterion).replace(/_/g, ' ') : null],
      ['val stage', c.val_stage_size], ['perfect score', c.perfect_score_threshold], ['stop at', c.stop_at_score],
      ['cache', c.cache_evaluation == null ? null : c.cache_evaluation ? 'on' : 'off'], ['workers', c.max_workers],
      ['protected files', ev.protected_files], ['extra commands', ev.extra_commands || null],
      ['feedback', [ev.include_stdout ? 'stdout' : null, ev.include_stderr ? 'stderr' : null].filter(Boolean).join(' + ') || 'none'],
    ]);
    const secs = details('evaluate', 'timed').reduce((m, d) => { m[d.name] = (m[d.name] || 0) + d.seconds; return m; }, {});
    const mods = (st.evaluator_modules || []).map((m) => [m.name, m.source, secs[m.name] ?? null]);
    for (const [n, sec] of Object.entries(secs)) if (!mods.some((m) => m[0] === n)) mods.push([n, null, sec]);
    const right = h('div', null,
      mods.length ? h('section', null, h('h4', { text: 'Evaluator modules' }), barRows(mods, '--pur')) : null,
      types.length ? h('section', null, h('h4', { text: 'Evaluator errors' }), h('dl', { class: 'dl' }, types.map(([k, v]) => [h('dt', { text: k }), h('dd', { class: 'w', text: String(v) })]))) : null);
    fill($('v-validation'), crumbs(['Loop', 'map'], ['Validation', '']),
      h('div', { class: 'ch' }, h('h2', { text: 'Validation' }), h('span', { class: 'mono', text: ev.command || '' })),
      fs, h('div', { class: 'cgrid' }, h('div', null, facts), right));
  }

  // ---------------- lineage, proposals, instance history ----------------
  function loadInto(o, key, path) {
    if (!S.run) { o.data = null; o.key = key; return Promise.resolve(null); }
    if (o.key === key) return Promise.resolve(o.data);
    if (o.loading && o.loading.key === key) return o.loading.p;
    const p = get(path).then((d) => { o.data = d; o.key = key; return d; }).catch(() => o.data)
      .finally(() => { if (o.loading && o.loading.key === key) o.loading = null; });
    o.loading = { key, p };
    return p;
  }
  // new attempts, skips and failures show up as a gate decision, a failure count or a new generation in the snapshot
  function lineageKey() {
    const r = S.run;
    return r ? [frontierKey(), r.loop && r.loop.gate && r.loop.gate.tried, r.generation && r.generation.current, r.status && r.status.state,
      JSON.stringify(r.health && r.health.failures)].join('|') : 'none';
  }
  function proposalsKey() {
    const r = S.run;
    return r ? lineageKey() + '|' + ((r.now && r.now.lanes) || []).map((l) => `${l.id}:${l.phase}:${l.result ? l.result.status : ''}`).join(',') : 'none';
  }

  // lineage: x = generation; each generation's nodes are placed as close to their parents' rows as the minimum gap allows
  async function drawLineage() {
    const V = $('v-lineage');
    if (!V.firstChild) fill(V, crumbs(['Loop', 'map'], ['Lineage', '']));
    await loadInto(S.lin, lineageKey(), '/api/lineage');
    if (S.view === 'lineage') paintLineage();
  }
  function paintLineage() {
    const V = $('v-lineage'), d = S.lin.data, nodes = (d && d.nodes) || [];
    S.lin.byId = {};
    for (const n of nodes) S.lin.byId[n.id] = n;
    // a pinned id the loaded run does not have is dropped; a pinned rejected node turns rejected nodes on so it can be seen
    const pinned = S.lin.pin && S.lin.byId[S.lin.pin];
    if (S.lin.pin && d && !pinned) { S.lin.pin = null; setHash(); }
    if (pinned && !pinned.accepted) S.lin.showRej = true;
    const nacc = nodes.filter((n) => n.accepted).length, nrej = nodes.length - nacc;
    const showRej = S.lin.showRej ?? nodes.length <= 400;
    const best = d && d.best ? S.byId[d.best] || null : null;
    const stats = h('div', { class: 'fs' },
      h('span', null, h('b', { text: S.run ? String(nacc) : '–' }), 'accepted'),
      h('span', null, h('b', { text: S.run ? String(nodes.filter((n) => n.front).length) : '–' }), 'on the frontier'),
      h('span', null, h('b', { text: S.run ? String(nrej) : '–' }), 'rejected'),
      best ? h('span', null, h('b', { text: score(best) }), 'best · ' + best.id) : null,
      d && num(d.truncated) && d.truncated > 0 ? h('span', null, h('b', { text: String(d.truncated) }), 'earlier rejected not shown') : null,
      nrej ? h('span', { class: 'sel' }, h('button', {
        class: 'tog' + (showRej ? ' on' : ''), type: 'button', id: 'lrej', 'aria-pressed': String(showRej), text: 'show rejected',
        onclick: () => {
          S.lin.showRej = !showRej;
          const p = S.lin.pin && S.lin.byId[S.lin.pin];
          if (showRej && p && !p.accepted) { S.lin.pin = null; setHash(); }
          paintLineage();
        },
      })) : null);
    const svg = h('div', { class: 'lgw', id: 'lg', role: 'img', 'aria-label': 'Lineage' });
    if (hoverEl && V.contains(hoverEl)) { hoverEl = null; hideCard(); }
    S.lin.hov = null;
    fill(V, crumbs(['Loop', 'map'], ['Lineage', '']), stats, svg);
    lineageGraph(svg, d, showRej ? nodes : nodes.filter((n) => n.accepted));
    linFocus();
    linPanel();
  }
  const counter = (id) => { const m = /(\d+)$/.exec(id); return m ? +m[1] : 0; };
  function layoutLineage(list) {
    const cols = new Map(), pos = new Map();
    for (const n of list) { const g = n.generation || 0; if (!cols.has(g)) cols.set(g, []); cols.get(g).push(n); }
    for (const g of [...cols.keys()].sort((a, b) => a - b)) {
      const want = new Map();
      for (const n of cols.get(g)) {
        const ps = (n.parents || []).map((p) => pos.get(p)).filter(num);
        want.set(n, ps.length ? avg(ps) : 0);
      }
      const col = cols.get(g).sort((a, b) => want.get(a) - want.get(b) || counter(a.id) - counter(b.id));
      const gap = (n) => (n.accepted ? 1 : 0.6), off = [0];
      for (let i = 1; i < col.length; i++) off.push(off[i - 1] + (gap(col[i - 1]) + gap(col[i])) / 2);
      // least-squares rows that keep the order and the gaps: pool adjacent violators on want - offset
      const blocks = [];
      col.forEach((n, i) => {
        let b = { sum: want.get(n) - off[i], cnt: 1 };
        while (blocks.length && blocks[blocks.length - 1].sum / blocks[blocks.length - 1].cnt > b.sum / b.cnt) {
          const q = blocks.pop();
          b = { sum: q.sum + b.sum, cnt: q.cnt + b.cnt };
        }
        blocks.push(b);
      });
      let i = 0;
      for (const b of blocks) for (let j = 0; j < b.cnt; j++, i++) pos.set(col[i].id, b.sum / b.cnt + off[i]);
    }
    return pos;
  }
  // the whole DAG (rejected nodes included, whatever is drawn): children, and ancestor and descendant sets memoized per data load
  function linIndex() {
    const d = S.lin.data;
    if (S.lin.ix && S.lin.ix.d === d) return S.lin.ix;
    const by = {}, kids = {};
    for (const n of (d && d.nodes) || []) { by[n.id] = n; kids[n.id] = []; }
    for (const n of Object.values(by)) for (const p of n.parents || []) if (kids[p]) kids[p].push(n.id);
    const walk = (memo, next) => (id) => {
      let out = memo.get(id);
      if (out) return out;
      out = new Set();
      const st = [...next(id)];
      while (st.length) { const v = st.pop(); if (out.has(v) || !by[v]) continue; out.add(v); st.push(...next(v)); }
      memo.set(id, out);
      return out;
    };
    const ix = { d, by, kids };
    ix.anc = walk(new Map(), (id) => (by[id] && by[id].parents) || []);
    ix.desc = walk(new Map(), (id) => kids[id] || []);
    S.lin.ix = ix;
    return ix;
  }
  // four stacked svgs: axis, lines, nodes and the focus overlay. A focus fades the lines and nodes layers as whole
  // composited layers and redraws only the overlay, so a hover at 1000+ nodes repaints a few elements, not the tree
  function lineageGraph(wrap, d, list) {
    const W = wrap.clientWidth || 800, Hh = wrap.clientHeight || 400, pl = 18, pr = 18, pt = 12, pb = 24;
    const C = { cyan: css('--cyan'), ink: css('--ink'), ink2: css('--ink2'), ink3: css('--ink3'), line: css('--line'), paper: css('--paper') };
    S.lin.pts = [];
    S.lin.g = null;
    const [AX, ED, ND, OV] = ['lx', 'le', 'ln', 'lo'].map((k) => wrap.appendChild(s('svg', { class: 'lg ' + k, 'aria-hidden': 'true' })));
    let G = Math.max(1, (d && d.max_generation) || 0);
    for (const n of list) if ((n.generation || 0) > G) G = n.generation;
    const x = (g) => pl + (g || 0) / G * (W - pl - pr);
    AX.appendChild(s('line', { x1: pl, x2: W - pr, y1: Hh - pb + 6, y2: Hh - pb + 6, stroke: C.line }));
    for (const g of ticks(G)) AX.appendChild(s('text', { x: x(g), y: Hh - 4, class: 'ax', 'text-anchor': 'middle', text: g }));
    if (!list.length) return;
    const pos = layoutLineage(list), byId = {};
    let lo = Infinity, hi = -Infinity, mlo = Infinity, mhi = -Infinity;
    for (const n of list) {
      byId[n.id] = n;
      const v = pos.get(n.id); if (v < lo) lo = v; if (v > hi) hi = v;
      if (n.accepted && num(n.mean)) { if (n.mean < mlo) mlo = n.mean; if (n.mean > mhi) mhi = n.mean; }
    }
    const unit = Math.min(44, (Hh - pt - pb) / Math.max(1, hi - lo)), mid = (pt + Hh - pb) / 2, c0 = (lo + hi) / 2;
    const y = (v) => mid + (v - c0) * unit;
    const big = list.length > 600, colW = (W - pl - pr) / G;
    const rmax = Math.max(2.4, Math.min(big ? 3.4 : 5.5, unit * 0.45, colW * 0.8 + 1.6));
    const tt = (n) => (num(n.mean) && mhi > mlo ? (n.mean - mlo) / (mhi - mlo) : 1);
    // edges: one path per kind keeps 1000+ nodes cheap; each segment is kept by child so a focus can draw its own lines
    const g = { wrap, W, els: new Map(), edges: new Map(), at: new Map(), cols: new Map(), cur: undefined };
    let dN = '', dB = '', dR = '';
    const f1 = (v) => v.toFixed(1);
    for (const n of list) {
      const es = [];
      for (const p of n.parents || []) {
        const pn = byId[p];
        if (!pn) continue;
        const x1 = x(pn.generation), y1 = y(pos.get(p)), x2 = x(n.generation), y2 = y(pos.get(n.id)), mx = (x1 + x2) / 2;
        const seg = `M${f1(x1)} ${f1(y1)}C${f1(mx)} ${f1(y1)} ${f1(mx)} ${f1(y2)} ${f1(x2)} ${f1(y2)}`;
        const k = !n.accepted ? 'r' : n.best_path && pn.best_path ? 'b' : 'n';
        if (k === 'r') dR += seg; else if (k === 'b') dB += seg; else dN += seg;
        es.push([p, seg, k]);
      }
      g.edges.set(n.id, es);
      g.at.set(n.id, [x(n.generation), y(pos.get(n.id))]);
      const gen = n.generation || 0;
      if (!g.cols.has(gen)) g.cols.set(gen, []);
      g.cols.get(gen).push(n.id);
    }
    for (const col of g.cols.values()) col.sort((a, b) => g.at.get(a)[1] - g.at.get(b)[1]);
    // at rest the ordinary lines recede so the nodes and the best path read first
    if (dR) ED.appendChild(s('path', { d: dR, fill: 'none', stroke: C.ink3, 'stroke-width': 0.7, opacity: big ? 0.08 : 0.12 }));
    if (dN) ED.appendChild(s('path', { d: dN, fill: 'none', stroke: C.ink3, 'stroke-width': big ? 0.7 : 0.9, opacity: big ? 0.14 : 0.2 }));
    if (dB) ED.appendChild(s('path', { d: dB, fill: 'none', stroke: C.cyan, 'stroke-width': big ? 1.6 : 2, opacity: 0.9, 'stroke-linecap': 'round' }));
    const own = (id, e) => { const a = g.els.get(id); if (a) a.push(e); else g.els.set(id, [e]); return e; };
    const dot = (n, a) => {
      const e = ND.appendChild(s('circle', { cx: f1(x(n.generation)), cy: f1(y(pos.get(n.id))), 'data-c': n.id, class: 'cd', ...a }));
      S.lin.pts.push({ x: x(n.generation), y: y(pos.get(n.id)), r: +a.r, el: e });
      own(n.id, e);
    };
    for (const n of list) if (!n.accepted) dot(n, { r: big ? 1.5 : 2.2, fill: C.paper, stroke: C.ink3, 'stroke-width': 1 });
    // dominated first, then the frontier with a ring, then the best on top
    const acc = list.filter((n) => n.accepted).sort((a, b) => (a.front - b.front) || (tt(a) - tt(b)));
    for (const n of acc) {
      const v = tt(n), r = n.front ? rmax : big ? 1.3 + 0.9 * v : 1.8 + (rmax - 1.8) * 0.75 * v;
      if (n.front) own(n.id, ND.appendChild(s('circle', { cx: f1(x(n.generation)), cy: f1(y(pos.get(n.id))), r: r + 2.2, fill: 'none', stroke: C.cyan, 'stroke-width': 1.2 })));
      dot(n, { r: r.toFixed(2), fill: C.cyan, 'fill-opacity': (0.3 + 0.7 * v).toFixed(2) });
    }
    const b = d && d.best && byId[d.best];
    if (b) own(b.id, ND.appendChild(s('circle', { cx: f1(x(b.generation)), cy: f1(y(pos.get(b.id))), r: rmax + 4.4, fill: 'none', stroke: C.ink, 'stroke-width': 1.2 })));
    // the overlay: the focus's lines (rejected, ordinary, best path), copies of its nodes, and the pin's ring
    const hl = OV.appendChild(s('g', { fill: 'none', 'stroke-linecap': 'round' }));
    g.hl = [
      hl.appendChild(s('path', { stroke: C.ink3, 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0.85 })),
      hl.appendChild(s('path', { stroke: C.ink2, 'stroke-width': big ? 1.2 : 1.4, opacity: 0.9 })),
      hl.appendChild(s('path', { stroke: C.cyan, 'stroke-width': big ? 2 : 2.4 })),
    ];
    g.lit = OV.appendChild(s('g'));
    g.pinRing = OV.appendChild(s('circle', { class: 'lpin', r: rmax + 3.2, fill: 'none', stroke: C.ink, 'stroke-width': 2 }));
    S.lin.g = g;
    // nearest-node hover and click: the dots themselves take no pointer events, so dense layers stay usable
    const near = (ev) => {
      const bb = wrap.getBoundingClientRect(), mx = ev.clientX - bb.left, my = ev.clientY - bb.top;
      let hit = null, bd = 64;
      for (const q of S.lin.pts) {
        const dx = q.x - mx, dy = q.y - my, dd = dx * dx + dy * dy - q.r * q.r;
        if (dd < bd) { bd = dd; hit = q; }
      }
      return hit ? hit.el : null;
    };
    wrap.onmousemove = (ev) => {
      const el = near(ev);
      wrap.style.cursor = el ? 'pointer' : '';
      if (el === hoverEl) return;
      hoverEl = el;
      S.lin.hov = el ? el.getAttribute('data-c') : null;
      linFocus();
      // while a node is pinned its panel says it all: no hover card on top of it
      if (el && !S.lin.pin) showCard(el); else hideCard();
    };
    wrap.onmouseleave = () => {
      if (hoverEl && wrap.contains(hoverEl)) { hoverEl = null; hideCard(); }
      S.lin.hov = null;
      linFocus();
    };
    // a click pins the node and opens its panel, empty space clears it, a double click opens the candidate
    wrap.onclick = (ev) => { const el = near(ev); linPin(el ? el.getAttribute('data-c') : null); };
    wrap.ondblclick = (ev) => { const el = near(ev); if (el) { $('card').style.display = 'none'; openCand(el.getAttribute('data-c')); } };
  }
  // light the focus (the hovered node, else the pinned one), its ancestors and its descendants: their lines and copies
  // of their nodes go in the overlay, while CSS fades the lines and nodes layers underneath
  function linFocus() {
    const g = S.lin.g;
    if (!g || !g.wrap.isConnected) return;
    const pin = S.lin.pin && g.els.has(S.lin.pin) ? S.lin.pin : null;
    const id = S.lin.hov && g.els.has(S.lin.hov) ? S.lin.hov : pin;
    const at = pin && g.at.get(pin);
    if (at) { g.pinRing.setAttribute('cx', at[0].toFixed(1)); g.pinRing.setAttribute('cy', at[1].toFixed(1)); }
    g.wrap.classList.toggle('pinned', !!at);
    if (id === g.cur) return;
    g.cur = id;
    g.wrap.classList.toggle('fx', !!id);
    // leaving focus keeps the overlay's content so it can fade out with the layers fading back in
    if (!id) return;
    const ix = linIndex(), anc = ix.anc(id), desc = ix.desc(id), frag = document.createDocumentFragment();
    const d = { r: '', n: '', b: '' };
    const light = (v, up) => {
      const es = g.els.get(v);
      if (!es) return;
      for (const e of es) frag.appendChild(e.cloneNode(false));
      for (const [p, seg, k] of g.edges.get(v) || []) if (up ? anc.has(p) : p === id || desc.has(p)) d[k] += seg;
    };
    light(id, true);
    for (const v of anc) light(v, true);
    for (const v of desc) light(v, false);
    g.hl[0].setAttribute('d', d.r); g.hl[1].setAttribute('d', d.n); g.hl[2].setAttribute('d', d.b);
    g.lit.replaceChildren(frag);
  }
  function linPin(id) {
    if (id && !S.lin.byId[id]) return;
    S.lin.pin = id || null;
    S.lin.hov = null;
    // the hover card is off while pinned; unpinning lets the next move over a node show it again
    hideCard();
    hoverEl = null;
    if (id && !S.lin.byId[id].accepted && !(S.lin.g && S.lin.g.els.has(id))) { S.lin.showRej = true; setHash(); paintLineage(); return; }
    setHash();
    linFocus();
    linPanel();
  }
  // keyboard moves of a pinned focus: Left the parent, Right the best child, Up and Down the neighbour in the same generation
  function linKey(key) {
    const g = S.lin.g, id = S.lin.pin;
    if (!g || !id || !g.els.has(id)) return false;
    const n = S.lin.byId[id], ix = linIndex();
    let to = null;
    if (key === 'ArrowLeft') to = (n.parents || []).find((p) => g.els.has(p)) || null;
    else if (key === 'ArrowRight') {
      const val = (c) => (c.accepted ? 1e6 + (num(c.mean) ? c.mean : -1) : pScore(c) ?? -1);
      for (const k of ix.kids[id] || []) if (g.els.has(k) && (!to || val(ix.by[k]) > val(ix.by[to]))) to = k;
    } else if (key === 'ArrowUp' || key === 'ArrowDown') {
      const col = g.cols.get(n.generation || 0) || [], i = col.indexOf(id);
      to = col[i + (key === 'ArrowUp' ? -1 : 1)] || null;
    } else return false;
    if (to) linPin(to);
    return true;
  }
  // the pinned node's panel: who it came from, how it scored, what it changed and what grew from it
  function linScore(n) {
    const c = n.accepted && S.byId[n.id];
    if (c && num(c.mean)) return [score(c), 'val'];
    if (n.accepted && num(n.mean)) return [n.mean.toFixed(3), 'val'];
    // a proposals row, once loaded, carries the gate's successes and n, so a binary run reads 3/8 rather than 0.375
    const r = propOf(n.id);
    return [propScore(r || n), basisTag(r || n)];
  }
  let propMap = { d: null, m: new Map() };
  function propOf(id) {
    const d = S.prop.data;
    if (propMap.d !== d) propMap = { d, m: new Map(((d && d.rows) || []).map((r) => [r.id, r])) };
    return propMap.m.get(id) || null;
  }
  function linSummary(id) {
    const c = S.byId[id], r = propOf(id);
    return (c && c.note) || (r && r.summary) || null;
  }
  // an accepted node's objective means, the run's best per objective in cyan
  function objVector(n) {
    const objs = (S.lin.data && S.lin.data.objectives) || [], om = n.objective_means;
    if (!objs.length || !om) return null;
    const best = Object.fromEntries(((S.run && S.run.objective_bests) || []).map((o) => [o.name, o.mean]));
    return h('div', { class: 'lpv' }, objs.map((o) => {
      const v = om[o], top = num(v) && num(best[o]) && v >= best[o] - 1e-9;
      return [h('span', { text: o, title: o }), h('b', { class: top ? 'best' : null, text: ofmt(v) })];
    }));
  }
  function linPanel() {
    const V = $('v-lineage'), old = $('lp'), id = S.lin.pin, n = id && S.lin.byId[id], g = S.lin.g;
    if (!n || !g || !g.wrap.isConnected) { if (old) old.remove(); return; }
    const ix = linIndex(), pars = n.parents || [], o = outcomeOf(n), c = S.byId[id] || {};
    const go2 = (p) => h('button', { class: 'lnk mono', type: 'button', onclick: () => linPin(p), onmouseenter: () => { S.lin.hov = p; linFocus(); }, onmouseleave: () => { S.lin.hov = null; linFocus(); }, text: p });
    const from = pars.length > 1 ? [' · merge of ', pars.map((p, i) => [i ? ' + ' : '', S.lin.byId[p] ? go2(p) : p])]
      : pars.length ? [' · from ', S.lin.byId[pars[0]] ? go2(pars[0]) : pars[0]] : [' · ', n.operation === 'seed' || /^g0-/.test(id) ? 'seed' : 'no parent'];
    const [sv, tag] = linScore(n), rank = S.rank[id];
    const desc = ix.desc(id);
    let dAcc = 0;
    for (const v of desc) if (ix.by[v].accepted) dAcc++;
    const kids = (ix.kids[id] || []).map((k) => ix.by[k]).sort((a, b) => (a.generation || 0) - (b.generation || 0) || counter(a.id) - counter(b.id));
    const summary = linSummary(id);
    if (!S.prop.data && S.run) loadInto(S.prop, proposalsKey(), '/api/proposals').then((pd) => { if (pd && S.lin.pin === id && S.view === 'lineage') linPanel(); });
    const panel = h('div', { class: 'lp', id: 'lp', role: 'region', 'aria-label': 'Candidate ' + id },
      h('div', { class: 'lph' }, h('b', { class: 'mono', text: id }),
        h('button', { class: 'lpx', type: 'button', 'aria-label': 'Close', title: 'Close (Esc)', onclick: () => linPin(null), text: '×' })),
      h('div', { class: 'lps' }, 'generation ' + (n.generation || 0), from),
      h('div', { class: 'lpk' }, h('span', null, h('b', { text: sv }), tag ? h('span', { class: 'bt', text: tag }) : null),
        h('span', null, h('span', { class: 'o-' + o, text: o }), why(n) ? h('span', { class: 'rs', text: ' · ' + why(n) }) : null)),
      objVector(n),
      n.front ? h('div', { class: 'lps' }, [rank ? `#${rank} of ${S.front.length}` : null, num(c.keys) ? `${c.keys} keys` : null, num(c.sole) ? `${c.sole} held alone` : null].filter(Boolean).join(' · ')) : null,
      h('div', { class: 'lpn' + (summary ? '' : ' muted'), text: summary || (pars.length ? 'No summary.' : S.static.config.seedless ? 'Generated from the objective.' : 'Your repository as given.') }),
      h('div', { class: 'lps' }, h('b', { text: String(desc.size) }), plural(desc.size, ' descendant', ' descendants') + ' · ', h('b', { text: String(dAcc) }), ' accepted'),
      kids.length ? [h('h4', { text: 'Children' }), h('div', { class: 'lpc' }, kids.slice(0, 40).map((k) => {
        const [ks, kt] = linScore(k), ko = outcomeOf(k);
        return h('button', { type: 'button', class: 'lpr', onclick: () => linPin(k.id), onmouseenter: () => { S.lin.hov = k.id; linFocus(); }, onmouseleave: () => { S.lin.hov = null; linFocus(); } },
          h('span', { class: 'mono', text: k.id }), h('span', { class: 'o-' + ko, text: ko }), h('span', { class: 'n' }, ks, kt ? h('span', { class: 'bt', text: kt }) : null));
      })), kids.length > 40 ? h('div', { class: 'muted', text: `+${kids.length - 40} more` }) : null] : null,
      ID_RE.test(id) ? h('button', { class: 'lpo', type: 'button', 'data-open': id, text: 'Open candidate' }) : null);

    // the panel sits over the tree's right edge, or its left edge when the pinned node is under it
    const at = g.at.get(id), wide = 312;
    const top = Math.round(g.wrap.getBoundingClientRect().top - V.getBoundingClientRect().top + V.scrollTop);
    panel.style.top = top + 'px';
    panel.style.maxHeight = `calc(100% - ${top + 4}px)`;
    panel.classList.toggle('left', !!at && at[0] > g.W - wide - 24 && at[0] > wide + 24);
    if (old) old.replaceWith(panel); else V.appendChild(panel);
  }
  // proposals: one row per proposal in `helix log` order; past 500 rows only the visible slice is in the DOM
  async function drawProposals() {
    const V = $('v-proposals');
    if (!V.firstChild) fill(V, crumbs(['Loop', 'map'], ['Proposals', '']));
    await loadInto(S.prop, proposalsKey(), '/api/proposals');
    if (S.view === 'proposals') paintProposals(true);
  }
  function paintProposals(keepScroll) {
    const V = $('v-proposals'), P = S.prop, d = P.data, rows = (d && d.rows) || [], counts = (d && d.counts) || {};
    P.byId = {};
    P.order = new Map();
    rows.forEach((r, i) => { P.byId[r.id] = r; P.order.set(r.id, i); });
    const old = $('ptw'), top = keepScroll && old ? old.scrollTop : 0;
    const chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'outcome' }, ['all', ...OUTCOMES].map((o) => {
      const n = o === 'all' ? rows.length : num(counts[o]) ? counts[o] : rows.filter((r) => r.outcome === o).length;
      return h('button', {
        class: 'chip' + (P.filter === o ? ' on' : ''), type: 'button', 'aria-pressed': String(P.filter === o), disabled: !n && o !== 'all' && P.filter !== o,
        onclick: () => { P.filter = o; setHash(); paintProposals(false); },
      }, o === 'all' ? 'All' : o[0].toUpperCase() + o.slice(1), h('span', { class: 'n', text: String(n) }));
    }));
    const sh = (k, label) => h('button', {
      class: 'sh' + (P.sort === k ? ' on' : ''), type: 'button',
      onclick: () => { if (P.sort === k) P.dir = -P.dir; else { P.sort = k; P.dir = k === 'gen' ? 1 : -1; } paintProposals(false); },
    }, label, P.sort === k ? (P.dir > 0 ? ' ▴' : ' ▾') : '');
    // usage per proposal (runs with usage records): dollars, or tokens when the backend reports no cost
    P.use = rows.some((r) => r.usage) ? (rows.some((r) => r.usage && r.usage.cost_usd > 0) ? 'cost' : 'tokens') : null;
    if (!P.use && P.sort === 'cost') P.sort = 'gen';
    const cols = [92, 44, 132, 150, 100, 70, 44, 62, P.use === 'tokens' ? 84 : P.use ? 62 : 0, null].filter((w) => w !== 0);
    P.ncol = cols.length;
    const table = h('table', { class: 'pt' },
      h('colgroup', null, cols.map((w) => h('col', { style: { width: w ? w + 'px' : null } }))),
      h('thead', null, h('tr', null, h('th', { text: 'id' }), h('th', { class: 'n' }, sh('gen', 'gen')), h('th', { text: 'from' }), h('th', { text: 'outcome' }),
        h('th', { class: 'n' }, sh('score', 'score')), h('th', { class: 'n' }, sh('delta', 'Δ parent')), h('th', { class: 'n', text: 'files' }),
        h('th', { class: 'n', text: 'agent' }), P.use ? h('th', { class: 'n' }, sh('cost', P.use)) : null, h('th', { text: 'summary' }))),
      h('tbody', { id: 'ptb' }));
    const box = h('div', { class: 'ptw', id: 'ptw', tabindex: 0, 'aria-label': 'Proposals' }, table);
    let raf = 0;
    box.addEventListener('scroll', () => { if (P.rows.length > 500 && !raf) raf = requestAnimationFrame(() => { raf = 0; propBody(); }); });
    if (hoverEl && V.contains(hoverEl)) { hoverEl = null; hideCard(); }
    fill(V, crumbs(['Loop', 'map'], ['Proposals', '']), chips, box);
    const f = P.filter, ord = (r) => P.order.get(r.id);
    const list = rows.filter((r) => f === 'all' || r.outcome === f);
    const val = P.sort === 'score' ? pScore : P.sort === 'delta' ? (r) => (num(r.delta) ? r.delta : null)
      : P.sort === 'cost' ? (r) => (r.usage ? uVal(r.usage, P.use === 'cost') : null) : null;
    list.sort(val ? (a, b) => {
      const p = val(a), q = val(b);
      if (p == null || q == null) return (p == null) - (q == null) || ord(a) - ord(b);
      return P.dir * (p - q) || ord(a) - ord(b);
    } : (a, b) => P.dir * (ord(a) - ord(b)));
    P.rows = list;
    P.bodyKey = null;
    propBody();
    box.scrollTop = top;
    propBody();
  }
  function propBody() {
    const P = S.prop, box = $('ptw'), tb = $('ptb');
    if (!box || !tb) return;
    const rows = P.rows, rh = P.rh || 27;
    let a = 0, z = rows.length;
    if (rows.length > 500) {
      a = Math.max(0, Math.floor(box.scrollTop / rh) - 15);
      z = Math.min(rows.length, a + Math.ceil(box.clientHeight / rh) + 30);
    }
    const key = `${a}:${z}:${P.sel}`;
    if (key === P.bodyKey) return;
    P.bodyKey = key;
    const nc = P.ncol || 9, pad = (n) => (n > 0 ? h('tr', { class: 'sp', 'aria-hidden': 'true' }, h('td', { colspan: nc, style: { height: n * rh + 'px' } })) : null);
    fill(tb, pad(a), rows.length ? rows.slice(a, z).map(propRow) : h('tr', null, h('td', { colspan: nc, class: 'muted', text: '–' })), pad(rows.length - z));
  }
  function outcomeCell(r) {
    const o = outcomeOf(r), w = why(r);
    return h('td', { title: r.reason ? String(r.reason) : null }, h('span', { class: 'o-' + o, text: o }), w ? h('span', { class: 'rs', text: ' · ' + w }) : null);
  }
  function scoreCell(r) { const tag = basisTag(r); return h('td', { class: 'n' }, propScore(r), tag ? h('span', { class: 'bt', text: tag }) : null); }
  function propRow(r) {
    const open = ID_RE.test(r.id), sel = r.id === S.prop.sel, pars = r.parents || [];
    return h('tr', { class: [open ? 'cd' : null, sel ? 'sel' : null].filter(Boolean).join(' ') || null, 'data-c': open ? r.id : null, 'data-row': r.id, 'aria-selected': sel ? 'true' : null },
      h('td', { class: 'mono', text: open ? r.id : '–', title: open ? null : r.id }),
      h('td', { class: 'n', text: num(r.generation) ? String(r.generation) : '–' }),
      h('td', null, pars.length ? pars.map((p, i) => [i ? ' + ' : '', h('button', { class: 'lnk', type: 'button', 'data-open': p, text: p })]) : '–'),
      outcomeCell(r), scoreCell(r),
      h('td', { class: 'n' + (num(r.delta) && r.delta > 0 ? ' up' : ''), text: num(r.delta) ? (r.delta > 0 ? '+' : r.delta < 0 ? '−' : '±') + Math.abs(r.delta).toFixed(3) : '–' }),
      h('td', { class: 'n', text: num(r.files) ? String(r.files) : '–' }),
      h('td', { class: 'n', text: dur(r.agent_seconds) }),
      S.prop.use ? useCell(r.usage, S.prop.use === 'cost') : null,
      h('td', { class: 'sm', text: r.summary || '', title: r.summary || null }));
  }
  function useCell(u, cost) {
    if (!u) return h('td', { class: 'n', text: '–' });
    const tok = `${compact(u.input_tokens)} in · ${compact(u.output_tokens)} out` + (num(u.cached_input_tokens) && u.cached_input_tokens ? ` · ${compact(u.cached_input_tokens)} cached` : '');
    return h('td', { class: 'n', title: tok, text: cost ? money(u.cost_usd) : compact((u.input_tokens || 0) + (u.output_tokens || 0)) });
  }
  function propKey(e) {
    const P = S.prop, rows = P.rows || [];
    if (!rows.length || !['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter'].includes(e.key)) return false;
    let i = rows.findIndex((r) => r.id === P.sel);
    if (e.key === 'Enter') {
      if (i >= 0 && ID_RE.test(rows[i].id)) { $('card').style.display = 'none'; openCand(rows[i].id); }
      return i >= 0;
    }
    if (e.key === 'ArrowDown') i = i < 0 ? 0 : Math.min(rows.length - 1, i + 1);
    else if (e.key === 'ArrowUp') i = i < 0 ? 0 : Math.max(0, i - 1);
    else i = e.key === 'Home' ? 0 : rows.length - 1;
    P.sel = rows[i].id;
    const box = $('ptw'), head = box.querySelector('thead'), hh = head ? head.offsetHeight : 0, rh = P.rh || 27, top = i * rh;
    if (top < box.scrollTop) box.scrollTop = top;
    else if (top + rh + hh > box.scrollTop + box.clientHeight) box.scrollTop = top + rh + hh - box.clientHeight;
    propBody();
    return true;
  }

  // instance history: a small panel under the clicked instance, with its score in every accepted candidate
  function closeInst() { if (S.inst) { S.inst.el.remove(); S.inst = null; } }
  const instVal = (v) => (!num(v) ? '–' : binary() ? (v ? 'ok' : 'fail') : v.toFixed(2));
  async function openInst(iid, after, where) {
    const was = S.inst;
    closeInst();
    if (!after || (was && was.iid === iid && was.after === after)) return;
    const head = (...rest) => h('div', { class: 'iph' }, h('b', { class: 'mono', text: iid, title: iid }), ...rest);
    const el = h('div', { class: 'ip' }, head(h('span', { class: 'muted', text: '…' })));
    after.after(el);
    S.inst = { iid, el, after, where };
    let d = null, err = null;
    try { d = await get('/api/instance/' + encodeURIComponent(iid)); } catch (e) { err = e; }
    if (!S.inst || S.inst.el !== el) return;
    if (!d) { fill(el, head(h('span', { class: 'muted', text: err && err.status === 404 ? 'not found' : 'could not load' }))); return; }
    const hold = d.holders || [], sv = s('svg', { class: 'isp' });
    fill(el, head(h('span', null, 'best ', h('b', { text: instVal(d.best) })),
      hold.length ? h('span', { class: 'hl' }, 'held by ', hold.slice(0, 4).map((x, i) => [i ? ', ' : '', h('button', { class: 'lnk', type: 'button', 'data-open': x, text: x })]),
        hold.length > 4 ? ` +${hold.length - 4}` : '') : null), sv);
    instSpark(sv, d);
  }
  function instSpark(S0, d) {
    const pts = (d.points || []).filter((p) => num(p.score) && num(p.generation)).sort((a, b) => a.generation - b.generation);
    const W = S0.clientWidth || 360, Hh = 58, pl = 30, pr = 8, pt = 6, pb = 14;
    if (!pts.length) return;
    let G = 1, lo = Infinity, hi = -Infinity;
    for (const p of pts) { if (p.generation > G) G = p.generation; if (p.score < lo) lo = p.score; if (p.score > hi) hi = p.score; }
    if (binary()) { lo = 0; hi = 1; } else if (hi - lo < 1e-9) { lo = Math.min(lo, 0); hi = Math.max(hi, lo + 1); }
    const x = (g) => pl + g / G * (W - pl - pr), y = (v) => pt + (1 - (v - lo) / (hi - lo)) * (Hh - pt - pb);
    const C = { cyan: css('--cyan'), ink: css('--ink'), ink3: css('--ink3'), line: css('--line') };
    S0.appendChild(s('line', { x1: pl, x2: W - pr, y1: Hh - pb, y2: Hh - pb, stroke: C.line }));
    for (const v of [lo, hi]) S0.appendChild(s('text', { x: pl - 5, y: y(v) + 4, class: 'ax', 'text-anchor': 'end', text: binary() ? (v ? 'ok' : 'fail') : v.toFixed(2) }));
    for (const g of [0, G]) S0.appendChild(s('text', { x: x(g), y: Hh - 2, class: 'ax', 'text-anchor': 'middle', text: g }));
    let run = -Infinity, path = '';
    for (const p of pts) {
      if (p.score > run) { path += path ? `H${x(p.generation).toFixed(1)}V${y(p.score).toFixed(1)}` : `M${x(p.generation).toFixed(1)} ${y(p.score).toFixed(1)}`; run = p.score; }
    }
    path += `H${x(G).toFixed(1)}`;
    S0.appendChild(s('path', { d: path, fill: 'none', stroke: C.cyan, 'stroke-width': 1.5, opacity: 0.8 }));
    const r = pts.length > 150 ? 1.8 : 2.6, best = num(d.best) ? d.best : hi;
    for (const p of pts) {
      const cx = x(p.generation + (p.generation ? jitter(p.id) * 0.5 : 0)), cy = y(p.score);
      if (p.id === S.cand) S0.appendChild(s('circle', { cx, cy, r: r + 2.5, fill: 'none', stroke: C.ink, 'stroke-width': 1.2 }));
      S0.appendChild(s('circle', { cx, cy, r, fill: p.score >= best - 1e-12 ? C.cyan : C.ink3, opacity: 0.7, 'data-c': p.id, class: 'cd' }));
    }
  }

  // ---------------- wiring ----------------
  function render() {
    if (!S.static) return;
    index();
    renderHeader();
    renderChart();
    renderMap();
    renderBand();
    renderRail();
    renderView();
  }
  function setRun(r) {
    S.run = r || null;
    S.recvAt = nowSec();
    S.error = null;
    render();
  }
  let pollTimer = null;
  function startPoll() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try { setRun(await get('/api/run')); } catch (e) { if (e.status === 401 || e.status === 403) { S.error = 'Not authorised: open the link helix ui printed.'; renderHeader(); } }
    }, 3000);
  }
  function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
  function connect() {
    if (!window.EventSource) { startPoll(); return; }
    const es = new EventSource('/api/events');
    es.addEventListener('static', (e) => { try { S.static = JSON.parse(e.data); S.mapKey = null; S.viewKey = null; render(); } catch (err) { /* keep the last good copy */ } });
    es.addEventListener('run', (e) => { try { setRun(JSON.parse(e.data)); } catch (err) { /* keep the last good copy */ } });
    es.onopen = stopPoll;
    es.onerror = () => { startPoll(); };
  }
  function tick() {
    if (!S.static || !running()) return;
    renderHeader();
    renderRail();
  }
  function fromHash() {
    const hs = decodeURIComponent(location.hash.slice(1));
    if (hs.startsWith('candidate/') && ID_RE.test(hs.slice(10))) openCand(hs.slice(10));
    else if (hs === 'proposals' || hs.startsWith('proposals/') || hs.startsWith('lineage/')) go(hs);
    else if (VIEWS.includes(hs)) go(hs);
    else if (!hs && S.view !== 'map') go('map');
  }
  function bind() {
    document.addEventListener('mouseover', (e) => {
      const t = e.target.closest && e.target.closest('[data-c]');
      if (t === hoverEl) return;
      hoverEl = t;
      if (t) showCard(t); else hideCard();
    });
    document.addEventListener('click', (e) => {
      const t = e.target.closest && e.target.closest('[data-open],[data-go],[data-c],[data-lane],[data-inst]');
      if (!$('pop').contains(e.target) && e.target !== $('name')) togglePop(false);
      if (e.target.closest && e.target.closest('#objs')) objPop(!S.opop.pin || !$('opop').classList.contains('on'), !S.opop.pin);
      else if (!$('opop').contains(e.target) || t) objPop(false);
      if (!(e.target.closest && e.target.closest('#vbtn,#menu'))) toggleMenu(false);
      const row = e.target.closest && e.target.closest('tr[data-row]');
      if (row && S.view === 'proposals') { S.prop.sel = row.getAttribute('data-row'); propBody(); }
      if (!t) return;
      if (t.hasAttribute('data-inst')) openInst(t.getAttribute('data-inst'), t.closest('.low > div'), 'candidate');
      else if (t.hasAttribute('data-open')) { $('card').style.display = 'none'; openCand(t.getAttribute('data-open')); }
      else if (t.hasAttribute('data-go')) go(t.getAttribute('data-go'));
      else if (t.hasAttribute('data-c')) { $('card').style.display = 'none'; openCand(t.getAttribute('data-c')); }
      else if (t.hasAttribute('data-lane')) { S.lane = t.getAttribute('data-lane'); renderRail(); }
    });
    document.addEventListener('keydown', (e) => {
      if ($('menu').classList.contains('on')) {
        const items = [...$('menu').children], i = items.indexOf(document.activeElement);
        if (e.key === 'Escape') { e.preventDefault(); toggleMenu(false, true); }
        else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
          e.preventDefault();
          const j = e.key === 'Home' ? 0 : e.key === 'End' ? items.length - 1 : i < 0 ? 0 : (i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
          items[j].focus();
        } else if (e.key === 'Tab') toggleMenu(false);
        return;
      }
      if (e.key === 'Escape') {
        if ($('opop').classList.contains('on')) objPop(false);
        else if ($('pop').classList.contains('on')) togglePop(false);
        else if (S.inst) closeInst();
        else if (S.view === 'lineage' && S.lin.pin) linPin(null);
        else go('map');
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target && e.target.tagName) || '';
      if (e.key === 'ArrowDown' && e.target === $('vbtn')) { e.preventDefault(); toggleMenu(true); return; }
      if (S.view === 'lineage' && S.lin.pin && /^Arrow/.test(e.key) && !/^(INPUT|SELECT|TEXTAREA)$/.test(tag) && linKey(e.key)) { e.preventDefault(); return; }
      if (S.view === 'proposals' && !/^(INPUT|SELECT|TEXTAREA)$/.test(tag) && !(e.key === 'Enter' && /^(BUTTON|A|SUMMARY)$/.test(tag)) && propKey(e)) { e.preventDefault(); return; }
      const g = (e.key === 'Enter' || e.key === ' ') && e.target.closest && e.target.closest('[data-go][role=button]');
      if (g && tag !== 'BUTTON') { e.preventDefault(); go(g.getAttribute('data-go')); }
    });
    $('name').addEventListener('click', () => togglePop());
    $('psel').addEventListener('change', () => { S.pobj = $('psel').value || null; store.set('helix-ui-progress', S.pobj || ''); chartLeave(); renderChart(); });
    // the objectives popover opens on hover and stays open on click
    const hd = document.querySelector('header'), inObj = (e) => e.target.closest && e.target.closest('#objs,#opop');
    hd.addEventListener('mouseover', (e) => {
      if (inObj(e)) { clearTimeout(S.opop.timer); if (!$('opop').classList.contains('on')) objPop(true); }
      else if (!S.opop.pin) { clearTimeout(S.opop.timer); S.opop.timer = setTimeout(() => objPop(false), 180); }
    });
    hd.addEventListener('mouseleave', () => { if (!S.opop.pin) { clearTimeout(S.opop.timer); S.opop.timer = setTimeout(() => objPop(false), 180); } });
    $('vbtn').addEventListener('click', () => toggleMenu());
    // the chart's background opens the lineage; its dots and rungs still open their candidate
    $('pc').addEventListener('click', (e) => { if (S.run && !(e.target.closest && e.target.closest('[data-c]'))) go('lineage'); });
    $('tog').addEventListener('click', () => {
      const on = $('tog').classList.toggle('on');
      $('tog').setAttribute('aria-pressed', String(on));
      $('tx').classList.toggle('nocmd', !on);
    });
    $('pc').addEventListener('mousemove', chartMove);
    $('pc').addEventListener('mouseleave', chartLeave);
    let rz = null;
    addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { renderChart(); renderBand(); renderView(true); }, 120); });
    const rerender = () => { S.mapKey = null; S.viewKey = null; render(); };
    dark.addEventListener('change', rerender);
    reduceMotion.addEventListener('change', rerender);
    addEventListener('hashchange', fromHash);
  }
  async function boot() {
    bind();
    try {
      const [st, run] = await Promise.all([get('/api/static'), get('/api/run')]);
      S.static = st;
      setRun(run);
    } catch (e) {
      fill($('stats'), h('span', { class: 'note-err', text: e.status === 401 || e.status === 403 ? 'Not authorised: open the link helix ui printed.' : 'Could not reach helix ui.' }));
      return;
    }
    fromHash();
    connect();
    setInterval(tick, 1000);
    requestAnimationFrame(frame);
  }
  boot();
})();
