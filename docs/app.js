/* Abalone PCA Experiment Lab: eight model experiments, no backend. */
(() => {
  'use strict';

  const COLORS = {
    'XGBoost': '#147b75',
    'Extra Trees': '#4278b3',
    'Random Forest': '#bc7e37',
    'Linear Regression': '#8b65aa',
  };
  const SCENARIOS = [
    { id: 'all', title: 'All samples', min: null, max: null },
    { id: 'young', title: 'Rings <= 8', min: null, max: 8 },
    { id: 'nine-ten', title: 'Rings 9-10', min: 9, max: 10 },
    { id: 'eleven-twelve', title: 'Rings 11-12', min: 11, max: 12 },
    { id: 'thirteen-fifteen', title: 'Rings 13-15', min: 13, max: 15 },
    { id: 'old', title: 'Rings >= 16', min: 16, max: null },
  ];
  const DESC = {
    MAE: 'Ranked by mean absolute error. Lower is better.',
    RMSE: 'Ranked by root mean squared error, which penalizes large misses. Lower is better.',
    MedAE: 'Ranked by median absolute error. Lower is better.',
    R2: 'Ranked by R-squared relative to the mean of actual rings in this subset. Higher is better.',
  };
  const $ = (s) => document.querySelector(s);
  const make = (tag, className = '', content = '') => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== '') node.textContent = String(content);
    return node;
  };
  const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
  const color = (name, index) => COLORS[name] || ['#147b75', '#4278b3', '#bc7e37', '#8b65aa'][index % 4];
  let activeSetting = 'PCA';
  const modelCol = (name, setting = activeSetting) => setting === 'PCA' ? name + '_predicted_rings' : 'NoPCA_' + name + '_predicted_rings';
  const settingLabel = setting => setting === 'PCA' ? 'PCA ON · 2 PCs' : 'PCA OFF · 9 encoded features';
  const absError = (row, name) => Math.abs(Number(row[modelCol(name)]) - Number(row.actual_rings));
  const metricBetter = (a, b, metric) => metric === 'R2' ? a > b : a < b;
  const minVal = (a) => a.reduce((x, y) => Math.min(x, y), Infinity);
  const maxVal = (a) => a.reduce((x, y) => Math.max(x, y), -Infinity);
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

  async function init() {
    const response = await fetch('./results.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error('Could not load docs/results.json (HTTP ' + response.status + ').');
    const data = await response.json();
    if (data.schemaVersion !== 3) throw new Error('Website requires the five NEW CSV exports. Run python update_site_data.py.');
    const names = (data.models || []).map(String);
    const rows = data.predictions || [];
    if (names.length < 2 || !rows.length) throw new Error('Result data has no models or test predictions.');
    for (const setting of ['PCA', 'NoPCA']) {
      for (const name of names) {
        if (!rows.every(r => Number.isFinite(Number(r[modelCol(name, setting)])))) {
          throw new Error('Missing or invalid prediction column: ' + modelCol(name, setting));
        }
      }
    }
    const hasFeatures = Boolean(data.metadata && data.metadata.hasFeatures && rows.every(r => r.Sex != null && Number.isFinite(Number(r.Length))));
    const quantiles = data.metadata && data.metadata.lengthQuantiles;
    const rings = rows.map(r => Number(r.actual_rings));
    const globalMin = Math.floor(minVal(rings));
    const globalMax = Math.ceil(maxVal(rings));
    const state = {
      min: globalMin, max: globalMax, metric: 'MAE', sex: 'all', length: 'all', setting: 'PCA',
      plot: names.includes('XGBoost') ? 'XGBoost' : names[0],
      duelA: names.includes('XGBoost') ? 'XGBoost' : names[0],
      duelB: names.includes('Linear Regression') ? 'Linear Regression' : names[1],
    };
    if (state.duelA === state.duelB) state.duelB = names.find(n => n !== state.duelA);
    const rangeMin = $('#rings-min');
    const rangeMax = $('#rings-max');
    for (const input of [rangeMin, rangeMax]) {
      input.min = String(globalMin);
      input.max = String(globalMax);
    }
    rangeMin.value = String(state.min);
    rangeMax.value = String(state.max);
    const q1 = quantiles && Number(quantiles.q1);
    const q3 = quantiles && Number(quantiles.q3);
    const featureFilters = hasFeatures && Number.isFinite(q1) && Number.isFinite(q3);
    $('#sex-select').disabled = !hasFeatures;
    $('#length-select').disabled = !featureFilters;
    $('#length-caption').textContent = featureFilters
      ? 'Small < ' + fmt(q1, 3) + '  /  Large > ' + fmt(q3, 3) + ' (shell length).'
      : 'Optional: provide a matching cleaned CSV when refreshing results.';
    $('#metric-select').value = state.metric;

    function options(select, selected) {
      select.replaceChildren();
      for (const name of names) {
        const opt = make('option', '', name);
        opt.value = name;
        select.append(opt);
      }
      select.value = selected;
    }
    options($('#duel-a'), state.duelA);
    options($('#duel-b'), state.duelB);
    options($('#plot-model'), state.plot);
    options($('#pair-a'), state.duelA);
    options($('#pair-b'), state.duelB);

    function filterData(min, max, respectRing = true) {
      return rows.filter(r => {
        const real = Number(r.actual_rings);
        if (respectRing && (real < min || real > max)) return false;
        if (state.sex !== 'all' && r.Sex !== state.sex) return false;
        if (state.length !== 'all' && featureFilters) {
          const length = Number(r.Length);
          if (state.length === 'small' && !(length < q1)) return false;
          if (state.length === 'mid' && !(length >= q1 && length <= q3)) return false;
          if (state.length === 'large' && !(length > q3)) return false;
        }
        return true;
      });
    }
    function stats(subset, name) {
      if (!subset.length) return { MAE: null, RMSE: null, MedAE: null, R2: null };
      const actual = subset.map(r => Number(r.actual_rings));
      const errors = subset.map(r => absError(r, name));
      const sorted = [...errors].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      const med = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      const ssRes = errors.reduce((sum, e) => sum + e * e, 0);
      const avg = mean(actual);
      const ssTot = actual.reduce((sum, v) => sum + (v - avg) ** 2, 0);
      return {
        MAE: mean(errors),
        RMSE: Math.sqrt(ssRes / subset.length),
        MedAE: med,
        R2: subset.length >= 2 && ssTot > 1e-12 ? 1 - ssRes / ssTot : null,
      };
    }
    function ranked(subset) {
      return names.map((name, i) => ({ name, color: color(name, i), score: stats(subset, name)[state.metric] }))
        .sort((a, b) => {
          if (a.score === null) return b.score === null ? 0 : 1;
          if (b.score === null) return -1;
          return state.metric === 'R2' ? b.score - a.score : a.score - b.score;
        });
    }
    function selectedScenario() {
      return SCENARIOS.find(s => (s.min ?? globalMin) === state.min && (s.max ?? globalMax) === state.max)?.id ?? null;
    }
    function makeScenarioButton(s) {
      const lo = s.min ?? globalMin;
      const hi = s.max ?? globalMax;
      const subset = filterData(lo, hi);
      const result = ranked(subset)[0];
      const b = make('button', 'scenario' + (selectedScenario() === s.id ? ' active' : ''));
      b.type = 'button';
      b.setAttribute('aria-pressed', selectedScenario() === s.id ? 'true' : 'false');
      b.append(make('div', 'scenario-title', s.title));
      b.append(make('div', 'scenario-winner', result && result.score !== null ? result.name : 'Not available'));
      b.append(make('div', 'scenario-count', subset.length + ' samples' + (result && result.score !== null ? '  /  ' + fmt(result.score) : '')));
      b.addEventListener('click', () => {
        state.min = lo;
        state.max = hi;
        rangeMin.value = String(lo);
        rangeMax.value = String(hi);
        render();
      });
      return b;
    }
    function renderScenarios() {
      $('#scenario-rail').replaceChildren(...SCENARIOS.map(makeScenarioButton));
    }
    function rankRow(row, i, subset) {
      const b = make('button', 'rank-row' + (i === 0 && row.score !== null ? ' winner' : ''));
      b.type = 'button';
      b.title = 'Click to inspect ' + row.name + ' predictions';
      b.disabled = !subset.length;
      b.append(make('div', 'rank-index', '#' + (i + 1)));
      const middle = make('div');
      const name = make('div', 'rank-name');
      const dot = make('span', 'model-dot');
      dot.style.background = row.color;
      name.append(dot, document.createTextNode(row.name));
      middle.append(name);
      const extra = i === 0 && row.score !== null ? 'Current leader' : 'Inspect predictions';
      middle.append(make('div', 'rank-extra', extra));
      b.append(middle);
      const score = make('div', 'rank-value', fmt(row.score, 4));
      score.append(make('small', '', state.metric === 'R2' ? 'higher is better' : 'lower is better'));
      b.append(score);
      b.addEventListener('click', () => {
        state.plot = row.name;
        $('#plot-model').value = row.name;
        renderPlot(subset);
        renderSamples(subset);
        $('#scatter-title').scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
      return b;
    }
    function renderRankings(subset, ranks) {
      $('#sample-badge').textContent = subset.length + ' / ' + rows.length + ' samples';
      $('#metric-note').textContent = DESC[state.metric];
      $('#rank-list').replaceChildren(...ranks.map((r, i) => rankRow(r, i, subset)));
      const alert = $('#low-count-alert');
      alert.hidden = true;
      if (!subset.length) {
        alert.hidden = false;
        alert.textContent = 'No samples match this combination.';
        return;
      }
      if (subset.length < 30) {
        alert.hidden = false;
        alert.textContent = 'Caution: only ' + subset.length + ' samples. Rankings can change substantially with such a small subset.';
      }
    }
    function renderHeatmap() {
      const head = make('tr');
      head.append(make('th', '', 'Rings'));
      head.append(make('th', '', 'n'));
      for (const name of names) head.append(make('th', '', name));
      $('#heatmap-head').replaceChildren(head);
      const body = $('#heatmap-body');
      body.replaceChildren();
      for (const s of SCENARIOS) {
        const lo = s.min ?? globalMin;
        const hi = s.max ?? globalMax;
        const subset = filterData(lo, hi);
        const scores = Object.fromEntries(names.map(n => [n, stats(subset, n)[state.metric]]));
        const vals = names.map(n => scores[n]).filter(Number.isFinite);
        const best = vals.length ? (state.metric === 'R2' ? maxVal(vals) : minVal(vals)) : null;
        const tr = make('tr', selectedScenario() === s.id ? 'current' : '');
        const first = make('td');
        const b = make('button', '', s.title.replace('Rings ', ''));
        b.type = 'button';
        b.addEventListener('click', () => {
          state.min = lo;
          state.max = hi;
          rangeMin.value = String(lo);
          rangeMax.value = String(hi);
          render();
          $('#leaderboard-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        first.append(b);
        tr.append(first, make('td', '', subset.length));
        for (const name of names) {
          const value = scores[name];
          const td = make('td', value !== null && best !== null && Math.abs(value - best) < 1e-10 ? 'best' : '', fmt(value, 3));
          tr.append(td);
        }
        body.append(tr);
      }
    }
    function renderDuel(subset) {
      const a = state.duelA;
      const b = state.duelB;
      $('#duel-a-bar').style.background = color(a, names.indexOf(a));
      $('#duel-b-bar').style.background = color(b, names.indexOf(b));
      $('#duel-a-label').textContent = a;
      $('#duel-b-label').textContent = b;
      const aBar = $('#duel-a-bar');
      const bBar = $('#duel-b-bar');
      const tieBar = $('#duel-tie-bar');
      if (!subset.length || a === b) {
        $('#duel-result').textContent = a === b ? 'Choose two different models to compare.' : 'No matching test samples.';
        aBar.style.width = '0%'; bBar.style.width = '0%'; tieBar.style.width = '100%';
        return;
      }
      let winsA = 0, winsB = 0, ties = 0;
      for (const r of subset) {
        const delta = absError(r, a) - absError(r, b);
        if (Math.abs(delta) < 1e-8) ties++;
        else if (delta < 0) winsA++;
        else winsB++;
      }
      const pct = n => (n / subset.length * 100);
      aBar.style.width = pct(winsA).toFixed(2) + '%';
      bBar.style.width = pct(winsB).toFixed(2) + '%';
      tieBar.style.width = pct(ties).toFixed(2) + '%';
      $('#duel-result').textContent = a + ': ' + winsA + ' wins (' + fmt(pct(winsA), 1) + '%)  /  '
        + b + ': ' + winsB + ' wins (' + fmt(pct(winsB), 1) + '%)'
        + (ties ? '  /  ' + ties + ' ties' : '');
    }

    const NS = 'http://www.w3.org/2000/svg';
    function svgNode(tag, attrs = {}, text = null) {
      const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
      if (text !== null) n.textContent = String(text);
      return n;
    }
    function renderPlot(subset) {
      const svg = $('#scatter-svg');
      svg.replaceChildren();
      const inset = { left: 49, right: 18, top: 18, bottom: 46 };
      const W = 640, H = 310, PW = W - inset.left - inset.right, PH = H - inset.top - inset.bottom;
      const yValues = subset.map(r => Number(r[modelCol(state.plot)]));
      const xValues = subset.map(r => Number(r.actual_rings));
      const both = xValues.concat(yValues);
      const low = both.length ? Math.floor(minVal(both) - .5) : 0;
      const high = both.length ? Math.ceil(maxVal(both) + .5) : 25;
      const lo = Math.min(low, high - 2);
      const hi = Math.max(high, low + 2);
      const sx = v => inset.left + ((v - lo) / (hi - lo)) * PW;
      const sy = v => inset.top + PH - ((v - lo) / (hi - lo)) * PH;
      const ticks = 5;
      for (let i = 0; i <= ticks; i++) {
        const value = lo + (hi - lo) * i / ticks;
        const x = sx(value), y = sy(value);
        svg.append(svgNode('line', { x1: x, y1: inset.top, x2: x, y2: inset.top + PH, class: 'gridline' }));
        svg.append(svgNode('line', { x1: inset.left, y1: y, x2: inset.left + PW, y2: y, class: 'gridline' }));
        svg.append(svgNode('text', { x: x, y: H - 26, 'text-anchor': 'middle', class: 'axis-label' }, fmt(value, 0)));
        svg.append(svgNode('text', { x: inset.left - 10, y: y + 4, 'text-anchor': 'end', class: 'axis-label' }, fmt(value, 0)));
      }
      svg.append(svgNode('line', { x1: inset.left, y1: inset.top + PH, x2: inset.left + PW, y2: inset.top + PH, class: 'axis-line' }));
      svg.append(svgNode('line', { x1: inset.left, y1: inset.top, x2: inset.left, y2: inset.top + PH, class: 'axis-line' }));
      svg.append(svgNode('line', { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: 'perfect' }));
      svg.append(svgNode('text', { x: inset.left + PW / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis-label' }, 'Actual rings'));
      svg.append(svgNode('text', { x: 14, y: inset.top + PH / 2, 'text-anchor': 'middle', class: 'axis-label', transform: 'rotate(-90 14 ' + (inset.top + PH / 2) + ')' }, 'Predicted rings'));
      // The points represent saved predictions, not inference from user-entered data.
      for (const row of subset) {
        const actual = Number(row.actual_rings);
        const predicted = Number(row[modelCol(state.plot)]);
        const e = Math.abs(predicted - actual);
        const circ = svgNode('circle', {
          cx: sx(actual), cy: sy(predicted), r: 3.2,
          fill: e <= 2 ? '#1c9186' : '#df9960', 'fill-opacity': .56,
          stroke: 'white', 'stroke-width': .55, tabindex: 0, role: 'button',
          'aria-label': 'Row ' + row.original_row_index + ', actual ' + actual + ', predicted ' + fmt(predicted, 2),
        });
        circ.append(svgNode('title', {}, 'Row ' + row.original_row_index + ' / Actual: ' + actual + ' / Predicted: ' + fmt(predicted, 2) + ' / Error: ' + fmt(e, 2)));
        circ.addEventListener('mouseenter', () => {
          $('#plot-inspector').textContent = 'Row ' + row.original_row_index + '  /  Actual ' + actual + '  /  Predicted ' + fmt(predicted, 2) + '  /  Error ' + fmt(e, 2);
        });
        const inspect = () => {
          $('#row-search').value = String(row.original_row_index);
          renderSamples(subset);
          $('#samples-title').scrollIntoView({ behavior: 'smooth', block: 'center' });
        };
        circ.addEventListener('click', inspect);
        circ.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); inspect(); } });
        svg.append(circ);
      }
      $('#plot-inspector').textContent = subset.length
        ? 'Hover to inspect, click a dot to find that observation in the table below.'
        : 'No observations match your filters.';
    }
    function rowWinner(row) {
      const scores = names.map((name, i) => ({ name, index: i, error: absError(row, name) }));
      const best = minVal(scores.map(s => s.error));
      return scores.filter(s => Math.abs(s.error - best) < 1e-8).map(s => s.name).join(' / ');
    }
    function renderSamples(subset) {
      const term = $('#row-search').value.trim();
      let shown = subset.filter(r => !term || String(r.original_row_index).includes(term));
      const sortMode = $('#sort-samples').value;
      const error = r => absError(r, state.plot);
      const disagree = r => maxVal(names.map(n => Number(r[modelCol(n)]))) - minVal(names.map(n => Number(r[modelCol(n)])));
      shown = [...shown].sort((a, b) => {
        if (sortMode === 'worst') return error(b) - error(a);
        if (sortMode === 'best') return error(a) - error(b);
        if (sortMode === 'disagree') return disagree(b) - disagree(a);
        return Number(a.original_row_index) - Number(b.original_row_index);
      });
      const limit = 15;
      const tbody = $('#sample-body');
      tbody.replaceChildren();
      for (const r of shown.slice(0, limit)) {
        const tr = make('tr', term && String(r.original_row_index) === term ? 'highlighted' : '');
        const predicted = Number(r[modelCol(state.plot)]);
        const entries = [r.original_row_index, r.Sex ?? '\u2014', r.actual_rings, fmt(predicted, 2), fmt(error(r), 2), rowWinner(r)];
        for (const entry of entries) tr.append(make('td', '', entry));
        tbody.append(tr);
      }
      if (!shown.length) {
        const tr = make('tr');
        const td = make('td', '', 'No test samples match this search and the active filters.');
        td.colSpan = 6;
        tr.append(td);
        tbody.append(tr);
      }
      $('#sample-status').textContent = 'Showing ' + Math.min(shown.length, limit) + ' of ' + shown.length + ' matching samples. Row winner = smallest absolute error on that observation.';
    }
    /* V3 extra visualization functions injected below. */
    const metricsMap = Object.fromEntries((data.metrics || []).map(r => [r.Setting + '|' + r.Model, r]));
    const cvMap = Object.fromEntries((data.cv || []).map(r => [r.Setting + '|' + r.Model, r]));
    const ciMap = Object.fromEntries((data.bootstrap || []).map(r => [r.Setting + '|' + r.Model, r]));
    const modelMetric = (setting, model) => metricsMap[setting + '|' + model];

    function pairedResult(sa, a, sb, b) {
      const direct = data.paired.find(r => r.Setting_A === sa && r.Model_A === a && r.Setting_B === sb && r.Model_B === b);
      if (direct) return direct;
      const reverse = data.paired.find(r => r.Setting_A === sb && r.Model_A === b && r.Setting_B === sa && r.Model_B === a);
      if (!reverse) return null;
      return {
        ...reverse,
        MAE_Difference_A_minus_B: -reverse.MAE_Difference_A_minus_B,
        CI_Lower_95: -reverse.CI_Upper_95,
        CI_Upper_95: -reverse.CI_Lower_95,
      };
    }

    function renderModeControls() {
      for (const [selector, selected] of [['#setting-pca', activeSetting === 'PCA'], ['#setting-nopca', activeSetting === 'NoPCA']]) {
        const button = $(selector);
        button.classList.toggle('active', selected);
        button.setAttribute('aria-pressed', selected ? 'true' : 'false');
      }
      $('#filter-setting-name').textContent = settingLabel(activeSetting);
      $('#current-setting-label').textContent = 'Viewing ' + settingLabel(activeSetting) + ' · ' + rows.length + ' held-out test samples';
      $('#ci-setting-caption').textContent = settingLabel(activeSetting);
    }

    function renderPcaImpact() {
      // Only draw the selected feature setting. Switching PCA ON/OFF changes
      // the visible values and bars without showing the inactive counterpart.
      const panel = $('#impact-list');
      panel.replaceChildren();
      const maxMAE = Math.max(...data.metrics.map(row => Number(row.MAE))) * 1.05;
      for (const [i, name] of names.entries()) {
        const metric = modelMetric(activeSetting, name);
        if (!metric) continue;
        const value = Number(metric.MAE);
        const container = make('div', 'impact-row');
        const title = make('div', 'model-name', name);
        title.style.color = color(name, i);
        const track = make('div', 'mini-bar-track');
        const fill = make('span', 'mini-bar-fill');
        fill.style.background = color(name, i);
        fill.style.width = (100 * value / maxMAE).toFixed(2) + '%';
        track.append(fill);
        const score = make('span', 'mini-bar-value', fmt(value, 3));
        container.append(title, track, score);
        panel.append(container);
      }
    }
    function renderConfidence() {
      const items = names.map((name,i) => ({ name, i, ci: ciMap[activeSetting + '|' + name] })).filter(x => x.ci);
      const scaleMin = Math.min(...items.map(x => Number(x.ci.MAE_CI_Lower_95))) - .06;
      const scaleMax = Math.max(...items.map(x => Number(x.ci.MAE_CI_Upper_95))) + .06;
      const pct = n => 100 * (Number(n) - scaleMin) / Math.max(.001,scaleMax - scaleMin);
      const area = $('#ci-list');
      area.replaceChildren();
      for (const item of items) {
        const ci = item.ci, box = make('div','ci-row');
        box.append(make('div','ci-name',item.name));
        const track = make('div','ci-track');
        track.title = item.name + ': MAE ' + fmt(ci.Test_MAE,4) + ', 95% CI [' + fmt(ci.MAE_CI_Lower_95,4) + ', ' + fmt(ci.MAE_CI_Upper_95,4) + ']';
        const whisker = make('span','ci-whisker');
        whisker.style.background = color(item.name,item.i);
        whisker.style.left = pct(ci.MAE_CI_Lower_95) + '%';
        whisker.style.width = (pct(ci.MAE_CI_Upper_95)-pct(ci.MAE_CI_Lower_95)) + '%';
        const point = make('span','ci-point');
        point.style.left = pct(ci.Test_MAE) + '%';
        point.style.background = color(item.name,item.i);
        track.append(whisker,point);
        box.append(track, make('div','ci-number',fmt(ci.Test_MAE,3)));
        box.append(make('div','ci-range-label','['+fmt(ci.MAE_CI_Lower_95,3)+', '+fmt(ci.MAE_CI_Upper_95,3)+']'));
        area.append(box);
      }
      renderPair();
      renderPcaPaired();
    }

    function renderPair() {
      const a = $('#pair-a').value, b = $('#pair-b').value;
      const note = $('#pair-judgment');
      note.className = 'pair-judgment';
      if (a === b) {
        $('#pair-number').textContent = '—';
        $('#pair-interval').textContent = '95% CI: —';
        note.textContent = 'Choose two different models.';
        return;
      }
      const result = pairedResult(activeSetting,a,activeSetting,b);
      if (!result) {
        $('#pair-number').textContent = '—';
        $('#pair-interval').textContent = '95% CI: —';
        note.textContent = 'No paired result exists for this comparison.';
        return;
      }
      const diff = Number(result.MAE_Difference_A_minus_B);
      $('#pair-number').textContent = (diff > 0 ? '+' : '') + fmt(diff,4) + ' rings';
      $('#pair-interval').textContent = '95% CI [' + fmt(result.CI_Lower_95,4) + ', ' + fmt(result.CI_Upper_95,4) + ']';
      if (Number(result.CI_Lower_95) > 0) {
        note.textContent = b + ' has lower MAE; the paired interval excludes zero.';
        note.classList.add('clear');
      } else if (Number(result.CI_Upper_95) < 0) {
        note.textContent = a + ' has lower MAE; the paired interval excludes zero.';
        note.classList.add('clear');
      } else {
        note.textContent = 'Uncertain comparison: the paired interval includes zero.';
        note.classList.add('uncertain');
      }
    }

    function renderPcaPaired() {
      const target = $('#pca-paired-body');
      target.replaceChildren();
      for (const name of names) {
        const result = pairedResult('PCA',name,'NoPCA',name);
        if (!result) continue;
        const difference = Number(result.MAE_Difference_A_minus_B);
        let label = 'No clear difference';
        if (Number(result.CI_Lower_95) > 0) label = 'No PCA lower MAE';
        if (Number(result.CI_Upper_95) < 0) label = 'PCA lower MAE';
        const tr = make('tr');
        for (const val of [name,(difference>0?'+':'')+fmt(difference,4),'['+fmt(result.CI_Lower_95,4)+', '+fmt(result.CI_Upper_95,4)+']',label]) tr.append(make('td','',val));
        target.append(tr);
      }
    }

    function renderBenchmark() {
      const target = $('#benchmark-body');
      target.replaceChildren();
      for (const setting of ['PCA','NoPCA']) {
        for (const name of names) {
          const test = modelMetric(setting,name), cross = cvMap[setting+'|'+name];
          const tr = make('tr',activeSetting === setting ? 'highlighted' : '');
          for (const val of [settingLabel(setting),name,fmt(Number(cross.CV_MAE),4),fmt(Number(test.MAE),4),fmt(Number(test.RMSE),4),fmt(Number(test.R2),4)]) tr.append(make('td','',val));
          target.append(tr);
        }
      }
    }

    function render() {
      renderModeControls();
      renderPcaImpact();
      renderConfidence();
      renderBenchmark();
      $('#rings-min-value').textContent = String(state.min);
      $('#rings-max-value').textContent = String(state.max);
      $('#ring-description').textContent = state.min + ' - ' + state.max;
      const subset = filterData(state.min, state.max);
      const ranks = ranked(subset);
      renderScenarios();
      renderRankings(subset, ranks);
      renderHeatmap();
      renderDuel(subset);
      renderPlot(subset);
      renderSamples(subset);
    }
    function exportSubset() {
      const subset = filterData(state.min, state.max);
      if (!subset.length) return;
      const cols = ['original_row_index', 'actual_rings', 'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'selected_setting'];
      for (const setting of ['PCA','NoPCA']) for (const name of names) cols.push(modelCol(name, setting), setting + '_' + name + '_absolute_error');
      cols.push('selected_setting_row_winner');
      const quote = value => '"' + String(value == null ? '' : value).replace(/"/g, '""') + '"';
      const csvRows = [cols.map(quote).join(',')];
      for (const r of subset) {
        const out = { ...r, selected_setting: activeSetting, selected_setting_row_winner: rowWinner(r) };
        for (const setting of ['PCA','NoPCA']) for (const name of names) out[setting + '_' + name + '_absolute_error'] = Math.abs(Number(r[modelCol(name, setting)]) - Number(r.actual_rings));
        csvRows.push(cols.map(k => quote(out[k])).join(','));
      }
      const blob = new Blob([csvRows.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'abalone_' + activeSetting.toLowerCase() + '_filtered_predictions.csv';
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 300);
    }
    const setSetting = setting => { state.setting = setting; activeSetting = setting; render(); };
    $('#setting-pca').addEventListener('click', () => setSetting('PCA'));
    $('#setting-nopca').addEventListener('click', () => setSetting('NoPCA'));
    $('#pair-a').addEventListener('change', renderPair);
    $('#pair-b').addEventListener('change', renderPair);
    $('#metric-select').addEventListener('change', e => { state.metric = e.target.value; render(); });
    rangeMin.addEventListener('input', e => {
      state.min = Number(e.target.value);
      if (state.min > state.max) { state.max = state.min; rangeMax.value = e.target.value; }
      render();
    });
    rangeMax.addEventListener('input', e => {
      state.max = Number(e.target.value);
      if (state.max < state.min) { state.min = state.max; rangeMin.value = e.target.value; }
      render();
    });
    $('#sex-select').addEventListener('change', e => { state.sex = e.target.value; render(); });
    $('#length-select').addEventListener('change', e => { state.length = e.target.value; render(); });
    $('#duel-a').addEventListener('change', e => { state.duelA = e.target.value; renderDuel(filterData(state.min, state.max)); });
    $('#duel-b').addEventListener('change', e => { state.duelB = e.target.value; renderDuel(filterData(state.min, state.max)); });
    $('#plot-model').addEventListener('change', e => {
      state.plot = e.target.value;
      const subset = filterData(state.min, state.max);
      renderPlot(subset); renderSamples(subset);
    });
    $('#row-search').addEventListener('input', () => renderSamples(filterData(state.min, state.max)));
    $('#sort-samples').addEventListener('change', () => renderSamples(filterData(state.min, state.max)));
    $('#reset-button').addEventListener('click', () => {
      state.min = globalMin; state.max = globalMax; state.metric = 'MAE'; state.sex = 'all'; state.length = 'all';
      $('#metric-select').value = 'MAE';
      rangeMin.value = String(globalMin); rangeMax.value = String(globalMax);
      $('#sex-select').value = 'all'; $('#length-select').value = 'all'; $('#row-search').value = '';
      render();
    });
    $('#export-button').addEventListener('click', exportSubset);
    render();
  }

  document.addEventListener('DOMContentLoaded', () => {
    init().catch(err => {
      const error = document.querySelector('#error');
      error.hidden = false;
      error.textContent = 'Unable to load the interactive dashboard: ' + err.message
        + '  Run update_site_data.py with your latest notebook CSV exports, then open the site via a local HTTP server or GitHub Pages.';
      console.error(err);
    });
  });
})();
