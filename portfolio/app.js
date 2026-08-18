/* ==========================================================================
   TTFT-RCA dashboard
   Renders portfolio-data.json, which is generated from the committed traces by
   `python -m rca_engine.scripts.gen_portfolio_data`. Nothing on this page is
   hand-authored: if a number here is wrong, the pipeline produced it.
   ========================================================================== */

const SVG = "http://www.w3.org/2000/svg";

const COLOR = {
  signal: "#5eddc6",
  sli: "#f0b849",
  exo: "#8b7cf0",
  warn: "#f08a5d",
  bad: "#e5646b",
  line: "#1e2533",
  text3: "#64708a",
};

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
const pct1 = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const fixed = (v, n = 3) => (v == null ? "—" : v.toFixed(n));

/** Largest observed value, ignoring gaps. */
function peak(values) {
  const finite = (values || []).filter((v) => v != null);
  return finite.length ? Math.max(...finite).toFixed(2) : "—";
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of [].concat(children)) node.appendChild(c);
  return node;
}

function h(tag, attrs = {}, html = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  if (html) node.innerHTML = html;
  return node;
}

/* ————————————————————————————————————————————————— line chart */

/**
 * Multi-series line chart on a shared x-axis.
 *
 * Each series is normalized to its own min/max: these are seconds, ratios and
 * token counts on one canvas, and the question the chart answers is "which
 * moved first", not "which is bigger". The peak value is printed in the legend
 * so the real magnitude is never lost.
 */
function lineChart(series, opts = {}) {
  const W = opts.width || 640;
  const H = opts.height || 220;
  const pad = { t: 14, r: 14, b: 22, l: 14 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label": opts.label || "metric time series",
  });

  // baseline / fault split
  if (opts.boundary != null) {
    const x = pad.l + iw * opts.boundary;
    svg.appendChild(
      el("rect", {
        x: pad.l, y: pad.t, width: x - pad.l, height: ih,
        fill: "rgba(255,255,255,.015)",
      })
    );
    svg.appendChild(
      el("line", {
        x1: x, y1: pad.t - 4, x2: x, y2: pad.t + ih,
        stroke: COLOR.text3, "stroke-width": 1, "stroke-dasharray": "3 3",
      })
    );
    svg.appendChild(
      el("text", {
        x: x + 6, y: pad.t + 8,
        fill: COLOR.text3, "font-size": 9, "font-family": "ui-monospace, monospace",
      })
    ).textContent = "fault window";
  }

  // horizontal guides
  for (let i = 1; i < 4; i++) {
    const y = pad.t + (ih * i) / 4;
    svg.appendChild(
      el("line", { x1: pad.l, y1: y, x2: pad.l + iw, y2: y, stroke: COLOR.line, "stroke-width": 1 })
    );
  }

  series.forEach((s) => {
    const vals = s.values;
    if (!vals || vals.length < 2) return;

    // null is a gap, not a zero: a histogram quantile over a window with no
    // requests has no value, and drawing it as 0 would invent a latency cliff
    // that never happened. Scale over the observed points and break the line.
    const finite = vals.filter((v) => v != null);
    if (finite.length < 2) return;
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const span = hi - lo || 1;

    const segments = [];
    let current = [];
    vals.forEach((v, i) => {
      if (v == null) {
        if (current.length) segments.push(current);
        current = [];
        return;
      }
      const x = pad.l + (iw * i) / (vals.length - 1);
      const y = pad.t + ih - ((v - lo) / span) * ih;
      current.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    });
    if (current.length) segments.push(current);

    segments.forEach((pts) => {
      if (pts.length < 2) return;
      svg.appendChild(
        el("polyline", {
          points: pts.join(" "),
          fill: "none", stroke: s.color, "stroke-width": s.width || 1.8,
          "stroke-linejoin": "round", "stroke-linecap": "round",
          opacity: s.dim ? 0.55 : 1,
        })
      );
    });
  });

  return svg;
}

function seriesColor(component, sliNode, index) {
  if (component === sliNode) return COLOR.sli;
  return [COLOR.signal, COLOR.exo, COLOR.warn, "#6aa9f0"][index % 4];
}

/* ————————————————————————————————————————————————— layered DAG */

/**
 * Longest-path layering. A node sits one column right of its deepest parent,
 * so every edge points forward and the picture reads left-to-right as
 * causality does.
 */
function layer(nodes) {
  const byName = new Map(nodes.map((n) => [n.name, n]));
  const depth = new Map();

  const resolve = (name, seen = new Set()) => {
    if (depth.has(name)) return depth.get(name);
    if (seen.has(name)) return 0; // defensive: a cycle would hang the layout
    seen.add(name);
    const parents = nodes.filter((n) => n.edges.includes(name));
    const d = parents.length
      ? Math.max(...parents.map((p) => resolve(p.name, seen) + 1))
      : 0;
    depth.set(name, d);
    return d;
  };
  nodes.forEach((n) => resolve(n.name));

  const columns = [];
  nodes.forEach((n) => {
    const d = depth.get(n.name);
    (columns[d] ||= []).push(n);
  });
  return { columns, depth, byName };
}

function dagSvg(nodes, opts = {}) {
  const { columns } = layer(nodes);
  const showMetrics = opts.showMetrics !== false;

  const NW = opts.nodeWidth || 152;
  const NH = showMetrics ? 42 : 28;
  const GAPX = opts.gapX || 62;
  const GAPY = 16;
  const pad = 18;

  const rows = Math.max(...columns.map((c) => c.length));
  const W = pad * 2 + columns.length * NW + (columns.length - 1) * GAPX;
  const H = pad * 2 + rows * NH + (rows - 1) * GAPY;

  const pos = new Map();
  columns.forEach((col, ci) => {
    const colH = col.length * NH + (col.length - 1) * GAPY;
    const y0 = pad + (H - pad * 2 - colH) / 2;
    col.forEach((n, ri) => {
      pos.set(n.name, {
        x: pad + ci * (NW + GAPX),
        y: y0 + ri * (NH + GAPY),
        node: n,
      });
    });
  });

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label": opts.label || "causal graph",
  });

  // edges first so nodes paint over them
  nodes.forEach((n) => {
    const from = pos.get(n.name);
    if (!from) return;
    n.edges.forEach((target) => {
      const to = pos.get(target);
      if (!to) return;
      const x1 = from.x + NW;
      const y1 = from.y + NH / 2;
      const x2 = to.x;
      const y2 = to.y + NH / 2;
      const mid = (x1 + x2) / 2;
      svg.appendChild(
        el("path", {
          d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
          class: n.exogenous ? "edge from-exo" : "edge",
        })
      );
    });
  });

  nodes.forEach((n) => {
    const p = pos.get(n.name);
    if (!p) return;
    const accent = n.exogenous ? COLOR.exo : n.sli ? COLOR.sli : COLOR.signal;
    const g = el("g", { class: "node-box" });
    g.appendChild(
      el("rect", {
        x: p.x, y: p.y, width: NW, height: NH, rx: 7,
        fill: "#0f131c", stroke: accent, "stroke-width": 1,
        "stroke-opacity": n.sli || n.exogenous ? 0.9 : 0.42,
      })
    );
    g.appendChild(
      el("rect", { x: p.x, y: p.y, width: 3, height: NH, rx: 1.5, fill: accent })
    );
    const label = el("text", {
      x: p.x + 11, y: p.y + (showMetrics ? 17 : 18), class: "node-label",
    });
    label.textContent = n.name;
    g.appendChild(label);

    if (showMetrics && n.metrics && n.metrics.length) {
      const m = el("text", { x: p.x + 11, y: p.y + 31, class: "node-metrics" });
      const joined = n.metrics.join(", ");
      m.textContent = joined.length > 24 ? `${n.metrics.length} metrics` : joined;
      g.appendChild(m);
    }

    const title = el("title");
    title.textContent = `${n.name}${
      n.metrics && n.metrics.length ? `\n${n.metrics.join("\n")}` : ""
    }`;
    g.appendChild(title);
    svg.appendChild(g);
  });

  return svg;
}

/* ————————————————————————————————————————————————— sections */

const METHOD_NOTE = {
  pipeline: "eight-layer change-point localization",
  threshold: "static runbook rules, in order",
  correlation: "rank by |r| against the SLI",
  topology: "always blame the graph root",
  llm: "Claude Opus 5, same evidence",
};

function renderMethods(data) {
  const host = document.getElementById("methods");
  if (!host) return;
  host.innerHTML = "";

  const best = {
    top1: Math.max(...data.methods.map((m) => m.top1)),
    top3: Math.max(...data.methods.map((m) => m.top3)),
    mrr: Math.max(...data.methods.map((m) => m.mrr)),
  };

  const head = h("div", { class: "method-row head" });
  ["method", "top-3 accuracy", "top-1", "top-3", "MRR", "FPR"].forEach((t) =>
    head.appendChild(h("div", { class: t === "method" ? "" : "num" }, t))
  );
  host.appendChild(head);

  data.methods.forEach((m) => {
    const row = h("div", {
      class: `method-row${m.name === "pipeline" ? " is-pipeline" : ""}`,
    });

    const name = h("div", { class: "method-name" });
    name.appendChild(h("b", {}, m.name));
    if (METHOD_NOTE[m.name]) {
      name.appendChild(h("span", { class: "method-note" }, METHOD_NOTE[m.name]));
    }
    row.appendChild(name);

    const bar = h("div", { class: "bar" });
    bar.appendChild(h("i", { style: `width:${(m.top3 * 100).toFixed(1)}%` }));
    row.appendChild(bar);

    const cell = (v, fmt, isBest, isBad) =>
      h(
        "div",
        { class: `num${isBest ? " best" : ""}${isBad ? " bad" : ""}` },
        fmt(v)
      );

    row.appendChild(cell(m.top1, pct, m.top1 === best.top1));
    row.appendChild(cell(m.top3, pct, m.top3 === best.top3));
    row.appendChild(cell(m.mrr, (v) => fixed(v, 3), m.mrr === best.mrr));
    row.appendChild(
      cell(m.falsePositiveRate, pct, m.falsePositiveRate === 0, m.falsePositiveRate > 0)
    );
    host.appendChild(row);
  });
}

function renderCases(data) {
  const host = document.getElementById("case-list");
  if (!host) return;
  const sli = data.mechanismGraph.nodes.find((n) => n.sli)?.name;
  host.innerHTML = "";

  data.caseStudies.forEach((c) => {
    const card = h("div", { class: "case" });

    const tag = c.ranked.length === 0
      ? '<strong class="tag null">named nothing</strong>'
      : c.correct
      ? '<strong class="tag ok">correct</strong>'
      : '<strong class="tag miss">scored as a miss</strong>';

    card.appendChild(
      h(
        "div",
        { class: "case-head" },
        `<div><h3>${c.title}</h3>
         <div class="meta">run ${c.runId} · scenario ${c.scenario} · verdict ${c.verdict}</div></div>
         ${tag}`
      )
    );

    const body = h("div", { class: "case-body" });

    // chart
    const chartCell = h("div", { class: "case-chart" });
    const flat = [];
    c.series.forEach((s) => {
      Object.entries(s.metrics).forEach(([metric, values]) => {
        flat.push({
          name: metric,
          component: s.component,
          values,
          color: seriesColor(s.component, sli, flat.length),
          width: s.component === sli ? 2.2 : 1.7,
        });
      });
    });
    chartCell.appendChild(
      lineChart(flat, { boundary: c.faultBoundary, label: `${c.scenario} telemetry` })
    );
    const legend = h("div", { class: "legend-inline" });
    flat.forEach((s) => {
      legend.appendChild(
        h(
          "span",
          {},
          `<i style="background:${s.color}"></i>${s.name} <span style="opacity:.6">(peak ${peak(
            s.values
          )})</span>`
        )
      );
    });
    chartCell.appendChild(legend);
    body.appendChild(chartCell);

    // side
    const side = h("div", { class: "case-side" });
    side.appendChild(h("p", { class: "case-note" }, c.note));

    if (c.ranked.length) {
      const list = h("ul", { class: "ranked" });
      c.ranked.forEach((r) => {
        const isHit = r.component === c.expected;
        const li = h("li", { class: isHit ? "hit" : "" });
        li.appendChild(h("span", { class: "r" }, `#${r.rank}`));
        li.appendChild(h("span", { class: "c" }, r.component));
        li.appendChild(
          h("span", { class: "onset" }, `+${r.onsetOffsetSeconds.toFixed(0)}s`)
        );
        list.appendChild(li);
      });
      side.appendChild(list);
      // Ranks come from the full candidate list, so a gap means Layer 8 or the
      // eligibility rule removed the entry that held that number. Say so —
      // otherwise "#1, #2, #4" reads as a rendering bug.
      const shown = c.ranked.map((r) => r.rank);
      if (Math.max(...shown) > shown.length) {
        side.appendChild(
          h(
            "p",
            { class: "fine", style: "margin:-.4rem 0 1rem" },
            "Missing ranks are candidates removed by the dependency filter, or " +
              "nodes never eligible as a root cause — the SLI itself and exogenous inputs."
          )
        );
      }
    } else {
      side.appendChild(
        h("div", { class: "ranked-empty" }, "no eligible candidate — silent")
      );
    }

    const expectedText = c.expected
      ? `expected <b>${c.expected}</b>`
      : "expected <b>nothing</b>";
    side.appendChild(
      h(
        "div",
        { class: "verdict-line" },
        `<span>${expectedText}</span><span>verdict <b>${c.verdict}</b> / ${c.expectedVerdict}</span>`
      )
    );
    body.appendChild(side);

    card.appendChild(body);
    host.appendChild(card);
  });
}

function renderConfusion(data) {
  const host = document.getElementById("confusion");
  if (!host) return;

  const rows = Object.keys(data.confusion);
  const cols = [
    ...new Set(rows.flatMap((r) => Object.keys(data.confusion[r]))),
  ].sort();

  const table = h("table", { class: "confusion" });
  const thead = h("thead");
  const hr = h("tr");
  hr.appendChild(h("th", {}, "injected ↓ / named →"));
  cols.forEach((c) => hr.appendChild(h("th", {}, c)));
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = h("tbody");
  rows.forEach((r) => {
    const tr = h("tr");
    tr.appendChild(h("th", {}, r));
    cols.forEach((c) => {
      const v = data.confusion[r][c] || 0;
      const cls = v === 0 ? "zero" : r === c ? "diag" : "off";
      tr.appendChild(h("td", { class: cls }, v === 0 ? "·" : String(v)));
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  host.innerHTML = "";
  host.appendChild(table);
}

function renderScenarios(data) {
  const host = document.getElementById("scenarios");
  if (!host) return;
  const table = h("table", { class: "scenarios" });
  table.appendChild(
    h(
      "thead",
      {},
      "<tr><th>scenario</th><th>runs</th><th>injected mechanism</th><th>what the pipeline named</th><th style='text-align:right'>top-1</th></tr>"
    )
  );
  const tbody = h("tbody");
  data.perScenario.forEach((s) => {
    const cls = s.top1 === 1 ? "perfect" : s.top1 === 0 ? "zero" : "";
    const preds = Object.entries(s.predictions)
      .map(([k, v]) => (v > 1 ? `${k} ×${v}` : k))
      .join(", ");
    const tr = h("tr", { class: cls });
    tr.appendChild(h("td", { class: "mono" }, s.scenario));
    tr.appendChild(h("td", {}, String(s.runs)));
    tr.appendChild(h("td", { class: "mono" }, s.expected || "—"));
    tr.appendChild(h("td", { class: "mono" }, preds));
    tr.appendChild(h("td", { class: "score" }, pct(s.top1)));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  host.innerHTML = "";
  host.appendChild(table);
}

function bind(data) {
  const hero = data.caseStudies[0];
  const values = {
    runsScored: String(data.generatedFrom.runsScored),
    cleanRuns: String(data.generatedFrom.cleanRuns),
    "headline.top1": pct1(data.headline.top1),
    "headline.top3": pct1(data.headline.top3),
    "headline.mrr": fixed(data.headline.mrr, 3),
    "headline.fpr": pct(data.headline.falsePositiveRate),
    "hero.runId": hero.runId,
    fprNote: `silent on all ${data.generatedFrom.cleanRuns} clean runs`,
    "footer.provenance":
      `${data.generatedFrom.runsScored} labelled runs · ` +
      `${data.generatedFrom.mechanisms} mechanisms · ` +
      `${data.generatedFrom.cleanRuns} clean · generated from committed traces`,
  };
  document.querySelectorAll("[data-bind]").forEach((node) => {
    const key = node.getAttribute("data-bind");
    if (values[key] != null) node.textContent = values[key];
  });
}

function renderQuiescence(data) {
  const node = document.getElementById("quiescence-copy");
  const q = data.baselineQuiescence;
  if (!node || !q) return;
  node.innerHTML =
    `On <strong>${q.contaminatedRuns} of ${q.quietRuns + q.contaminatedRuns}</strong> ` +
    `scored runs the SLI was already more than ${q.ratioThreshold}&times; its own ` +
    `median when the baseline window opened. Top-1 on those runs is ` +
    `<strong>${pct(q.contaminatedTop1)}</strong>. On the ${q.quietRuns} runs that ` +
    `did start quiet it is <strong>${pct(q.quietTop1)}</strong> — so this one ` +
    `defect in the capture protocol, not the ranking logic, accounts for much ` +
    `of the gap between the headline number and what the method does on clean input.`;
}

function renderHero(data) {
  const host = document.getElementById("hero-chart");
  if (!host) return;
  const c = data.caseStudies[0];
  const sli = data.mechanismGraph.nodes.find((n) => n.sli)?.name;
  const flat = [];
  c.series.forEach((s) => {
    Object.entries(s.metrics).forEach(([metric, values]) => {
      flat.push({
        name: metric,
        values,
        color: seriesColor(s.component, sli, flat.length),
        width: s.component === sli ? 2.4 : 1.9,
      });
    });
  });
  host.innerHTML = "";
  host.appendChild(
    lineChart(flat, { boundary: c.faultBoundary, width: 640, height: 236 })
  );
  const legend = h("div", { class: "legend-inline" });
  flat.forEach((s) =>
    legend.appendChild(
      h("span", {}, `<i style="background:${s.color}"></i>${s.name}`)
    )
  );
  host.appendChild(legend);
}

async function main() {
  let data;
  try {
    const res = await fetch("./data/portfolio-data.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    document.body.prepend(
      h(
        "div",
        {
          style:
            "padding:1rem;background:#2a1416;color:#e5646b;font-family:ui-monospace,monospace;font-size:.85rem",
        },
        `Could not load portfolio-data.json (${err.message}). Serve this directory over HTTP: <code>python3 -m http.server 4173 --directory portfolio</code>`
      )
    );
    return;
  }

  bind(data);
  renderQuiescence(data);
  renderHero(data);
  renderMethods(data);
  renderCases(data);
  renderConfusion(data);
  renderScenarios(data);

  const mech = document.getElementById("mechanism-graph");
  if (mech) mech.appendChild(dagSvg(data.mechanismGraph.nodes, { label: "vLLM mechanism graph" }));

  const boutique = document.getElementById("boutique-graph");
  if (boutique) {
    boutique.appendChild(
      dagSvg(
        data.boutiqueGraph.nodes.map((n) => ({ ...n, metrics: [] })),
        { showMetrics: false, nodeWidth: 168, gapX: 54, label: "Online Boutique dependency graph" }
      )
    );
  }
}

main();
