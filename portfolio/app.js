const DATA_URL = "./data/portfolio-data.json";

const $ = (selector) => document.querySelector(selector);

function fmtSeconds(value) {
  return `${Number(value).toFixed(value < 10 ? 1 : 0)}s`;
}

function fmtMs(value) {
  return `${Number(value).toFixed(1)}ms`;
}

function elapsedLabel(time, start) {
  const delta = Math.max(0, time - start);
  return `+${delta.toFixed(delta < 10 ? 1 : 0)}s`;
}

function pointsFor(values, width, height, pad = 12) {
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = Math.max(max - min, 0.001);
  return values.map((value, index) => {
    const x = pad + (index / (values.length - 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return [x, y];
  });
}

function pathFromPoints(points) {
  return points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
}

function renderHeroChart(data) {
  const svg = $("#hero-chart");
  const width = 620;
  const height = 260;
  const values = [26, 28, 29, 31, 30, 34, 48, 58, 63, 57, 52, 44, 39, 34, 31, 29];
  const pts = pointsFor(values, width, height, 28);
  const max = Math.max(...values, data.run.sloThresholdMs);
  const min = Math.min(...values, data.run.baselineP95Ms);
  const span = Math.max(max - min, 0.001);
  const thresholdY = height - 28 - ((data.run.sloThresholdMs - min) / span) * (height - 56);
  const line = pathFromPoints(pts);
  const area = `${line} L ${width - 28} ${height - 28} L 28 ${height - 28} Z`;

  svg.innerHTML = `
    <defs>
      <linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#78c8a2" stop-opacity="0.45"></stop>
        <stop offset="100%" stop-color="#78c8a2" stop-opacity="0"></stop>
      </linearGradient>
    </defs>
    <path d="${area}" fill="url(#areaFill)"></path>
    <line x1="28" y1="${thresholdY}" x2="${width - 28}" y2="${thresholdY}" stroke="#f2c66d" stroke-width="2" stroke-dasharray="8 8"></line>
    <text x="34" y="${thresholdY - 10}" fill="#f2c66d" font-size="13" font-weight="800">SLO threshold</text>
    <path d="${line}" fill="none" stroke="#9ee3bd" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"></path>
    <circle cx="${pts[6][0]}" cy="${pts[6][1]}" r="7" fill="#f08b70"></circle>
    <text x="${pts[6][0] + 12}" y="${pts[6][1] - 12}" fill="#f6f4ed" font-size="13" font-weight="800">violation</text>
  `;
}

function renderTimeline(data) {
  const start = data.run.timeline[0].time;
  $("#timeline").innerHTML = data.run.timeline.map((event) => `
    <li>
      <time>${elapsedLabel(event.time, start)}</time>
      <div>
        <strong>${event.label}</strong>
        <span>${new Date(event.time * 1000).toISOString().replace(".000", "")}</span>
      </div>
    </li>
  `).join("");
}

function renderRanking(data) {
  $("#ranking").innerHTML = data.rankedServices.map((item) => `
    <article class="rank-row">
      <span class="rank-badge">${item.rank}</span>
      <div>
        <h3>${item.service}</h3>
        <p>${item.abnormalMetrics.join(", ")} - ${item.note}</p>
      </div>
      <div class="confidence">${Math.round(item.confidence * 100)}%</div>
    </article>
  `).join("");
}

function renderSparkline(series) {
  const width = 280;
  const height = 128;
  const pts = pointsFor(series.values, width, height, 14);
  const line = pathFromPoints(pts);
  const area = `${line} L ${width - 14} ${height - 14} L 14 ${height - 14} Z`;
  const peak = pts[series.values.indexOf(Math.max(...series.values))];
  return `
    <svg class="spark" viewBox="0 0 ${width} ${height}" role="img" aria-label="${series.service} ${series.metric} sparkline">
      <path d="${area}" fill="#dcebe4"></path>
      <path d="${line}" fill="none" stroke="#2f7d58" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>
      <line x1="116" x2="116" y1="10" y2="${height - 10}" stroke="#c95f45" stroke-width="2" stroke-dasharray="5 6"></line>
      <circle cx="${peak[0]}" cy="${peak[1]}" r="5" fill="#315f9f"></circle>
    </svg>
  `;
}

function renderMetrics(data) {
  $("#metric-grid").innerHTML = data.metricSeries.map((series) => `
    <article class="metric-card">
      <h3>${series.service}</h3>
      <p class="metric-meta">${series.metric} normalized signal</p>
      ${renderSparkline(series)}
    </article>
  `).join("");
}

function nodeClass(name, data) {
  if (name === data.run.targetService) return "target";
  if (data.rankedServices.some((svc) => svc.service === name)) return "suspect";
  return "";
}

function renderGraph(data) {
  const svg = $("#dependency-graph");
  const positions = {
    frontend: [80, 250],
    productcatalogservice: [330, 70],
    recommendationservice: [330, 170],
    cartservice: [330, 275],
    checkoutservice: [330, 395],
    adservice: [330, 500],
    "redis-cart": [595, 275],
    currencyservice: [595, 90],
    shippingservice: [595, 190],
    paymentservice: [595, 395],
    emailservice: [595, 500]
  };
  const serviceByName = Object.fromEntries(data.services.map((service) => [service.name, service]));
  const edges = data.services.flatMap((service) => service.calls.map((target) => [service.name, target]));
  const edgeMarkup = edges.map(([source, target]) => {
    const [x1, y1] = positions[source];
    const [x2, y2] = positions[target];
    const mid = Math.max(24, (x2 - x1) / 2);
    return `<path class="edge" d="M ${x1 + 170} ${y1 + 24} C ${x1 + 170 + mid} ${y1 + 24}, ${x2 - mid} ${y2 + 24}, ${x2} ${y2 + 24}"></path>`;
  }).join("");
  const nodeMarkup = Object.entries(positions).map(([name, [x, y]]) => {
    const group = serviceByName[name]?.group || "service";
    return `
      <g class="node ${nodeClass(name, data)}" transform="translate(${x} ${y})">
        <rect width="170" height="48" rx="8"></rect>
        <text x="14" y="21">${name}</text>
        <text x="14" y="37" fill="#5c6a63" font-size="11" font-weight="700">${group}</text>
      </g>
    `;
  }).join("");
  svg.innerHTML = `
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L0,6 L9,3 z" fill="#9aa79f"></path>
      </marker>
    </defs>
    ${edgeMarkup}
    ${nodeMarkup}
  `;
}

function renderPipeline(data) {
  $("#pipeline-steps").innerHTML = data.pipeline.map((step, index) => `
    <article class="pipeline-step">
      <span>${index + 1}</span>
      <h3>${step.name}</h3>
      <p>${step.detail}</p>
    </article>
  `).join("");
}

function hydrateStats(data) {
  $("#stat-fault").textContent = `${data.run.fault} on ${data.run.targetService}`;
  $("#stat-latency").textContent = fmtSeconds(data.run.diagnosisLatencySeconds);
  $("#stat-runtime").textContent = fmtSeconds(data.run.rcaRuntimeSeconds);
  $("#slo-threshold").textContent = `${fmtMs(data.run.baselineP95Ms)} baseline / ${fmtMs(data.run.sloThresholdMs)} SLO`;
  $("#trigger-mode").textContent = data.run.triggerMode.replace("_", " ");
}

async function init() {
  const response = await fetch(DATA_URL);
  const data = await response.json();
  hydrateStats(data);
  renderHeroChart(data);
  renderTimeline(data);
  renderRanking(data);
  renderMetrics(data);
  renderGraph(data);
  renderPipeline(data);
}

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML("afterbegin", '<p class="load-error">Could not load dashboard data.</p>');
});
