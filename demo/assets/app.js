(() => {
  "use strict";

  const data = window.FORGE_RELEASE;
  if (!data) {
    document.querySelector("#headline-copy").textContent =
      "Release data is missing. Run make demo-build to restore the offline artifact.";
    return;
  }

  const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
  const num = (value, digits = 3) => Number(value).toFixed(digits);
  const money = (value, digits = 4) => `$${Number(value).toFixed(digits)}`;
  const headline = data.training.headline;

  document.querySelector("#headline-copy").textContent = headline.statement;
  const hero = [
    [pct(headline.task_success), "task success"],
    [`+${(headline.paired_delta_vs_r1.mean_task_success_delta * 100).toFixed(1)} pp`, "paired gain vs R1"],
    [headline.gpu_hours.toFixed(2), "RTX 4090 hours"],
    [money(headline.usd, 3), "measured training cost"],
  ];
  document.querySelector("#hero-metrics").innerHTML = hero.map(([value, label]) =>
    `<div class="hero-metric"><strong>${value}</strong><span>${label}</span></div>`
  ).join("");

  const ladder = data.training.ladder;
  const ladderChart = document.querySelector("#ladder-chart");
  const detail = document.querySelector("#ladder-detail");
  const showRung = (index) => {
    const rung = ladder[index];
    document.querySelectorAll(".ladder-row").forEach((row, rowIndex) =>
      row.classList.toggle("active", rowIndex === index));
    detail.innerHTML = `
      <span class="status-chip">${rung.status}</span>
      <h3>${rung.label}</h3>
      <dl>
        <dt>Task success</dt><dd>${pct(rung.task_success)}</dd>
        <dt>95% CI</dt><dd>${pct(rung.ci95[0])}–${pct(rung.ci95[1])}</dd>
        <dt>Schema valid</dt><dd>${pct(rung.schema_valid)}</dd>
        <dt>Tool accuracy</dt><dd>${pct(rung.tool_accuracy)}</dd>
        <dt>GPU hours</dt><dd>${rung.gpu_hours.toFixed(3)}</dd>
        <dt>Run cost</dt><dd>${money(rung.usd, 3)}</dd>
      </dl>`;
  };
  ladder.forEach((rung, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "ladder-row";
    row.innerHTML = `<span>${rung.label}</span><span class="bar-track"><i class="bar" style="width:${rung.task_success * 100}%"></i></span><span class="ladder-value">${pct(rung.task_success)}</span>`;
    row.addEventListener("click", () => showRung(index));
    ladderChart.append(row);
  });
  showRung(2);

  const serving = data.serving.serving_at_4_qps;
  const tabs = document.querySelector("#serving-tabs");
  const servingMetrics = document.querySelector("#serving-metrics");
  const showServing = (index) => {
    const point = serving[index];
    tabs.querySelectorAll("button").forEach((button, buttonIndex) =>
      button.classList.toggle("active", buttonIndex === index));
    const metrics = [
      [num(point.e2e_p50_s), "E2E p50 seconds"],
      [num(point.e2e_p95_s), "E2E p95 seconds"],
      [num(point.output_tokens_per_s, 1), "output tokens / second"],
      [money(point.cost_per_1k_successful_tasks_usd), "cost / 1k successful"],
      [pct(point.task_success), `verifier success (n=${point.requests})`],
      [point.vram_peak_mib.toFixed(0), "peak VRAM MiB"],
    ];
    servingMetrics.innerHTML = metrics.map(([value, label]) =>
      `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`
    ).join("");
  };
  serving.forEach((point, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = point.label;
    button.addEventListener("click", () => showServing(index));
    tabs.append(button);
  });
  showServing(2);

  const boundary = data.serving.speculative_boundary.points;
  const maxDelta = Math.max(...boundary.map((point) => Math.abs(point.p95_delta_s)));
  document.querySelector("#boundary-chart").innerHTML = boundary.map((point) => {
    const height = 45 + (Math.abs(point.p95_delta_s) / maxDelta) * 120;
    return `<div class="boundary-point ${point.verdict}"><div class="boundary-stick" style="height:${height}px">${point.p95_delta_s > 0 ? "+" : ""}${point.p95_delta_s.toFixed(3)}s</div><b>${point.qps} QPS</b><br>${point.verdict}</div>`;
  }).join("");

  document.querySelector("#constraint-grid").innerHTML = data.serving.structured_output.map((item) => `
    <article class="constraint-card">
      <p class="mini-title">${item.backend}</p>
      <h3>One pass → two pass</h3>
      <div class="before-after">
        <div class="score"><strong>${pct(item.simultaneous_task_success, 0)}</strong><span>task success</span></div>
        <span class="arrow">→</span>
        <div class="score good"><strong>${pct(item.two_pass_task_success, 0)}</strong><span>task success</span></div>
      </div>
      <p class="caption">p50 ${item.simultaneous_latency_p50_s.toFixed(3)}s → ${item.two_pass_latency_p50_s.toFixed(3)}s · tool-call rate stayed 100%.</p>
    </article>`).join("");

  const gateway = data.gateway;
  const summary = [
    [`${gateway.stable_pair_count}`, "stable paired cells"],
    [`${gateway.stable_median_e2e_p50_overhead_pct.toFixed(1)}%`, "median stable p50 overhead"],
    [`${pct(gateway.nonstable_gateway_error_rate_range[0], 0)}–${pct(gateway.nonstable_gateway_error_rate_range[1], 0)}`, "non-stable gateway errors"],
  ];
  document.querySelector("#gateway-summary").innerHTML = summary.map(([value, label]) =>
    `<div class="summary-card"><strong>${value}</strong><span>${label}</span></div>`
  ).join("");
  document.querySelector("#overload-table").innerHTML = gateway.overload.map((row) => `
    <tr>
      <td>${row.multiplier.toFixed(0)}× / ${row.offered_qps.toFixed(0)} QPS</td>
      <td>${pct(row.gateway_error_rate)}</td>
      <td>${pct(row.direct_error_rate)}</td>
      <td>${row.queue_depth_max.toFixed(0)}</td>
      <td><code>502 / upstream_error</code><br><small>reject_overload=${row.routing_decisions.reject_overload}</small></td>
    </tr>`).join("");
  document.querySelector("#gateway-limitation").innerHTML = `<b>Known limitation:</b> ${gateway.known_limitation}`;

  const receipts = [
    ["Schema", data.schema_version],
    ["Dataset hash", data.provenance.dataset_hash],
    ["Source manifest", data.provenance.source_manifest_sha256],
    ["Release model", headline.run_id],
    ["MTP export", data.exports.bf16_mtp_preserved.sha256],
    ["Gateway run", gateway.run_id],
  ];
  document.querySelector("#receipt-list").innerHTML = receipts.map(([term, value]) =>
    `<dt>${term}</dt><dd>${value}</dd>`).join("");
})();
