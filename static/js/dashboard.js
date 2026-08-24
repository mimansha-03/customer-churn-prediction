function bar(ctx, labels, values, color) {
  const CSS = getComputedStyle(document.documentElement);
  const c = (name) => CSS.getPropertyValue(name).trim();
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: color,
        borderRadius: 6,
        maxBarThickness: 46,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.y}% churn` } } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: (v) => v + "%" }, grid: { color: c("--line") } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderCharts(dash) {
  const CSS = getComputedStyle(document.documentElement);
  const c = (name) => CSS.getPropertyValue(name).trim();

  Chart.defaults.font.family = "Inter, sans-serif";
  Chart.defaults.color = c("--text-dim");
  Chart.defaults.borderColor = c("--line");

  bar(
    document.getElementById("chart-contract"),
    Object.keys(dash.churn_by_contract),
    Object.values(dash.churn_by_contract),
    c("--accent")
  );

  bar(
    document.getElementById("chart-internet"),
    Object.keys(dash.churn_by_internet),
    Object.values(dash.churn_by_internet),
    c("--accent-2")
  );

  const tenureOrder = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"];
  bar(
    document.getElementById("chart-tenure"),
    tenureOrder,
    tenureOrder.map((k) => dash.churn_by_tenure_bucket[k] ?? 0),
    c("--risk-med")
  );

  bar(
    document.getElementById("chart-payment"),
    Object.keys(dash.churn_by_payment),
    Object.values(dash.churn_by_payment),
    c("--risk-high")
  );

  new Chart(document.getElementById("chart-split"), {
    type: "doughnut",
    data: {
      labels: ["Retained", "Churned"],
      datasets: [{
        data: [dash.churn_distribution.No, dash.churn_distribution.Yes],
        backgroundColor: [c("--accent"), c("--risk-high")],
        borderColor: c("--surface"),
        borderWidth: 3,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, padding: 16 } } },
      cutout: "68%",
    },
  });
}

async function init() {
  const [dash, info] = await Promise.all([
    fetch("/api/dashboard-data").then((r) => r.json()),
    fetch("/api/model-info").then((r) => r.json()),
  ]);

  // Stats and model metrics render regardless of whether the chart
  // library loaded successfully.
  document.getElementById("stat-total").textContent = dash.total_customers.toLocaleString();
  document.getElementById("stat-churn-rate").textContent = dash.overall_churn_rate + "%";
  document.getElementById("stat-avg-charges").textContent = "$" + dash.avg_monthly_charges;
  document.getElementById("stat-avg-tenure").textContent = dash.avg_tenure + " mo";

  const metricsEl = document.getElementById("model-metrics");
  const best = info.all_results[info.best_model];
  const rows = [
    ["Best model", info.best_model],
    ["Accuracy", (best.accuracy * 100).toFixed(1) + "%"],
    ["Precision", (best.precision * 100).toFixed(1) + "%"],
    ["Recall", (best.recall * 100).toFixed(1) + "%"],
    ["F1 score", (best.f1 * 100).toFixed(1) + "%"],
    ["ROC-AUC", best.roc_auc.toFixed(3)],
    ["Test set size", info.n_test + " records"],
  ];
  metricsEl.innerHTML = rows
    .map(([label, value]) => `<div class="metric-row"><span>${label}</span><span>${value}</span></div>`)
    .join("");

  if (typeof Chart === "undefined") {
    console.error("Chart.js failed to load — charts skipped, stats still shown.");
    document.querySelectorAll("#chart-contract, #chart-internet, #chart-tenure, #chart-payment, #chart-split")
      .forEach((el) => {
        const msg = document.createElement("p");
        msg.textContent = "Chart library unavailable.";
        msg.style.color = "var(--text-faint)";
        msg.style.fontSize = "0.85rem";
        el.replaceWith(msg);
      });
    return;
  }

  renderCharts(dash);
}

init();
