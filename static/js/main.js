const form = document.getElementById("predict-form");
const btn = document.getElementById("predict-btn");
const resultEmpty = document.getElementById("result-empty");
const resultBody = document.getElementById("result-body");

const GAUGE_CIRCUMFERENCE = 251.2; // path length of the semicircle arc

const RISK_COLORS = {
  Low: "var(--risk-low)",
  Medium: "var(--risk-med)",
  High: "var(--risk-high)",
};

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    const el = document.getElementById("footer-model");
    if (el) {
      const acc = (data.all_results?.[data.best_model]?.accuracy * 100).toFixed(1);
      el.textContent = `${data.best_model} (${acc}% accuracy)`;
    }
  } catch (e) {
    /* non-critical */
  }
}
loadModelInfo();

function collectFormData() {
  const fd = new FormData(form);
  const payload = {};
  for (const [key, value] of fd.entries()) {
    payload[key] = value;
  }
  return payload;
}

function setLoading(isLoading) {
  btn.classList.toggle("loading", isLoading);
  btn.disabled = isLoading;
}

function renderResult(data) {
  resultEmpty.hidden = true;
  resultBody.hidden = false;

  const prob = data.churn_probability; // 0-100
  const riskLevel = data.risk_level;
  const color = RISK_COLORS[riskLevel] || RISK_COLORS.Low;

  // Gauge arc fill (semicircle, 0-100% maps to full dasharray offset -> 0)
  const arc = document.getElementById("gauge-arc");
  const offset = GAUGE_CIRCUMFERENCE * (1 - prob / 100);
  requestAnimationFrame(() => {
    arc.style.stroke = color;
    arc.style.strokeDashoffset = offset;
  });

  // Needle rotation: -90deg (0%) to +90deg (100%) across the semicircle
  const needle = document.getElementById("gauge-needle");
  const angle = -90 + (prob / 100) * 180;
  requestAnimationFrame(() => {
    needle.style.transform = `rotate(${angle}deg)`;
  });

  document.getElementById("gauge-number").textContent = `${prob}%`;
  document.getElementById("gauge-number").style.color = color;

  const badge = document.getElementById("risk-badge");
  badge.textContent = `${riskLevel} risk`;
  badge.className = `risk-badge ${riskLevel.toLowerCase()}`;

  document.getElementById("pct-churn").textContent = `${data.churn_probability}%`;
  document.getElementById("pct-retain").textContent = `${data.retain_probability}%`;
  requestAnimationFrame(() => {
    document.getElementById("bar-churn").style.width = `${data.churn_probability}%`;
    document.getElementById("bar-retain").style.width = `${data.retain_probability}%`;
  });

  const list = document.getElementById("factors-list");
  list.innerHTML = "";
  data.key_factors.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    list.appendChild(li);
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setLoading(true);
  try {
    const payload = collectFormData();
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.error || "Something went wrong.");
      return;
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert("Could not reach the prediction service. Is the Flask app running?");
  } finally {
    setLoading(false);
  }
});

document.getElementById("sample-btn").addEventListener("click", () => {
  const samples = [
    {
      tenure: 2, Contract: "Month-to-month", MonthlyCharges: 95.5, TotalCharges: 191,
      PaperlessBilling: "Yes", PaymentMethod: "Electronic check", gender: "Female",
      SeniorCitizen: "No", Partner: "No", Dependents: "No", PhoneService: "Yes",
      MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No",
      OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
      StreamingTV: "Yes", StreamingMovies: "Yes",
    },
    {
      tenure: 60, Contract: "Two year", MonthlyCharges: 24.9, TotalCharges: 1495,
      PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)", gender: "Male",
      SeniorCitizen: "No", Partner: "Yes", Dependents: "Yes", PhoneService: "Yes",
      MultipleLines: "No", InternetService: "No", OnlineSecurity: "No internet service",
      OnlineBackup: "No internet service", DeviceProtection: "No internet service",
      TechSupport: "No internet service", StreamingTV: "No internet service",
      StreamingMovies: "No internet service",
    },
  ];
  const sample = samples[Math.floor(Math.random() * samples.length)];
  Object.entries(sample).forEach(([key, value]) => {
    const el = form.elements[key];
    if (el) el.value = value;
  });
});
