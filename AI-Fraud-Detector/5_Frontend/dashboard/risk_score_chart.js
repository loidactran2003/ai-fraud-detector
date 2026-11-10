/* KPI tiles + risk-score histogram + top-reasons chart.  Reads GET /api/stats. */
const REASON_LABELS = {
  amount: "Unusual amount", checkout: "Rushed checkout", time: "Odd hour", logins: "Failed logins",
  is_vpn: "VPN", is_proxy: "Proxy", country_mismatch: "Country mismatch", high_risk_country: "High-risk country",
  is_emulator: "Emulator", is_rooted: "Rooted device", timezone_mismatch: "Timezone mismatch",
  accounts_per_device: "Shared device", users_per_card: "Shared card", account_age_days: "New account",
  pages_viewed: "Straight to checkout",
};

const RiskCharts = (() => {
  let hist, reasons;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt$ = (n) => "$" + Math.round(n || 0).toLocaleString();
  const th = () => window.THRESHOLDS || { review: 30, block: 70 };   // set from /api/health
  // a 5-point bucket is coloured by the band its lowest score falls in
  const bandColor = (b) => css(b >= th().block ? "--critical" : b >= th().review ? "--warning" : "--good");
  const bandName = (b) => (b >= th().block ? "Block" : b >= th().review ? "Review" : "Approve");
  const haveChart = () => typeof Chart !== "undefined";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Chart.js unavailable (CDN blocked / offline): show the same numbers as a small table
  function fallback(canvasId, rows, head) {
    const canvas = document.getElementById(canvasId);
    const box = canvas.parentElement;
    box.querySelector(".fallback-table")?.remove();
    canvas.hidden = true;
    box.insertAdjacentHTML("beforeend", `<table class="fallback-table"><thead><tr><th>${head[0]}</th><th style="text-align:right">${head[1]}</th></tr></thead>` +
      `<tbody>${rows.map(([a, b]) => `<tr><td>${esc(a)}</td><td class="num">${esc(b)}</td></tr>`).join("")}</tbody></table>`);
  }

  function baseOptions(horizontal) {
    const grid = css("--border"), ink = css("--text-2");
    return {
      responsive: true, maintainAspectRatio: false, indexAxis: horizontal ? "y" : "x",
      animation: false, plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: horizontal, color: grid }, ticks: { color: ink }, border: { color: grid } },
        y: { grid: { display: !horizontal, color: grid }, ticks: { color: ink }, border: { color: grid } },
      },
    };
  }

  function drawHistogram(rows) {
    const buckets = Array.from({ length: 21 }, (_, i) => i * 5);
    const counts = Object.fromEntries(rows.map((r) => [r.bucket, r.count]));
    const data = buckets.map((b) => counts[b] || 0);
    if (!haveChart()) return fallback("hist-chart", buckets.filter((b, i) => data[i]).map((b) => [`${b}–${b + 4} (${bandName(b)})`, counts[b]]), ["Score", "Orders"]);
    const opts = baseOptions(false);
    opts.scales.y.type = "logarithmic";
    opts.scales.y.title = { display: true, text: "orders (log scale)", color: css("--muted") };
    opts.scales.x.title = { display: true, text: "risk score", color: css("--muted") };
    opts.plugins.tooltip = { callbacks: {
      title: (items) => `Score ${items[0].label}–${+items[0].label + 4} · ${bandName(+items[0].label)}`,
      label: (item) => ` ${item.raw.toLocaleString()} orders`,
    } };
    hist?.destroy();
    hist = new Chart(document.getElementById("hist-chart"), {
      type: "bar",
      data: { labels: buckets, datasets: [{ data, backgroundColor: buckets.map(bandColor),
              borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: "bottom", categoryPercentage: 0.92, barPercentage: 0.95 }] },
      options: opts,
    });
  }

  function drawReasons(rows) {
    const top = rows.slice(0, 8);
    if (!haveChart()) return fallback("reasons-chart", top.map((r) => [REASON_LABELS[r.feature] || r.feature, r.count]), ["Top reason", "Orders"]);
    const opts = baseOptions(true);
    opts.plugins.tooltip = { callbacks: { label: (item) => ` ${item.raw} flagged orders` } };
    reasons?.destroy();
    reasons = new Chart(document.getElementById("reasons-chart"), {
      type: "bar",
      data: { labels: top.map((r) => REASON_LABELS[r.feature] || r.feature),
              datasets: [{ data: top.map((r) => r.count), backgroundColor: css("--accent"),
                           borderRadius: { topRight: 4, bottomRight: 4 }, borderSkipped: "left", barPercentage: 0.7 }] },
      options: opts,
    });
  }

  async function load() {
    const set = (id, v) => (document.getElementById(id).textContent = v);
    let s;
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      s = await res.json();
    } catch (e) {
      set("k-orders", "–");
      set("k-avg", `stats unavailable (${e.message})`);
      return;
    }
    set("k-orders", (s.orders || 0).toLocaleString());
    set("k-avg", `average score ${s.avg_score ?? "–"}`);
    set("k-review", (s.review || 0).toLocaleString());
    set("k-blocked", (s.blocked || 0).toLocaleString());
    set("k-risk", fmt$(s.amount_at_risk));
    set("k-await", (s.awaiting_review || 0).toLocaleString());
    drawHistogram(s.histogram || []);
    drawReasons(s.top_reasons || []);
  }

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => load());
  return { load };
})();
