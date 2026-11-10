/* Flagged-orders table with reviewer decisions + the "Check an order" form. */
const FlaggedTable = (() => {
  const body = document.getElementById("flagged-body");
  const tableError = document.getElementById("table-error");
  const moreBtn = document.getElementById("load-more");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const money = (n) => "$" + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const LEVEL_TEXT = { high: "▲ Block", medium: "● Review", low: "✓ Approve" };
  const DECISION_TEXT = { approved: "Approved", blocked: "Blocked", confirmed_fraud: "Confirmed fraud", false_positive: "False positive" };
  const PAGE = 200;
  let offset = 0;

  // Optional API key (server started with FRAUD_API_KEY): localStorage.setItem("fraudApiKey", "...")
  function headers() {
    const h = { "Content-Type": "application/json" };
    try { const k = localStorage.getItem("fraudApiKey"); if (k) h["X-API-Key"] = k; } catch { /* storage blocked */ }
    return h;
  }
  async function errorText(res) {
    try { const j = await res.json(); return j.error || `HTTP ${res.status}`; } catch { return `HTTP ${res.status}`; }
  }

  const badge = (level) => `<span class="badge ${level}">${LEVEL_TEXT[level]}</span>`;
  const reasonList = (reasons) => reasons.length
    ? `<ul class="reasons">${reasons.map((r) => `<li>${esc(r.text)} <span class="pct">${r.impact}%</span></li>`).join("")}</ul>`
    : `<span class="decision">No single dominant signal</span>`;

  function row(t) {
    const decided = t.review_decision
      ? `<span class="decision">${DECISION_TEXT[t.review_decision]}</span>`
      : `<div class="actions" data-id="${esc(t.transaction_id)}">
           <button data-d="approved">Approve</button><button data-d="blocked">Block</button>
           <button data-d="confirmed_fraud">Fraud</button><button data-d="false_positive">False alarm</button></div>`;
    return `<tr>
      <td><strong>${esc(t.transaction_id)}</strong><br><span class="decision">${esc(t.user_id)} · ${esc(String(t.created_at).slice(0, 16))}</span></td>
      <td>${badge(t.risk_level)}<br><span class="score">${t.risk_score}</span>/100</td>
      <td class="num">${money(t.amount)}</td>
      <td>${esc(t.ip_country)} → ${esc(t.billing_country)}${t.is_vpn ? " · VPN" : ""}${t.is_proxy ? " · proxy" : ""}</td>
      <td>${reasonList(t.reasons)}</td>
      <td>${decided}</td></tr>`;
  }

  async function load(append = false) {
    const level = document.getElementById("f-level").value;
    const unreviewed = document.getElementById("f-unreviewed").checked ? "&unreviewed=1" : "";
    const minScore = (window.THRESHOLDS || { review: 30 }).review;
    if (!append) offset = 0;
    moreBtn.disabled = true;
    try {
      const res = await fetch(`/api/transactions?min_score=${minScore}&limit=${PAGE}&offset=${offset}` +
                              `${level ? "&level=" + level : ""}${unreviewed}`);
      if (!res.ok) throw new Error(await errorText(res));
      const { transactions, total } = await res.json();
      const html = transactions.map(row).join("");
      if (append) body.insertAdjacentHTML("beforeend", html);
      else body.innerHTML = html || `<tr><td colspan="6" class="empty">Nothing flagged. Nice.</td></tr>`;
      offset += transactions.length;
      const shown = offset;
      document.getElementById("table-count").textContent =
        total != null ? `${shown.toLocaleString()} of ${total.toLocaleString()} shown` : `${shown} shown`;
      moreBtn.hidden = !(total != null && shown < total);
    } catch (e) {
      if (append) { tableError.textContent = `Could not load more: ${e.message}`; tableError.hidden = false; }
      else body.innerHTML = `<tr><td colspan="6" class="error">Could not load transactions: ${esc(e.message)}</td></tr>`;
    } finally {
      moreBtn.disabled = false;
    }
  }

  body.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-d]");
    if (!btn) return;
    const box = btn.parentElement;
    const id = box.dataset.id;
    const buttons = box.querySelectorAll("button");
    buttons.forEach((b) => (b.disabled = true));
    box.querySelector(".error")?.remove();
    try {
      const res = await fetch(`/api/transactions/${encodeURIComponent(id)}/review`, {
        method: "POST", headers: headers(),
        body: JSON.stringify({ decision: btn.dataset.d, reviewer: "dashboard" }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      box.outerHTML = `<span class="decision">${DECISION_TEXT[btn.dataset.d]}</span>`;
      RiskCharts.load().catch(() => {});
    } catch (err) {
      buttons.forEach((b) => (b.disabled = false));
      box.insertAdjacentHTML("beforeend", `<div class="error" role="alert">Not saved: ${esc(err.message)}</div>`);
    }
  });
  document.getElementById("f-level").addEventListener("change", () => load());
  document.getElementById("f-unreviewed").addEventListener("change", () => load());
  moreBtn.addEventListener("click", () => load(true));

  // ---- "Check an order" form (dry run: nothing is stored)
  const form = document.getElementById("tester");
  const out = document.getElementById("tester-result");
  const submit = form.querySelector("button[type=submit]");
  const created = form.elements.account_created;
  created.value = new Date(Date.now() - 2 * 3600e3 - new Date().getTimezoneOffset() * 60e3).toISOString().slice(0, 16);
  const FLAGS = ["is_vpn", "is_proxy", "is_emulator", "is_rooted", "timezone_mismatch"];

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(form);
    const order = Object.fromEntries([...f.entries()].filter(([k]) => !FLAGS.includes(k)));
    for (const k of FLAGS) order[k] = f.has(k) ? 1 : 0;
    for (const k of ["amount", "checkout_seconds", "login_attempts", "failed_logins", "pages_viewed"]) {
      if (order[k] === "") delete order[k]; else order[k] = Number(order[k]);
    }
    order.account_created = (order.account_created || "").replace("T", " ");
    if (!order.account_created) delete order.account_created;
    submit.disabled = true;
    const label = submit.textContent;
    submit.textContent = "Scoring…";
    out.classList.add("show");
    try {
      const res = await fetch("/api/score?dry_run=1", { method: "POST", headers: headers(), body: JSON.stringify(order) });
      if (!res.ok) throw new Error(await errorText(res));
      const r = await res.json();
      out.innerHTML = `${badge(r.risk_level)} <span class="score">${r.risk_score}</span>/100 → <strong>${esc(r.action)}</strong>` +
                      ` <span class="decision">(not stored)</span>${reasonList(r.reasons)}`;
    } catch (err) {
      out.innerHTML = `<span class="error">Could not score this order: ${esc(err.message)}</span>`;
    } finally {
      submit.disabled = false;
      submit.textContent = label;
    }
  });

  return { load: () => load(false) };
})();
