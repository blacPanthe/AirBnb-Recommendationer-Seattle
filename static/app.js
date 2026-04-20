/* ── Airbnb Recommender — Topic Browser ─────────────────────────────────── */

// ── Helpers ──────────────────────────────────────────────────────────────────
function sentimentClass(label) {
  const l = (label || "").toLowerCase();
  if (l === "positive") return "sentiment-positive";
  if (l === "negative") return "sentiment-negative";
  return "sentiment-neutral";
}

function sentimentIcon(label) {
  const l = (label || "").toLowerCase();
  if (l === "positive") return "▲";
  if (l === "negative") return "▼";
  return "●";
}

function stars(rating) {
  if (rating == null) return '<span class="card-rating muted">No rating</span>';
  return `<span class="card-rating"><span class="star">★</span> ${parseFloat(rating).toFixed(1)}</span>`;
}

function formatPrice(p) {
  return p != null ? `$${Math.round(p)}/night` : "";
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function buildCard(listing) {
  const kws = (listing.keywords || []).slice(0, 4)
    .map(k => `<span class="card-kw">${escHtml(k)}</span>`).join("");

  const meta = [
    listing.room_type    ? `<span>🏠 ${escHtml(listing.room_type)}</span>` : "",
    listing.bedrooms     ? `<span>🛏 ${listing.bedrooms} bed</span>` : "",
    listing.price        ? `<span>💰 ${formatPrice(listing.price)}</span>` : "",
    listing.neighbourhood? `<span>📍 ${escHtml(listing.neighbourhood)}</span>` : "",
  ].filter(Boolean).join("");

  const sentClass = sentimentClass(listing.sentiment_label);
  const sentIcon  = sentimentIcon(listing.sentiment_label);

  return `
    <div class="listing-card">
      <div class="card-name">${escHtml(listing.name || "Untitled Listing")}</div>
      <div class="card-meta">${meta}</div>
      ${stars(listing.rating)}
      <span class="card-sentiment ${sentClass}">${sentIcon} ${listing.sentiment_label || "Neutral"}</span>
      ${kws ? `<div class="card-keywords">${kws}</div>` : ""}
    </div>`;
}

function showSpinner(el) {
  el.innerHTML = '<div class="spinner"></div>';
}

// ── Topic Browser ─────────────────────────────────────────────────────────────
const pillList   = document.getElementById("topic-pills");
const topicTitle = document.getElementById("topic-title");
const topicKws   = document.getElementById("topic-keywords");
const topicHdr   = document.getElementById("topic-header");
const topicEmpty = document.getElementById("topic-empty");
const topicRes   = document.getElementById("topic-results");

async function loadTopics() {
  try {
    const res  = await fetch("/api/topics");
    const data = await res.json();
    if (data.error) { pillList.innerHTML = `<p class="err">${data.error}</p>`; return; }

    pillList.innerHTML = "";
    data.forEach(topic => {
      const wrapper = document.createElement("div");
      wrapper.className = "pill-wrapper";

      const btn = document.createElement("button");
      btn.className = "topic-pill";
      btn.dataset.id = topic.id;
      btn.innerHTML =
        `<span class="pill-label">${escHtml(topic.label)}</span>` +
        `<span class="pill-count">${topic.listing_count}</span>`;
      btn.addEventListener("click", () => selectTopic(topic, btn));

      const kwHtml = (topic.keywords || [])
        .map(k => `<span class="tt-kw">${escHtml(k)}</span>`).join("");
      const tooltip = document.createElement("div");
      tooltip.className = "pill-tooltip";
      tooltip.innerHTML =
        `<div class="tt-title">Top keywords</div>` +
        `<div class="tt-kws">${kwHtml}</div>`;

      wrapper.appendChild(btn);
      wrapper.appendChild(tooltip);
      pillList.appendChild(wrapper);
    });

    // Auto-select the first topic
    const first = pillList.querySelector(".topic-pill");
    if (first) first.click();
  } catch (e) {
    pillList.innerHTML = `<p class="err">Could not load topics. Is the server running?</p>`;
  }
}

async function selectTopic(topic, btn) {
  pillList.querySelectorAll(".topic-pill").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");

  topicTitle.textContent = topic.label;
  topicKws.innerHTML = (topic.keywords || [])
    .map(k => `<span class="keyword-chip">${escHtml(k)}</span>`).join("");
  topicHdr.classList.remove("hidden");
  topicEmpty.classList.add("hidden");
  showSpinner(topicRes);

  try {
    const res  = await fetch(`/api/listings?topic=${topic.id}&top_n=30`);
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) {
      topicRes.innerHTML = '<p class="muted" style="padding:20px">No listings found for this topic.</p>';
      return;
    }
    topicRes.innerHTML = data.map(buildCard).join("");
  } catch {
    topicRes.innerHTML = `<p class="err">Error loading listings.</p>`;
  }
}

loadTopics();
