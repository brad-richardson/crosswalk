/* Stats dashboard: renders index.json into tiles, charts, and tables. */
import {
  loadIndex, renderNav, showError, esc, el, withBase,
  fmtInt, fmtPct,
} from "./app.js";

renderNav("dashboard");
el("browseLink").href = withBase("browse.html");

/** Flatten index.json into a per-(release,dataset) list. */
function flatten(index) {
  const out = [];
  const releases = index.releases || {};
  for (const [release, rel] of Object.entries(releases)) {
    for (const [dataset, d] of Object.entries(rel.datasets || {})) {
      out.push({ release, dataset, ...d });
    }
  }
  return out;
}

function bar(label, fillFrac, valText) {
  const w = Math.max(0, Math.min(1, fillFrac || 0)) * 100;
  return (
    `<div class="bar-row"><div class="label" title="${esc(label)}">${esc(label)}</div>` +
    `<div class="track"><div class="fill" style="width:${w.toFixed(1)}%"></div></div>` +
    `<div class="val">${esc(valText)}</div></div>`
  );
}

function chartCard(title, note, bodyHtml) {
  return `<div class="chart"><div class="title">${esc(title)}</div>` +
    `<div class="note">${esc(note)}</div>${bodyHtml || '<p class="empty">No data.</p>'}</div>`;
}

(async function main() {
  let index;
  try {
    index = await loadIndex();
  } catch (e) {
    showError(el("meta"), e);
    return;
  }

  const rows = flatten(index);
  const published = rows.filter((r) => r.status === "published");
  const excluded = rows.filter((r) => r.status !== "published");

  // ---- meta + tiles ----
  el("meta").innerHTML =
    `<p><span class="pill">latest release: ${esc(index.latest_release || "—")}</span>` +
    `<span class="pill">releases: ${fmtInt((index.totals || {}).n_releases)}</span>` +
    `<span class="pill">generated: ${esc(index.generated_at || "—")}</span></p>`;

  const totMatched = published.reduce((a, r) => a + ((r.stats || {}).n_matched || 0), 0);
  const totTargets = published.reduce((a, r) => a + ((r.stats || {}).n_target || 0), 0);
  const tiles = [
    ["published datasets", fmtInt(published.length)],
    ["excluded (pending)", fmtInt(excluded.length)],
    ["matched rows", fmtInt(totMatched)],
    ["overall match rate", totTargets ? fmtPct(totMatched / totTargets) : "—"],
  ];
  el("tiles").innerHTML = tiles
    .map(([k, v]) => `<div class="tile"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`)
    .join("");

  // ---- charts ----
  const byRate = [...published].sort((a, b) => ((b.stats || {}).match_rate || 0) - ((a.stats || {}).match_rate || 0));
  const rateBars = byRate
    .map((r) => bar(r.dataset, (r.stats || {}).match_rate || 0, fmtPct((r.stats || {}).match_rate)))
    .join("");

  const bySize = [...published].sort((a, b) => ((b.stats || {}).n_target || 0) - ((a.stats || {}).n_target || 0));
  const maxTarget = Math.max(1, ...bySize.map((r) => (r.stats || {}).n_target || 0));
  const sizeBars = bySize
    .map((r) => bar(r.dataset, ((r.stats || {}).n_target || 0) / maxTarget, fmtInt((r.stats || {}).n_target)))
    .join("");

  el("charts").innerHTML =
    chartCard("Match rate by dataset", "matched / target features, match-decision only", rateBars) +
    chartCard("Dataset size", "target (local) feature count", sizeBars);

  // ---- published table ----
  const pubHead =
    "<thead><tr><th>dataset</th><th>type</th><th>release</th>" +
    "<th class='num'>targets</th><th class='num'>matched</th><th class='num'>match rate</th>" +
    "<th class='num'>review</th><th class='num'>groups</th><th>license</th><th>files</th></tr></thead>";
  const pubBody = published
    .sort((a, b) => a.dataset.localeCompare(b.dataset))
    .map((r) => {
      const s = r.stats || {}, disp = r.display || {}, lic = r.license || {};
      const link = withBase(`browse.html#dataset=${encodeURIComponent(r.dataset)}&release=${encodeURIComponent(r.release)}`);
      return (
        `<tr><td><a href="${link}"><code>${esc(r.dataset)}</code></a>` +
        `<div class="sub">${esc(disp.display_name || r.dataset)}</div></td>` +
        `<td>${esc(disp.type || "—")}</td><td>${esc(r.release)}</td>` +
        `<td class="num">${fmtInt(s.n_target)}</td><td class="num">${fmtInt(s.n_matched)}</td>` +
        `<td class="num">${fmtPct(s.match_rate)}</td><td class="num">${fmtInt(s.n_review)}</td>` +
        `<td class="num">${fmtInt(s.n_groups)}</td>` +
        `<td>${esc(lic.license || "—")}<div class="sub">${esc(lic.attribution || "")}</div></td>` +
        `<td class="dl">${fileLinks(r)}</td></tr>`
      );
    })
    .join("");
  el("published").innerHTML = pubHead + `<tbody>${pubBody || emptyRow(10)}</tbody>`;

  // ---- excluded table ----
  el("excluded").innerHTML =
    "<thead><tr><th>dataset</th><th>type</th><th>reason</th><th>likely license (hint)</th></tr></thead>" +
    "<tbody>" +
    (excluded
      .sort((a, b) => a.dataset.localeCompare(b.dataset))
      .map((r) => {
        const disp = r.display || {}, lic = r.license || {};
        return (
          `<tr><td><code>${esc(r.dataset)}</code><div class="sub">${esc(disp.display_name || r.dataset)}</div></td>` +
          `<td>${esc(disp.type || "—")}</td><td>${esc(r.reason || "excluded")}</td>` +
          `<td class="sub">${esc(lic.note || "—")}</td></tr>`
        );
      })
      .join("") || emptyRow(4)) +
    "</tbody>";

  // ---- licensing ----
  const ov = index.overture || {};
  el("licensing-body").innerHTML =
    `<p>Every published table is a derived work of <strong>both</strong> the local source dataset ` +
    `and Overture Maps. Redistribution must carry both attributions.</p>` +
    `<p><strong>Overture:</strong> ${esc(ov.attribution || "")} ` +
    (ov.url ? `(<a href="${esc(ov.url)}">${esc(ov.url)}</a>)` : "") + `</p>` +
    `<p><strong>Per-dataset source license</strong> is shown in the published table above ` +
    `and in each dataset's <code>index.json</code> entry. Datasets with unverified licenses ` +
    `are excluded rather than published under a guess.</p>`;

  el("footer").innerHTML =
    `Machine-readable index: <a href="${esc(index.site_url || "")}/index.json"><code>index.json</code></a> · ` +
    `per-release <code>checksums.txt</code> (sha256) accompanies every release. ` +
    `Independent project · <a href="https://github.com/brad-richardson/matcher">source on GitHub</a>.`;
})();

function fileLinks(r) {
  const files = r.files || {};
  const parts = [];
  if (files["bridge.parquet"]) {
    parts.push(`<a href="${esc(bridgeHref(r))}" download>bridge.parquet</a>`);
  }
  if (files["manifest.json"]) {
    parts.push(`<a href="${esc(manifestHref(r))}">manifest</a>`);
  }
  return parts.join(" · ") || "—";
}
function bridgeHref(r) {
  return `${base()}/bridges/release=${r.release}/dataset=${r.dataset}/bridge.parquet`;
}
function manifestHref(r) {
  return `${base()}/bridges/release=${r.release}/dataset=${r.dataset}/manifest.json`;
}
function base() {
  const override = new URLSearchParams(location.search).get("base");
  return (override || (window.MATCHER_CONFIG || {}).DEFAULT_BASE_URL || "").replace(/\/+$/, "");
}
function emptyRow(cols) {
  return `<tr><td colspan="${cols}" class="empty">None.</td></tr>`;
}
