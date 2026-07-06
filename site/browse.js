/* Query browser: reverse GERS lookup, per-dataset browse, free SQL — all via
 * DuckDB-WASM reading the published Parquet over HTTPS. */
import {
  loadIndex, getDB, runQuery, isReadOnlySql, renderNav, withBase,
  bridgeUrl, allBridgesUrl, getBaseUrl,
  esc, el, fmtInt, wireCopyButtons, showError,
} from "./app.js";

const CFG = window.CROSSWALK_CONFIG || {};
const PAGE_SIZE = CFG.PAGE_SIZE || 50;

renderNav("browse");
el("statsLink").href = withBase("index.html");
el("footer").innerHTML =
  `Queries run in your browser (DuckDB-WASM) against read-only Parquet — nothing is sent to a server. ` +
  `An independent community project · <a href="https://github.com/brad-richardson/crosswalk">source on GitHub</a> · ` +
  `<a href="https://github.com/brad-richardson/crosswalk/issues">feedback welcome</a>.`;

// --------------------------------------------------------------------------
// Shared rendering
// --------------------------------------------------------------------------
function sqlStr(s) {
  return "'" + String(s).replace(/'/g, "''") + "'";
}

function renderTable(container, result, emptyMsg = "No rows.") {
  if (!result || !result.rows.length) {
    container.innerHTML = `<p class="empty">${esc(emptyMsg)}</p>`;
    return;
  }
  const { columns, rows } = result;
  const head = "<thead><tr>" + columns.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr></thead>";
  const body = rows
    .map((r) => "<tr>" + columns.map((c) => cell(c, r[c])).join("") + "</tr>")
    .join("");
  container.innerHTML = `<div class="tablewrap"><table>${head}<tbody>${body}</tbody></table></div>`;
}

function cell(col, v) {
  if (v === null || v === undefined) return `<td class="sub">∅</td>`;
  if (col === "match_decision") {
    const cls = v === "match" ? "match" : v === "review" ? "review" : "";
    return `<td><span class="chip ${cls}">${esc(v)}</span></td>`;
  }
  if (col === "confidence") return `<td class="num">${Number(v).toFixed(3)}</td>`;
  if (col === "gers_id" || col === "local_id") return `<td class="mono">${esc(v)}</td>`;
  if (typeof v === "number") return `<td class="num">${col.endsWith("frac") ? Number(v).toFixed(4) : v}</td>`;
  return `<td>${esc(v)}</td>`;
}

function setSql(preId, sql) {
  const pre = el(preId);
  pre.textContent = sql;
  pre.setAttribute("data-copy-text", sql);
  wireCopyButtons();
}

function busy(node, on, msg) {
  node.innerHTML = on ? `<span class="spinner"></span> ${esc(msg || "running…")}` : esc(msg || "");
}

// --------------------------------------------------------------------------
// Boot: populate selectors, wire everything
// --------------------------------------------------------------------------
(async function main() {
  let index;
  try {
    index = await loadIndex();
  } catch (e) {
    showError(el("fatal"), e);
    return;
  }

  // published (release -> [datasets]) map
  const relToDatasets = {};
  for (const [release, rel] of Object.entries(index.releases || {})) {
    const pub = Object.entries(rel.datasets || {})
      .filter(([, d]) => d.status === "published")
      .map(([name]) => name)
      .sort();
    if (pub.length) relToDatasets[release] = pub;
  }
  const releases = Object.keys(relToDatasets).sort().reverse();
  const latest = index.latest_release && relToDatasets[index.latest_release]
    ? index.latest_release
    : releases[0];

  if (!releases.length) {
    el("fatal").innerHTML =
      `<div class="msg info">No published datasets in this data source yet.</div>`;
    return;
  }

  fillSelect(el("revRelease"), releases, latest);
  fillSelect(el("brRelease"), releases, latest);
  syncDatasetSelect(latest);
  el("brRelease").addEventListener("change", () => syncDatasetSelect(el("brRelease").value));

  function syncDatasetSelect(release) {
    fillSelect(el("brDataset"), relToDatasets[release] || [], null);
  }

  // Warm DuckDB in the background; report readiness.
  el("ddStatus").innerHTML = `<span class="spinner"></span> booting DuckDB-WASM…`;
  getDB().then(
    () => (el("ddStatus").textContent = "DuckDB-WASM ready."),
    (e) => (el("ddStatus").innerHTML = `<span class="msg err">DuckDB-WASM failed to load: ${esc(e.message)}</span>`)
  );

  // ---- reverse lookup ----
  el("revBtn").addEventListener("click", doReverse);
  el("gersId").addEventListener("keydown", (e) => { if (e.key === "Enter") doReverse(); });

  async function doReverse() {
    const release = el("revRelease").value;
    const gid = el("gersId").value.trim();
    if (!gid) { busy(el("revStatus"), false, "Enter a GERS id."); return; }
    const url = allBridgesUrl(release);
    const sql =
      `SELECT dataset, local_id, confidence, match_type, match_decision,\n` +
      `       gers_start_frac, gers_end_frac, local_start_frac, local_end_frac\n` +
      `FROM read_parquet(${sqlStr(url)})\n` +
      `WHERE gers_id = ${sqlStr(gid)}\n` +
      `ORDER BY dataset, confidence DESC;`;
    setSql("revSql", sql);
    busy(el("revStatus"), true, "querying all_bridges.parquet…");
    try {
      const res = await runQuery(sql);
      busy(el("revStatus"), false, `${res.rows.length} row(s) reference this GERS id.`);
      renderTable(el("revResult"), res, "No dataset in this release references that GERS id.");
    } catch (e) {
      busy(el("revStatus"), false, "");
      el("revResult").innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
    }
  }

  // ---- per-dataset browse (paginated) ----
  const brState = { offset: 0 };
  el("brBtn").addEventListener("click", () => { brState.offset = 0; doBrowse(); });
  el("brPrev").addEventListener("click", () => { brState.offset = Math.max(0, brState.offset - PAGE_SIZE); doBrowse(); });
  el("brNext").addEventListener("click", () => { brState.offset += PAGE_SIZE; doBrowse(); });

  function browseSql(withLimit) {
    const release = el("brRelease").value;
    const dataset = el("brDataset").value;
    const url = bridgeUrl(release, dataset);
    const conds = [];
    const dec = el("brDecision").value;
    if (dec) conds.push(`match_decision = ${sqlStr(dec)}`);
    const lo = parseFloat(el("brMinConf").value);
    const hi = parseFloat(el("brMaxConf").value);
    if (!Number.isNaN(lo) && lo > 0) conds.push(`confidence >= ${lo}`);
    if (!Number.isNaN(hi) && hi < 1) conds.push(`confidence <= ${hi}`);
    const where = conds.length ? `\nWHERE ${conds.join("\n  AND ")}` : "";
    let sql = `SELECT *\nFROM read_parquet(${sqlStr(url)})${where}\nORDER BY confidence DESC`;
    if (withLimit) sql += `\nLIMIT ${PAGE_SIZE} OFFSET ${brState.offset}`;
    return { sql: sql + ";", url, dataset, release };
  }

  async function doBrowse() {
    const { sql, url, dataset, release } = browseSql(true);
    setSql("brSql", sql);
    el("brLinks").innerHTML =
      `<a href="${esc(url)}" download>download bridge.parquet</a> · ` +
      `<a href="${esc(`${getBaseUrl()}/bridges/release=${release}/dataset=${dataset}/manifest.json`)}">manifest.json</a>`;
    busy(el("brStatus"), true, "querying bridge.parquet…");
    el("brPager").hidden = true;
    try {
      const res = await runQuery(sql);
      renderTable(el("brResult"), res, "No rows match those filters.");
      const end = brState.offset + res.rows.length;
      el("brPageInfo").textContent = res.rows.length
        ? `rows ${brState.offset + 1}–${end}`
        : brState.offset > 0
          ? "no more rows"
          : "no rows";
      el("brPrev").disabled = brState.offset === 0;
      el("brNext").disabled = res.rows.length < PAGE_SIZE;
      el("brPager").hidden = false;
      busy(el("brStatus"), false, "");
    } catch (e) {
      busy(el("brStatus"), false, "");
      el("brResult").innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
    }
  }

  // ---- free SQL ----
  const exDataset = () => browseSqlExample();
  function browseSqlExample() {
    const release = latest;
    const dataset = (relToDatasets[latest] || ["<dataset>"])[0];
    return `SELECT match_decision, count(*) AS n, round(avg(confidence), 3) AS avg_conf\n` +
      `FROM read_parquet(${sqlStr(bridgeUrl(release, dataset))})\n` +
      `GROUP BY match_decision\nORDER BY n DESC;`;
  }
  function reverseSqlExample() {
    return `SELECT dataset, count(*) AS n_matches\n` +
      `FROM read_parquet(${sqlStr(allBridgesUrl(latest))})\n` +
      `GROUP BY dataset\nORDER BY n_matches DESC;`;
  }
  el("sqlBox").value = exDataset();
  el("sqlExDataset").addEventListener("click", () => (el("sqlBox").value = exDataset()));
  el("sqlExReverse").addEventListener("click", () => (el("sqlBox").value = reverseSqlExample()));
  el("sqlBtn").addEventListener("click", runFreeSql);

  async function runFreeSql() {
    const sql = el("sqlBox").value.trim();
    if (!sql) return;
    if (!isReadOnlySql(sql)) {
      el("sqlStatus").innerHTML =
        `<span class="msg err">Read-only queries only (SELECT / WITH / DESCRIBE / PRAGMA / SHOW). ` +
        `Write / attach / install statements are blocked.</span>`;
      return;
    }
    busy(el("sqlStatus"), true, "running…");
    const t0 = performance.now();
    try {
      const res = await runQuery(sql);
      const ms = Math.round(performance.now() - t0);
      busy(el("sqlStatus"), false, `${res.rows.length} row(s) in ${ms} ms (showing up to 1000).`);
      renderTable(el("sqlResult"), { columns: res.columns, rows: res.rows.slice(0, 1000) });
    } catch (e) {
      busy(el("sqlStatus"), false, "");
      el("sqlResult").innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
    }
  }

  // ---- geometry join example ----
  const exDs = (relToDatasets[latest] || ["<dataset>"])[0];
  const joinSql =
    `-- Join bridge ids to Overture geometry (WKT) for mapping.\n` +
    `-- Overture segments are published on public S3; DuckDB reads them directly.\n` +
    `INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;\n` +
    `WITH bridge AS (\n` +
    `  SELECT * FROM read_parquet(${sqlStr(bridgeUrl(latest, exDs))})\n` +
    `  WHERE match_decision = 'match'\n` +
    `)\n` +
    `SELECT b.local_id, b.gers_id, b.confidence, ST_AsText(s.geometry) AS overture_wkt\n` +
    `FROM bridge b\n` +
    `JOIN read_parquet(\n` +
    `  's3://overturemaps-us-west-2/release/${latest}/theme=transportation/type=segment/*',\n` +
    `  filename=false, hive_partitioning=true\n` +
    `) s ON s.id = b.gers_id\n` +
    `LIMIT 100;`;
  setSql("joinSql", joinSql);

  wireCopyButtons();

  // ---- deep link: browse.html#dataset=…&release=… ----
  applyHash();
  function applyHash() {
    const h = new URLSearchParams(location.hash.replace(/^#/, ""));
    const ds = h.get("dataset"), rel = h.get("release");
    if (rel && relToDatasets[rel]) { el("brRelease").value = rel; syncDatasetSelect(rel); }
    if (ds) {
      const r = el("brRelease").value;
      if ((relToDatasets[r] || []).includes(ds)) { el("brDataset").value = ds; brState.offset = 0; doBrowse(); }
    }
  }
})();

function fillSelect(sel, values, selected) {
  sel.innerHTML = values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
  if (selected != null) sel.value = selected;
}
