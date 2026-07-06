/*
 * Shared helpers for the live data browser.
 *
 * - Resolves the data base URL (config.js DEFAULT_BASE_URL, or ?base= override).
 * - Loads the machine-readable index.json.
 * - Lazily boots DuckDB-WASM (from jsDelivr) and runs read-only SQL against the
 *   HTTPS-hosted Parquet via range reads.
 * - Small formatting / DOM utilities used by both pages.
 *
 * ES module: imported by dashboard.js and browse.js. Relies on the global
 * `window.CROSSWALK_CONFIG` set by config.js (loaded first as a classic script).
 */

const CFG = window.CROSSWALK_CONFIG || {};

// --------------------------------------------------------------------------
// Base URL + URL builders
// --------------------------------------------------------------------------
export function getBaseUrl() {
  const override = new URLSearchParams(location.search).get("base");
  const raw = (override || CFG.DEFAULT_BASE_URL || "").trim();
  return raw.replace(/\/+$/, "");
}

export function indexUrl() {
  return `${getBaseUrl()}/index.json`;
}

export function bridgeUrl(release, dataset) {
  return `${getBaseUrl()}/bridges/release=${release}/dataset=${dataset}/bridge.parquet`;
}

export function allBridgesUrl(release) {
  return `${getBaseUrl()}/bridges/release=${release}/all_bridges.parquet`;
}

/** Preserve the ?base= override when linking between pages. Inserts the query
 * BEFORE any #fragment and after any existing query string. */
export function withBase(path) {
  const override = new URLSearchParams(location.search).get("base");
  if (!override) return path;
  const hashIdx = path.indexOf("#");
  const bare = hashIdx >= 0 ? path.slice(0, hashIdx) : path;
  const frag = hashIdx >= 0 ? path.slice(hashIdx) : "";
  const sep = bare.includes("?") ? "&" : "?";
  return `${bare}${sep}base=${encodeURIComponent(override)}${frag}`;
}

/** Only allow http(s) URLs sourced from config to become live hrefs. */
export function safeUrl(u) {
  const s = String(u ?? "");
  return /^https?:\/\//i.test(s) ? s : "#";
}

// --------------------------------------------------------------------------
// index.json
// --------------------------------------------------------------------------
let _indexPromise = null;
export function loadIndex() {
  if (!_indexPromise) {
    _indexPromise = fetch(indexUrl(), { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${indexUrl()}`);
      return r.json();
    });
  }
  return _indexPromise;
}

// --------------------------------------------------------------------------
// DuckDB-WASM (lazy singleton)
// --------------------------------------------------------------------------
let _dbPromise = null;

async function bootDuckDB() {
  const ver = CFG.DUCKDB_WASM_VERSION || "1.32.0";
  const duckdb = await import(
    /* @vite-ignore */ `https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@${ver}/+esm`
  );
  const bundles = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(bundles);
  const workerUrl = URL.createObjectURL(
    new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" })
  );
  const worker = new Worker(workerUrl);
  const db = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  URL.revokeObjectURL(workerUrl);
  // Cast 64-bit ints to JS doubles so counts render without BigInt friction.
  await db.open({ query: { castBigIntToDouble: true } });
  return db;
}

export function getDB() {
  if (!_dbPromise) _dbPromise = bootDuckDB();
  return _dbPromise;
}

/** Statements we allow from the free-form SQL box (read-only intent). */
const READONLY_RE = /^\(*\s*(with|select|from|pragma|describe|explain|show|values|table)\b/i;
const WRITE_RE = /\b(attach|copy|install|load|create|insert|update|delete|drop|alter|export|call)\b/i;

/** Strip leading whitespace and SQL comments so a comment-led SELECT still passes. */
function stripLeading(sql) {
  let s = sql, prev;
  do {
    prev = s;
    s = s.replace(/^\s+/, "").replace(/^--[^\n]*\n?/, "").replace(/^\/\*[\s\S]*?\*\//, "");
  } while (s !== prev);
  return s;
}

export function isReadOnlySql(sql) {
  return READONLY_RE.test(stripLeading(sql)) && !WRITE_RE.test(sql);
}

/**
 * Run a SQL query and return { columns, rows } with JSON-safe values.
 * Arrow BigInt / Date values are normalized to numbers / ISO strings.
 */
export async function runQuery(sql) {
  const db = await getDB();
  const conn = await db.connect();
  try {
    const res = await conn.query(sql);
    const columns = res.schema.fields.map((f) => f.name);
    const rows = res.toArray().map((r) => {
      const o = r.toJSON();
      const clean = {};
      for (const c of columns) clean[c] = normalizeCell(o[c]);
      return clean;
    });
    return { columns, rows };
  } finally {
    await conn.close();
  }
}

function normalizeCell(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") return Number(v);
  if (v instanceof Date) return v.toISOString();
  // Arrow may hand back typed values with a toString (e.g. timestamps).
  if (typeof v === "object" && typeof v.toString === "function" && !(Array.isArray(v))) {
    const s = v.toString();
    return s === "[object Object]" ? JSON.stringify(v) : s;
  }
  return v;
}

// --------------------------------------------------------------------------
// Formatting
// --------------------------------------------------------------------------
export function fmtInt(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString("en-US");
}
export function fmtPct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(100 * Number(x)).toFixed(digits)}%`;
}
export function fmtNum(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}
export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// --------------------------------------------------------------------------
// DOM utilities
// --------------------------------------------------------------------------
export function el(id) { return document.getElementById(id); }

export function renderNav(active) {
  const base = getBaseUrl();
  const banner = document.createElement("div");
  banner.className = "banner";
  banner.innerHTML =
    "<strong>Unofficial / independent project.</strong> Community-built bridge tables linking " +
    "local road & path datasets to Overture Maps GERS ids. <em>Not</em> an Overture Maps Foundation product " +
    "or endorsement. See licensing &amp; attribution below.";
  const nav = document.createElement("nav");
  nav.className = "top";
  nav.innerHTML =
    `<span class="brand">GERS Bridge Tables</span>` +
    `<a href="${withBase("index.html")}" class="${active === "dashboard" ? "active" : ""}">Stats</a>` +
    `<a href="${withBase("browse.html")}" class="${active === "browse" ? "active" : ""}">Query &amp; browse</a>` +
    `<span class="spacer"></span>` +
    `<span class="base" title="data source (edit site/config.js or use ?base=)">source: ${esc(base || "(unset)")}</span>`;
  document.body.prepend(nav);
  document.body.prepend(banner);
}

/** Attach copy-to-clipboard buttons to every <pre data-copy> block. */
export function wireCopyButtons(root = document) {
  root.querySelectorAll("pre[data-copy]").forEach((pre) => {
    if (pre.querySelector(".copybtn")) return;
    const btn = document.createElement("button");
    btn.className = "copybtn";
    btn.textContent = "copy";
    btn.addEventListener("click", () => {
      const text = pre.getAttribute("data-copy-text") || pre.textContent.replace(/copy$/, "");
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "copied";
        setTimeout(() => (btn.textContent = "copy"), 1200);
      });
    });
    pre.appendChild(btn);
  });
}

export function showError(container, err) {
  const base = getBaseUrl();
  container.innerHTML =
    `<div class="msg err"><strong>Could not load data.</strong> ${esc(err.message || err)}` +
    `<br><br>Data source is <code>${esc(base || "(unset)")}</code>. ` +
    `Set <code>DEFAULT_BASE_URL</code> in <code>site/config.js</code> to your R2 public domain, ` +
    `or append <code>?base=&lt;url&gt;</code> to this page's URL to point at a staging host ` +
    `(e.g. <code>?base=http://localhost:8000</code>).</div>`;
}
