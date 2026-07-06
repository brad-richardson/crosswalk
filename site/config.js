/*
 * Live data browser — configuration.
 *
 * THIS IS THE ONE PLACE TO SET THE DATA SOURCE.
 *
 * DEFAULT_BASE_URL must point at the ROOT of the published bridge-table tree —
 * i.e. the directory that contains `index.json` and the `bridges/` folder. When
 * the Cloudflare R2 bucket is live, set this to its public HTTPS domain
 * (custom domain or the bucket's `*.r2.dev` URL), with NO trailing slash.
 *
 * You can always override this at runtime for testing against any staging host
 * without editing the file:
 *
 *     https://<pages-site>/?base=http://localhost:8000
 *     https://<pages-site>/browse.html?base=https://my-staging-host
 *
 * The `?base=` query parameter (if present) wins over DEFAULT_BASE_URL.
 */
window.CROSSWALK_CONFIG = {
  // R2 public development URL for the crosswalk-bridges bucket. r2.dev is
  // rate-limited / dev-tier; swap to a custom domain (e.g. under bradr.dev)
  // for production traffic.
  DEFAULT_BASE_URL: "https://pub-1960acc8b68148ac82da2fd033be804f.r2.dev",

  // Pinned DuckDB-WASM release (loaded from jsDelivr as an ES module).
  DUCKDB_WASM_VERSION: "1.32.0",

  // Rows per page in the per-dataset browser.
  PAGE_SIZE: 50,
};
