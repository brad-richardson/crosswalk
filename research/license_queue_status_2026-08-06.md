# License Queue Status — 2026-08-06

Per-dataset checklist for clearing the publish-queue exclusions in the latest
staging release (`2026-06-17.0`, `data/publish_staging/index.json`): **20
published, 14 excluded** — 13 excluded as `license status 'pending_review'
(excluded-pending-review)`, 1 on a quality hold (`co_bogota_bike_network`,
license already approved).

**What flips a dataset to published** (docs/PUBLISHING.md §"The registry"):
the dataset YAMLs carry no license fields; licensing lives entirely in
`datasets/licenses.toml`. Set `status = "approved"` **plus** `license` **plus**
`attribution` on the dataset's entry, then re-run `factory publish`. That is
the whole gate — a one-line human action per dataset once the source terms are
verified. (A quality hold excludes a dataset even with an approved license.)

Prior work this checklist builds on:

- `research/license_burndown_2026_07.md` — 2026-07-09 adversarial re-verification;
  drafted **copy-paste `licenses.toml` approval blocks** (§1.1–1.5) for the five
  sign-off-ready datasets below.
- Panel review 2026-07-06 — verdict + confidence recorded per entry in
  `datasets/licenses.toml` (`panel_recommendation`, `panel_confidence`, `note`).

Categories: **(a)** likely trivial — license known-permissive, just flip the
toml; **(b)** needs a human read/decision on source terms; **(c)** genuinely
problematic (no grant, or affirmatively restrictive). Offline snapshot — all
URLs below are what a human must open; nothing was re-fetched today.

## Checklist table (13 pending_review + 1 quality hold)

| # | Dataset | Likely license | Cat | Blocking item | Action needed | URL to check |
|---|---------|----------------|-----|---------------|---------------|--------------|
| 1 | us_philadelphia_sidewalks | DVRPC Data License (custom, attribution-only); dataset metadata says "No License Provided" but Use Limitations = "Unrestricted: can be shared internally and externally" | a | Registry status still `pending_review`; full license text already read verbatim in burndown (credit + no-updates + accuracy obligations only; no redistribution/derivative/SA/NC bar) | Paste burndown §1.1 toml block into `datasets/licenses.toml` (status=approved + license + attribution). Optional courtesy confirmation to mruane@dvrpc.org | https://catalog.dvrpc.org/dvrpc_data_license.html and https://catalog.dvrpc.org/dataset/dvrpc-pedestrian-network |
| 2 | sg_singapore_footpaths | Singapore Open Data Licence v1.0 (attribution-only, permits distribute/modify/adapt incl. commercial) | a | Panel's SODL-vs-website-ToU conflict resolved by DataMall landing-page acceptance statement; status not yet flipped | 1-minute eyeball of the landing-page sentence "Use of LTA's datasets and APIs on DataMall constitutes acceptance of the Singapore Open Data Licence", then paste burndown §1.2 toml block | https://datamall.lta.gov.sg/content/datamall/en.html and https://datamall.lta.gov.sg/content/datamall/en/SingaporeOpenDataLicence.html |
| 3 | sg_singapore_roads | Singapore Open Data Licence v1.0 | a | Same as #2 (identical instruments; published target IDs are synthetic geometry-hash ids = "derived analysis", expressly distributable under SODL) | Same check as #2 covers both; paste burndown §1.5 toml block | Same as #2 |
| 4 | de_berlin_roads | dl-de/by-2.0 per the actual Esri DE source item; official Berlin records say dl-de/zero-2.0 — attributing under the stricter by-2.0 satisfies both, conflict moot | a | by-vs-zero variant conflict (resolved by stricter-variant posture in burndown); status not yet flipped | Paste burndown §1.3 toml block (attribution "Datenquelle: Geoportal Berlin / Detailnetz…") | https://www.arcgis.com/sharing/rest/content/items/94bd52bffeef412daee37565fce745e2 (licenseInfo) and https://www.govdata.de/dl-de/by-2-0 |
| 5 | hk_hongkong_roads | DATA.GOV.HK Terms and Conditions of Use (open custom terms: attribution + IP acknowledgement; re-sale prohibited; **indemnity clause**) | a | Approval must consciously accept the indemnity clause (indemnify HK Government against third-party IP claims; HK SAR governing law). Derivative-silence concern resolved by FAQ ("Except re-sale… no restriction") | Accept indemnity term, then paste burndown §1.4 toml block; confirm current re-sale/FAQ wording while on the page | https://data.gov.hk/en/terms-and-conditions (+ DATA.GOV.HK FAQ) |
| 6 | us_austin_sidewalks | Public Domain (City of Austin Open Data Terms: "offered free and without restriction… not subject to copyright protection") | b | ArcGIS item-level boilerplate conflict: item licenseInfo says "Reproduction is not allowed without permission from Public Works"; city-wide PD policy contradicts it. Also austintexas.gov terms page 404s (verified via OSM wiki reproduction) | Decide PD-overrides-boilerplate (burndown lean: approve); re-anchor `license_url`/citation to the data.austintexas.gov Sidewalks dataset (pc5y-5bpw) so the boilerplate leaves the chain of title; then write approved+license+attribution (proposed_attribution already in licenses.toml) | https://data.austintexas.gov/d/pc5y-5bpw and https://data.austintexas.gov/stories/s/ranj-cccq (terms story; JS-only) |
| 7 | au_sydney_roads | CC-BY-4.0 (dataset-level on 4 mirrors + spatial.nsw.gov.au copyright policy) | b | Spatial Collaboration Portal's own ToS is Cloudflare-403 to bots — never human-read; need to confirm no SIX-Maps-style "on-selling/on-supplying" clause attaches to the CC-BY web-service exports | Open portal ToS in a browser; if clean, write approved + CC-BY-4.0 + the drafted attribution (proposed_attribution already in licenses.toml). Burndown lean: approve (CC BY §2(a)(5)(B) no-additional-restrictions controls) | https://portal.spatial.nsw.gov.au (terms/ToS page) and https://spatial.nsw.gov.au/copyright |
| 8 | br_sao_paulo_roads | CC-BY-SA-4.0 (GeoSampa portal-wide, verified verbatim) | b | Share-alike vs ODbL: BY-SA is NOT ODbL-compatible, so the bridge cannot go into the merged `all_bridges.parquet`; standalone per-dataset publication beside ODbL data is defensible mere-aggregation | Decision: publish as standalone CC-BY-SA-4.0 artifact excluded from `all_bridges.parquet` (small publisher change in `src/crosswalk/factory/publish.py`)? If yes → approvable; if unified table is mandatory → keep excluded. Optional ruling: geosampa@prefeitura.sp.gov.br | https://prefeitura.sp.gov.br/web/licenciamento/w/licen%C3%A7a-para-uso-de-dados-do-geosampa |
| 9 | us_fort_collins_streets | None (no named license; ArcGIS item licenseInfo is a pure liability disclaimer: "developed for use by the City of Fort Collins for its internal purposes only… AS IS, WITH ALL FAULTS") | c | No affirmative reuse/redistribution grant anywhere; Hub "open data" boilerplate is consumption-framed and contradicted by the internal-purposes disclaimer; CORA is access-only | Email gis@fcgov.com asking whether the Hub free-use statement is an affirmative redistribution grant (one email covers both Fort Collins datasets). Burndown lean: exclude absent written confirmation | https://www.arcgis.com/sharing/rest/content/items/f6380ef587ac4e54aaf988b15d8d1746?f=pjson and https://city-of-fort-collins-gis-fcgov.hub.arcgis.com/pages/open-data |
| 10 | us_fort_collins_sidewalks | None (same city posture as #9; FeatureServer copyrightText empty, no licenseInfo) | c | Same as #9 | Same email as #9 | https://city-of-fort-collins-gis-fcgov.hub.arcgis.com/pages/open-data |
| 11 | us_frisco_trails | None (Texas PIA release + warranty disclaimer; no named license, no redistribution/derivative grant; sibling service asserts "City of Frisco GIS" copyright) | c | PIA is an access statute, not a reuse license; no grant to hang an approval on | Burndown recommends flipping to exclude (alongside us_frisco_roads). Rescue would require written permission from City of Frisco GIS | https://maps.friscotexas.gov/gis/rest/services/Public/Search_Layers/MapServer/info/metadata |
| 12 | in_mumbai_streets | Unknown — zero license anywhere (BMC FeatureServer: empty copyrightText/description, no licenseInfo, no owner field; no BMC open-data portal; GODL-India/NDSAP don't automatically cover a municipal corporation) | c | Default all-rights-reserved; nothing to verify — there are no terms | Keep excluded. Rescue path: written permission from BMC's Centre for GIS & IT | https://services8.arcgis.com/r6MmJtuWAzMawmJ8/ArcGIS/rest/services/Streets/FeatureServer/0?f=pjson |
| 13 | jp_tokyo_emergency_roads | MLIT KSJ N10 tier = 非商用 (non-commercial only; redistribution of copies excluded; DB-format redistribution needs prior MLIT approval). Panel verdict: **exclude** (0.92) | c | License affirmatively prohibits our use; MLIT's ongoing CC-BY migration (e.g. N13 → CC BY 4.0 from Apr 2026) does not cover N10 | Flip registry status to a terminal exclude; periodically re-check whether N10 joins the CC-BY migration | https://nlftp.mlit.go.jp/ksj/other/agreement_02.html and the KSJ N10 datalist page |
| 14 | co_bogota_bike_network | CC-BY-4.0 — **already approved** | n/a (quality hold) | Not a license issue: cross-mode defect (cycleways matched to parallel road centerlines at 0.82–0.95 confidence), held since 2026-07-06 awaiting optimizer cross-mode gate / learned optimizer | No license work. Ships automatically once the quality hold is lifted | — |

## Fastest path to N more published datasets (easiest first)

Each of steps 1–5 is literally: paste the drafted toml block from
`research/license_burndown_2026_07.md` §1.x into `datasets/licenses.toml`,
re-run publish. Impact figures are target-segment counts from the burndown.

1. **us_philadelphia_sidewalks** (§1.1, 204,760 segs) — zero open questions; full license text already read. *+1*
2. **sg_singapore_footpaths** (§1.2, 109,960 segs) — 1-minute eyeball of the DataMall landing-page SODL sentence. *+2*
3. **sg_singapore_roads** (§1.5, 15,319 segs) — same check as #2, no extra work. *+3*
4. **de_berlin_roads** (§1.3, 43,369 segs) — stricter-variant (by-2.0) posture makes the by/zero conflict moot. *+4*
5. **hk_hongkong_roads** (§1.4, 36,107 segs) — only extra step is consciously accepting the indemnity clause. *+5*
6. **us_austin_sidewalks** (11,945 segs) — accept PD-overrides-boilerplate, re-anchor citation to pc5y-5bpw, write the entry. ~15 min. *+6*
7. **au_sydney_roads** (178,227 segs) — one browser visit to the portal ToS (bot-blocked), then write the entry. *+7*
8. **br_sao_paulo_roads** (212,264 segs) — decision + small publisher change (exclude from `all_bridges.parquet`, publish standalone CC-BY-SA). Largest pending dataset; worth the code change. *+8*
9. **us_fort_collins_streets + us_fort_collins_sidewalks** — one email to gis@fcgov.com covers both; async, uncertain outcome. *+10 if confirmed*
10. **in_mumbai_streets** — stuck without written BMC permission.
11. **us_frisco_trails**, **jp_tokyo_emergency_roads** — flip to terminal exclude (no lawful path as sourced); removes them from the pending queue rather than publishing.
12. **co_bogota_bike_network** — not license work; unblocks when the optimizer cross-mode gate lands.

Bottom line: **+5 datasets are copy-paste today**, +2 more with ~30 minutes of
human browsing (Austin, Sydney), +1 with a small publisher change (São Paulo),
+2 contingent on a Fort Collins email reply. Remaining 3 pending entries should
be resolved by excluding (Frisco trails, Tokyo N10) or shelved awaiting
permission (Mumbai).

*Also note:* the burndown's sixth sign-off-ready dataset,
`ch_grand_geneva_cycle_schema` (§1.6), is not in the 2026-06-17.0 staging
exclusion list (not in that factory run) but has a drafted approval block ready
for whenever it enters a release.
