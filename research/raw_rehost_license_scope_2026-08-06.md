# Raw-rehost license scope — re-scoring `approved` for full-geometry redistribution

**Date:** 2026-08-06 · **Scope:** read-only re-score of every `status = "approved"`
entry in `datasets/licenses.toml` (31 after PR #468), asking a
*different* question than the 2026-07 burndown asked: **does the recorded license
evidence permit republishing the complete dataset (geometry + attributes) in
transformed format?** No new web research — this works purely from
`research/license_burndown_2026_07.md`, `research/license_queue_status_2026-08-06.md`,
`research/seattle_license_clearance.md`, and the registry `note`/`license` fields.

## TL;DR

1. **No approved dataset is `raw-problematic`.** The single `approved` flag has not
   yet shipped anything it shouldn't have. The exposure is real but the current
   inventory is clean.
2. **The second axis is already designed and was never built.**
   `docs/plans/2026-07-06-unmatched-export-design.md` §3 specifies exactly this field
   (`geometry_status`), explicitly names `targets/` (#370) as in scope, and says
   `targets/` "already redistributes full geometry + `source_tags` for every
   `status = "approved"` dataset — a larger exposure than the unmatched complement."
   **Recommendation: implement that design, do not invent `target_mirror`.**
3. **One live factual defect:** `us_utah_slc_roads`'s published attribution string
   asserts *"segment identifiers only, geometry and attributes removed"* — and it is
   currently shipping in the targets mirror **attached to a 17 MB full-geometry
   parquet**. This is in `targets/index.json` and in the snapshot `meta.yaml` today.
4. **#468: flip all five, but hold `sg_singapore_*` out of the raw mirror**
   pending a read of the SODL Downstream Sub-Licensee clause. Everything else in
   #468 is raw-clear-with-attribution-fixes.

**Implementation status (2026-08-06):** the default-deny geometry gate described
below is now implemented on `agent/geometry-license-axis`. Of the 31 bridge-approved
entries, 24 are geometry-approved and 7 remain pending: Hong Kong, Seoul, both
Singapore datasets, and the three Montana datasets. Hong Kong is conservatively
pending Brad's explicit acceptance of the indemnity at full-mirror scale.

---

## 1. Inventory

**Approved today (31).** In the targets mirror (`data/publish_staging_targets`,
generated 2026-07-09, 21 datasets) unless marked. The 5 approved-but-absent
(`co_bogota_sidewalks`, `ke_mombasa_roads`, `kr_seoul_roads`, `us_fresno_roads`,
`us_montana_bozeman`) are absent only because no `data/raw/<name>_v1.0.parquet`
exists locally — **they will enter the mirror automatically on the next fetch+publish.**
No license decision is holding them back.

**PR #468 (`0695034`, merged 2026-08-06)** flipped five to `approved`:
`us_philadelphia_sidewalks`, `sg_singapore_footpaths`, `sg_singapore_roads`,
`de_berlin_roads`, `hk_hongkong_roads`. All five have local raw parquets, so all
five enter the raw mirror on the next publish (+364K target segments of
full geometry).

**Staleness note:** the on-disk mirror contains `co_bogota_bike_network`, which has
carried a `quality_hold` since 2026-07-06. The hold was only enforced on the targets
path by #380 (`db6a36c`, 2026-07-09), *after* this staging tree was built. A
re-publish drops it. Not a license issue, but the mirror on R2 is not what current
code would produce.

---

## 2. Score table

`raw-clear` = express grant to redistribute/adapt the data itself.
`raw-arguable` = grant exists but our recorded rationale leaned on ID-only
minimalism, or a term bites differently at full-data scale.
`raw-problematic` = terms restrict redistribution/derivatives, or impose conditions
we don't currently meet. **None scored `raw-problematic`.**

### Currently approved

| Dataset | Mirror | Raw score | Load-bearing language (as recorded) | Attribution adequate for raw? | Action |
|---|---|---|---|---|---|
| `ca_toronto_roads` | ✓ | raw-clear | "copy, modify, publish, translate, adapt, **distribute** or otherwise use the Information **in any medium, mode or format** for any lawful purpose" | Yes | none |
| `nl_amsterdam_roads` | ✓ | raw-clear | CC0-1.0, "License: CC-0 (1.0); Publisher/Rightsholder: Rijkswaterstaat" | Yes (courtesy) | none |
| `gb_london_roads` | ✓ | raw-clear | OGL-UK-3.0 (copy/publish/distribute/transmit/adapt) | Yes, but `[year]` token was flagged "must be confirmed at publish time"; entry has no `source_url` (absent from index) | refresh year; add `source_url` |
| `fi_helsinki_roads` | ✓ | raw-clear | "right to **distribute**, remix, tweak, and build upon your work, **even commercially**" (CC-BY-4.0) | Yes — includes the hyperlink Väylä requires | optional: add download date + service name per Väylä's example |
| `au_melbourne_roads` | ✓ | raw-clear | CC BY 3.0 AU, dataset-level | Yes | none |
| `co_bogota_roads` / `co_bogota_sidewalks` / `co_bogota_bike_network` | ✓/–/✓(held) | raw-clear | "Todos los datos geográficos del Mapa de Referencia cuentan con una **Licencia de Uso Abierta**" → CC BY 4.0 deed.es | Yes | none |
| `ke_kisumu_roads` / `ke_mombasa_roads` / `ke_nairobi_roads` | ✓/–/✓ | raw-clear | "Creative Commons Attribution 4.0" dataset-level on energydata.info | Yes | verifier noted energydata.info's general terms flag *some* third-party sets carry provider-consent conditions; this one is labelled plain CC BY 4.0 — no action |
| `ch_geneva_hiking_routes` / `ch_geneva_pedestrian_network` | ✓ | raw-clear | "en **libre accès (open data)** pour un usage privé ou une **utilisation commerciale**… La **mention de la source** des données est requise lors de leur réutilisation" | Yes — both carry ICDG/SITG source mention; hiking entry carries the required extraction date | pedestrian entry has no extraction date; SITG's own template includes "extrait en date du" — add it |
| `us_boston_streets` / `us_boston_sidewalks` / `us_boston_bike_network` | ✓ | raw-clear | PDDL-1.0 (public domain dedication) dataset-level on data.boston.gov | Yes (courtesy) | none |
| `us_seattle_sidewalks` | ✓ | raw-clear | Registry note is the **only** entry that says it out loud: *"Clearance covers **geometry redistribution** too (carry the accuracy disclaimer forward)"* (`research/seattle_license_clearance.md`) | **No** — the note instructs carrying the accuracy disclaimer forward; the published attribution string omits it | append the SDOT accuracy/no-warranty disclaimer |
| `us_utah_slc_roads` | ✓ | raw-clear (license) / **defective (string)** | UGRC default CC BY 4.0, confirmed dataset-level via ArcGIS item `478fbef6…` `licenseInfo` | **No — actively false.** String reads "…transformed into a GERS **bridge table; segment identifiers only, geometry and attributes removed**" while shipping a 17 MB full-geometry parquet | **fix before next publish** (see §3) |
| `us_fresno_roads` | – | raw-clear | "Creative Commons Attribution… satisfies the Open Definition… Rights: **No restrictions on public use**" (version unspecified — immaterial, every CC-BY version permits redistribution) | Yes | none |
| `tn_tunis_ml_roads` | ✓ | raw-clear **with share-alike conditions** | "Data in this repository has been licensed by Microsoft under the **Open Database License (ODbL)**" | Partially — string names ODbL, which is the core notice, but ODbL §4.4/4.6 require the *redistributed database* to be offered under ODbL with the licence text/URI available | add explicit "this snapshot is licensed ODbL-1.0" + licence URI to the mirror's attribution |
| `kr_seoul_roads` | – | raw-clear (thin evidence) | Badge only: "공공저작물 : 출처표시 (제 1유형)" — KOGL Type 1 permits commercial use and derivative works | Yes | verifier already recommended recording `kogl.or.kr/info/licenseType1.do` as the operative text; **the licence body itself was never fetched** → `needs-web-recheck` (low risk) |
| `us_montana_bozeman` / `helena` / `missoula` | –/✓/✓ | raw-clear (bare assertion) | Registry note is a single unsourced sentence: *"Montana State Library MSDI framework data is published in the public domain."* No `license_url`, no panel review, no burndown dossier | Yes ("Montana State Library (MSDI Transportation Framework)") | `needs-web-recheck` — cite the MSDI/MSL public-domain policy URL. Low risk, zero evidence on file |
| `us_usfs_flathead` / `us_usfs_lolo` | ✓ | raw-clear | "US federal government work — public domain under **17 U.S.C. § 105**" | Yes | none |

### PR #468 additions (merged before this scope review completed)

| Dataset | Raw score | Load-bearing language (as recorded) | Attribution adequate for raw? | Action |
|---|---|---|---|---|
| `us_philadelphia_sidewalks` | raw-clear | Burndown §4: *"clearly permitted (w/ DVRPC credit) — 'Unrestricted: can be shared internally and externally' **covers the data file itself**"*. Full DVRPC Data License read verbatim: warranty disclaimer + three obligations (credit DVRPC; "not entitled to any file revisions, updates, corrections or new releases"; "responsible for understanding the accuracy limitations"). No redistribution/derivative/SA/NC bar | Mostly — carries credit + "as-is without warranty". Missing the **no-updates/staleness** obligation, which matters far more for a frozen mirror snapshot than for a bridge | add snapshot date + "not an official DVRPC distribution; see catalog.dvrpc.org for current data" |
| `de_berlin_roads` | raw-clear | Burndown §4: *"clearly permitted — dl-de/by-2.0 grants **copy/distribute/make-available/modify** incl. commercial (zero-2.0 even freer)"* | **No** — dl-de/by-2.0 requires a **Änderungshinweis** (modification notice) when the data has been changed. Our mirror is normalized/reprojected parquet, i.e. changed. Current string is source-note only | add a modification notice, e.g. "Der Datenbestand wurde verändert (normalisiert, Format- und Schemakonvertierung)." |
| `hk_hongkong_roads` | raw-clear (grant) / **raw-arguable (indemnity)** | *"You are allowed to browse, download, **distribute, reproduce**, hyperlink to, and print **the Data** for both commercial and non-commercial purposes on a free-of-charge basis"*; FAQ: *"**Except re-sale of the data, there is no restriction on the uses of the data**"* | Carries source + IP acknowledgement (good). Missing the **no-resale** pass-through to downstream users | see §5 — the grant is *better* suited to raw mirroring than to ID extraction; the indemnity is what scales |
| `sg_singapore_footpaths` | **raw-arguable** | SODL v1.0: *"use, access, download, copy, **distribute**, transmit, modify and adapt **the datasets, or any derived analyses or applications**, whether commercially or non-commercially"*. **But** the approval note itself records: *"**Downstream Sub-Licensee clause noted: our users need their own LTA licence for the RAW dataset.**"* | Carries conspicuous SODL notice (SODL's stated requirement). Missing the sub-licensee notice | see §5 — **recommend bridge-only until the SODL sub-licence clause is read** |
| `sg_singapore_roads` | **raw-arguable** | Same instruments. The §1.5 approval note leans explicitly on ID minimalism: *"Published ids are **synthetic geometry-hash ids** (a 'derived analysis', expressly distributable)"* — that reasoning covers the bridge and says nothing about the geometry the hashes were derived from | Same gap | same as footpaths |

### Datasets whose recorded rationale used ID-only reasoning

Grepping the burndown for ID-only framing surfaces exactly three places where the
bridge conclusion does **not** transfer for free. All three are in #468 or already
resolved:

- `us_philadelphia_sidewalks` §1.1 — *"an explicit external-sharing grant with only
  attribution attached covers an **ID-only cross-reference a fortiori**"*. The
  a-fortiori step is bridge-shaped, **but** §4 separately concludes the file itself is
  covered by the same "shared externally" language. Resolved.
- `hk_hongkong_roads` §1.4 — *"an **ID-only bridge table** is a free
  reproduction/extraction squarely inside 'distribute/reproduce'"*. Here the raw case
  is *stronger*, not weaker: the grant is literally over "the Data".
- `sg_singapore_*` §1.2/§1.5 — the sub-licensee carve-out and the geometry-hash
  framing. **This is the genuine split.**
- Also: `us_utah_slc_roads`'s attribution *string* (not its rationale) is the
  ID-only claim baked into a shipped artifact.

---

## 3. Config gap analysis

**Does the single flag suffice?** For today's inventory, yes on legality — nothing
approved is raw-problematic. But it fails on three counts that will get worse as the
mirror grows toward "as many datasets as possible":

1. **It cannot express the sg_singapore case** (bridge OK, raw pending a clause read),
   which arrives with #468.
2. **It has no place to record the raw-specific attribution obligations** —
   dl-de/by-2.0's modification notice, ODbL's derivative-database licensing, DVRPC's
   no-updates term, Seattle's accuracy disclaimer, HK's no-resale pass-through. Those
   are conditions on *raw* redistribution that the bridge attribution correctly omits.
3. **It has already let a false attribution ship** (Utah), because one string serves
   two artifacts with opposite content claims.

### Recommendation: implement the already-designed `geometry_status`

Do **not** add a new `target_mirror` field. `docs/plans/2026-07-06-unmatched-export-design.md`
§3 already specifies the exact mechanism, with the vocabulary, the default-deny
semantics, the reviewer checklist, and — critically — the scope decision that it
gates `targets/` too. Implementing a differently-named field now would fork that
design.

**`datasets/licenses.toml`** — two fields per dataset, both optional:

```toml
[datasets.de_berlin_roads]
status = "approved"              # bridge gate (existing, unchanged)
geometry_status = "approved"     # NEW — gates targets/ (and future unmatched/)
license = "dl-de/by-2.0 (…)"
attribution = "Datenquelle: Geoportal Berlin / Detailnetz …"
geometry_attribution = "Datenquelle: Geoportal Berlin / Detailnetz …  Der Datenbestand wurde verändert (normalisiert)."  # NEW, optional; falls back to `attribution`
geometry_note = "dl-de/by-2.0 §2 requires a modification notice; snapshot is normalized parquet."
```

`geometry_attribution` is the second half of the fix and is what actually retires the
Utah defect: it lets the bridge string keep saying "identifiers only" while the mirror
string says what the mirror is.

**Code touch points** (all small, all read-side):

| File | Location | Change |
|---|---|---|
| `src/crosswalk/factory/licenses.py` | `LicenseDecision` dataclass, lines 27–47 | add `geometry_approved: bool = False`, `geometry_attribution: str \| None = None`, `geometry_reason: str \| None = None`; surface all three in `to_dict()` |
| `src/crosswalk/factory/licenses.py` | `decision()`, lines 91–133 | read `entry.get("geometry_status", "pending_review")`; **default-deny** — `geometry_approved` is True only when `status == "approved"` **and** `geometry_status == "approved"` **and** an attribution exists. Resolve `geometry_attribution or attribution`. Bridge behaviour is untouched |
| `src/crosswalk/factory/publish_targets.py` | line 288, `if not decision.approved:` | → `if not decision.approved or not decision.geometry_approved:` with `reason=decision.geometry_reason or decision.reason` |
| `src/crosswalk/factory/publish_targets.py` | `meta` dict, lines 305–317 | `"attribution": decision.geometry_attribution` |
| `src/crosswalk/factory/publish_targets.py` | `index_datasets[name]`, lines 329–336 | same substitution; optionally add `"geometry_note"` |
| `datasets/licenses.toml` | header comment, lines 1–15 | document the second axis (the header currently describes only `status`) |
| `docs/PUBLISHING.md` | "The registry" | document the two-flag gate and the reviewer checklist from the design doc §3 |

`publish.py` (bridges) needs **no change** — that is the point of the orthogonal field.

**Migration:** the default-deny `geometry_status` migration explicitly annotates all
31 bridge-approved entries: 24 are geometry-approved and 7 remain pending review.
The pending set is Hong Kong, Seoul, both Singapore datasets, and all three Montana
datasets.

### Does `index.json` carry per-dataset license + attribution?

**Confirmed yes.** `publish_targets.py:329-336` writes `license`, `attribution`,
`source_url`, `display_name`, `latest_snapshot`, `size_bytes` per dataset, and
`meta.yaml` (lines 305–317) repeats `license` + `attribution` inside each immutable
snapshot dir. Verified against the live staging tree — all 21 entries carry a non-empty
license and attribution; only `gb_london_roads` is missing `source_url` (its registry
entry has none).

Two structural gaps worth noting:

- **Resolved by the geometry-gate implementation:** each published snapshot now has
  an `ATTRIBUTION.txt` alongside its JSON/YAML metadata. This gives conspicuous,
  portable notice for terms such as dl-de/by-2.0 and ODbL.
- **The `[overture]` block is correctly absent** — target snapshots contain no
  Overture-derived content, unlike bridges. No action.

---

## 4. Verdict on the datasets merged by PR #468

**Flip all five to `status = "approved"` — the bridge analysis is sound and none of it
is disturbed by this review.** For the raw mirror:

| Dataset | Raw mirror recommendation |
|---|---|
| `us_philadelphia_sidewalks` | **allow**, with the no-updates/snapshot-date line added to the raw attribution |
| `de_berlin_roads` | **allow**, with the dl-de/by-2.0 modification notice added — this is a licence condition, not a nicety |
| `hk_hongkong_roads` | **pending explicit acceptance**, with the no-resale notice ready to pass through once the indemnity is consciously accepted at raw scale (§5) |
| `sg_singapore_footpaths` | **bridge-only for now** — `status = "approved"`, `geometry_status = "pending_review"` |
| `sg_singapore_roads` | **bridge-only for now** — same |

Since `geometry_status` doesn't exist yet, the interim mechanism for the Singapore hold
is either (a) land `geometry_status` before the next `factory publish --targets`, or
(b) pass `--datasets` explicitly on the targets publish to exclude them. **(a) is
strongly preferred** — an interim exclusion that lives in a shell command is exactly
the failure mode the registry exists to prevent.

---

## 5. Special attention

### `hk_hongkong_roads` — raw is better supported than the bridge, but the indemnity scales

The DATA.GOV.HK grant runs to **"the Data"** itself — *"browse, download, **distribute,
reproduce**, hyperlink to, and print the Data for both commercial and non-commercial
purposes on a free-of-charge basis"*. Mirroring a normalized copy is a paradigm case of
"reproduce and distribute"; the burndown had to work harder to fit an *ID extraction*
into that verb list than a mirror does. Derivative-silence is closed by the FAQ
(*"Except re-sale of the data, there is no restriction on the uses of the data"*).

Two raw-specific deltas:

1. **Re-sale pass-through.** We don't re-sell. But a mirror hands the full dataset to
   third parties who are not obviously on notice of the no-resale condition. The
   attribution should carry it; the bridge never needed to.
2. **Indemnity scales with volume.** The accepted clause is *"You shall **indemnify the
   Government**… against any allegations or claims of infringement… in relation to your
   **use, reproduction and/or distribution** of the Data"*, under HK SAR law. Brad
   accepted this for a bridge table. It reads the same but means more when the thing
   distributed is 36,107 segments of the Government's own geometry served from our
   bucket. **Recommend re-accepting explicitly** rather than inheriting the bridge
   acceptance silently — record it in `geometry_note`.

Not a blocker. Score: **raw-clear on the grant, raw-arguable on risk posture.**

### `sg_singapore_*` — the downstream sub-licensee question is the one real split

SODL v1.0's grant is broad and covers the datasets themselves, not only derived
analyses: *"use, access, download, copy, **distribute**, transmit, modify and adapt
**the datasets**, or any derived analyses or applications, whether commercially or
non-commercially"*. On its face that permits a mirror.

But our own approval note records the carve-out: *"**Downstream Sub-Licensee clause
noted: our users need their own LTA licence for the RAW dataset.**"* That distinction
is *precisely* the bridge/raw boundary:

- **Bridge:** downstream users receive our IDs, not LTA's dataset. The sub-licensee
  clause never engages.
- **Raw mirror:** downstream users receive LTA's dataset from us. If SODL requires each
  recipient to take their own licence from LTA (rather than us sub-licensing onward),
  a public mirror may be distributing to parties who have not accepted SODL — and we
  may not be able to cure that with a notice alone.

The `sg_singapore_roads` note compounds it by reasoning from ID minimalism: *"Published
ids are synthetic geometry-hash ids (a 'derived analysis', expressly distributable)."*
Correct for the bridge, silent on the geometry the hashes were derived from.

**This is not a refusal.** The likely resolution is that SODL's clause is a standard
"your recipients are bound by these terms too" pass-through, which a conspicuous notice
plus a link to the SODL text fully discharges — and burndown §4 already scored both
"clearly permitted". But the clause was flagged by our own reviewer and never read
against the raw case. **Read the SODL sub-licensee clause (one page), then flip
`geometry_status`.** ~10 minutes of human time for 125K segments across the two
datasets.

### `us_utah_slc_roads` — fix before the next publish

Not a licensing problem — CC BY 4.0 plainly permits the mirror. It is a truthfulness
problem: the artifact currently tells downstream consumers that geometry and attributes
were removed, next to a file containing geometry and attributes. Either shorten the
`attribution` (the bridge-scope sentence is not required by CC BY) or, better, add
`geometry_attribution` per §3 and let each artifact describe itself.

---

## 6. Needs web re-verification

Ordered by how much it matters. None blocks the current mirror.

| Dataset | What to check | Why |
|---|---|---|
| `sg_singapore_footpaths`, `sg_singapore_roads` | SODL v1.0 **Downstream Sub-Licensee clause** — does onward distribution of the raw dataset require each recipient to take their own LTA licence, or is a pass-through notice sufficient? | The only genuine bridge/raw split in the inventory; gates 125K segments |
| `us_montana_bozeman`, `us_montana_helena`, `us_montana_missoula` | A citable Montana State Library / MSDI public-domain statement | The registry has a bare unsourced sentence, no `license_url`, no panel review — and two of the three are **already in the mirror** |
| `kr_seoul_roads` | `kogl.or.kr/info/licenseType1.do` — the actual KOGL Type 1 text (commercial use, derivative works, redistribution) | We have the badge, never the licence body. Enters the mirror as soon as its raw parquet is fetched |
| `hk_hongkong_roads` | Current re-sale / FAQ wording, and whether the indemnity clause has changed | Re-accept knowingly at raw scale |
| `gb_london_roads` | OS OpenData attribution year token for the current publish | Flagged at panel time as "must be confirmed at publish time"; still says 2026 |
| `us_seattle_sidewalks` | Nothing new — just carry the accuracy disclaimer the clearance doc told us to carry | Already-recorded instruction not implemented in the attribution string |

---

## 7. Suggested order of work

1. Land the implemented `geometry_status` + `geometry_attribution` gate and the 24/7
   registry migration (§3 touch points).
2. Resolve the cross-mode quality holds before any live targets publish.
3. Read the SODL sub-licensee clause; flip the two Singapore entries if supported.
4. Cite the Montana PD source and the KOGL Type 1 text; flip those four entries.
5. Decide whether to accept the DATA.GOV.HK indemnity at full-mirror scale; if yes,
   flip Hong Kong.
6. Re-run `factory publish --targets` locally, inspect the exact inventory and
   attribution sidecars, then perform the live sync.
