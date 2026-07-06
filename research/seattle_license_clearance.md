# Seattle sidewalks license clearance — adversarial hunt (2026-07-06)

Decision rule from Brad: treat City of Seattle open data as public-domain-equivalent
and publish, UNLESS a dedicated adversarial hunt finds concrete evidence against that
treatment. This document records the hunt (Opus research agent, 37 tool calls, live
browser rendering of the JS-only ToU page) and its verdict.

## VERDICT: NO BLOCKING EVIDENCE — approved

## The adversarial near-miss (and why it does not block)

A 2014 open-data critique (opengovdata.io) quotes Seattle's **old** `/data-policy`
terms (per the author, unchanged since 2010) reserving the right to *"require the
termination of any and all displaying, distributing or otherwise using any or all of
the data for any reason."* That is a genuine downstream-use restriction — exactly the
"older restrictive terms contradicting the newer policy" risk.

The **current** operative ToU (rendered in a browser; the live page is a JS Socrata
story) has **removed and replaced** that clause with a much narrower one:

> "**Right to Modify or Discontinue Datasets** … The City of Seattle reserves the
> right to discontinue **providing** any or all datasets and associated data products,
> visualizations, etc. at any time without prior notice."

This only lets the City stop *publishing* upstream. It contains no right to force
downstream users to cease displaying/distributing/using data they already hold. The
current terms impose no redistribution or derivative-works restriction.

## Full current ToU (data.seattle.gov story 6ukr-wvup), operative clauses

- Commercial restriction scoped only to personal lists: *"To the extent the data
  consists of a list of individuals or can be readily sorted, filtered, or configured
  as a list of individuals, it is not to be used for a commercial purpose."* —
  categorically inapplicable to sidewalk centerlines or an ID→GERS table.
- Warranty/accuracy disclaimer + use-at-own-risk (not a redistribution restriction).
- Attribution optional: *"Unless otherwise indicated, data on this site does not
  require specific attribution."*
- "Discontinue providing" clause (above).

## Per-layer license on the actual serving site

The dataset is served from ArcGIS Hub (`data-seattlecitygis.opendata.arcgis.com`).
Authoritative item metadata (item `20abc2269f6f4283a53cf31b93de718f`):
`licenseInfo` = the accuracy disclaimer only ("The City of Seattle makes no
representation or warranty as to its accuracy…"); `access` = public;
`accessInformation` = "City of Seattle, Seattle Department of Transportation."
The Hub's "Custom License" label is only that disclaimer — no reuse restriction.
(The burndown panel's earlier debunking of the stray "PDDL" search artifact stands:
there is no named open license and no CC0/PDDL dedication.)

## Other vectors checked

- **Open Data Policy PDF (v1.0, Feb 2016):** defines the program's "open license" as
  no restrictions on copying, publishing, distributing, modifying, or
  commercial/noncommercial use; Seattle is "open by preference." Corroborates intent.
- **Washington copyright (RCW 42.56):** the Public Records Act compels disclosure but
  does not waive copyright, and WA has no statute placing municipal records in the
  public domain — so PD-equivalent treatment rests on the City's own open-data grant,
  not automatic uncopyrightability. A nuance, not a blocker; sidewalk locations are
  facts with at most thin compilation copyright, of which an ID-only table copies
  essentially none.
- **Precedent (strong FOR):** the OSM **Seattle Sidewalk Import** wiki records this
  exact SDOT sidewalk data as *"Data license: public domain with attribution"* with
  *"ODbL Compliance verified: yes"* (confirmed in raw wikitext). A documented
  third-party import of the same data with license review; no takedown found. Also
  cataloged on data.gov.
- **Access gates:** FeatureServer is public, Query + Extract enabled, no token, no
  internal-use markings.

## Scope: covers geometry too

The governing terms contain no redistribution or derivative-works restriction of any
kind, so the future unmatched-geometry export is equally clear under the same ToU.
Carry forward: (1) pass through the City's accuracy/warranty disclaimer in geometry
artifacts; (2) the "discontinue providing" clause means upstream hosting can stop, but
that does not retroactively restrict already-published derivatives.

## Registry entry

- License label: `City of Seattle Open Data Terms of Use — public-domain-equivalent
  (no attribution required, no share-alike; warranty/accuracy disclaimer only)`
- Attribution (courtesy, not required): "Contains data from the City of Seattle,
  Seattle Department of Transportation (SDOT), via the Seattle Open Data program —
  used under the City of Seattle Open Data Terms of Use."
- ODbL co-publication: no conflict (no attribution mandate, no copyleft).

## Key citations

1. Current operative ToU: https://data.seattle.gov/stories/s/Data-Policy/6ukr-wvup/
2. Item metadata (licenseInfo = disclaimer only):
   https://www.arcgis.com/sharing/rest/content/items/20abc2269f6f4283a53cf31b93de718f?f=json
   and the public FeatureServer:
   https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/Sidewalks_(Active)/FeatureServer/0
3. OSM import precedent:
   https://wiki.openstreetmap.org/wiki/Seattle,_Washington/Sidewalk_Import
4. Superseded 2010–2014 termination clause (for the record):
   https://opengovdata.io/2014/no-discrimination-license-free/
