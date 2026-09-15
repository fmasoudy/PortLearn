# Fixture Manifest — SYNTHETIC provider-format FRED replicas

All fixtures in this directory are **synthetic provider-format replicas**:
deterministically generated from invented values (fixed-seed LCG) using a
builder that mirrors only the structural FORMAT of the FRED API JSON
responses (observations payload shape with `Realtime start`/`Realtime end`
window labels and per-observation `date`/`realtime_start`/`realtime_end`
/`value` cells, series-metadata payload shape, provider error-payload
shape, and the `.` missing-value marker). **They contain no provider-owned
data and are not redistributions of downloaded FRED bytes.** The
replicas mirror the provider's API response format as inspected
implementation-time 2026-09-11.

## Licence basis

- Format reference: the provider's public API response layout (upstream
  attribution notice reproduced in replica form inside the metadata
  fixtures: "This product uses the FRED(R) API but is not endorsed by or
  certified by the Federal Reserve Bank of St. Louis" followed by an
  explicit synthetic marker). Values are invented; no provider-owned data
  is redistributed.
- The adapter (`portlearn.data.adapters.fred`) carries the FRED attribution
  notice in every provenance and marks every decode with its data mode
  from the closed set `{CURRENT_SNAPSHOT, POINT_IN_TIME}`;
  synthetic fixtures decode strictly as `CURRENT_SNAPSHOT` retrievals
  whose availability is floored at the retrieval instant.

## Key handling

Fixtures are SYNTHETIC and **key-free by construction**: no api-key
material (real or fake) appears in any fixture, in any provenance field,
or anywhere in this repository. Fetch-time keys are read only from the
environment at research time and never enter any committed artifact. An
automated key-absence scan runs in the test suite (L9 node).

## Retrieval-timestamp semantics

Fixtures are SYNTHETIC: no network retrieval occurred. The deterministic
creation instant recorded by the builder is `2026-09-12T00:55:00Z`; the
synthetic realtime window pinned in every observations payload is
`2026-09-12..2026-09-12` (a single snapshot day, deliberately matching
the `CURRENT_SNAPSHOT` mode the fixtures exercise). Tests never touch the
network.

## Pins (sha256 of the exact fixture bytes)

| fixture | sha256 |
| --- | --- |
| `obs_synthcpim_monthly.json` | `141eee56eef451a23e0ea232e30eb6eaf211acf75b0003b2aa9cd5db420c2639` |
| `meta_synthcpim.json` | `edc878ff39c6efcdb978ba8fb21e07be5b1684dab6fe8112d9c9fc439cf6d16a` |
| `obs_synthgdpq_quarterly.json` | `dec55f04df139de8e830f20645123406f473b327fdb4f5d10b42bda374cfba17` |
| `meta_synthgdpq.json` | `b78b5967639894ec34beb3d71a53803e47aca02690597db2299265e6f8f963cd` |
| `obs_synthdffd_daily.json` | `b9a70b396bf4f77617ae72419b11c39433b7e3c37b8a92c32612c1eb63ff0c0d` |
| `meta_synthdffd.json` | `77a2de82eb93abe80884685c16085699e8c81868fc894201cdf62c9d9e8821d7` |
| `obs_unknown_series.json` | `f6795c3501492a4d87a8485d2f08f142de47822cd0aa4fbf615c57c5edc143ab` |

## Contents (synthetic values, provider format)

- `obs_synthcpim_monthly.json` — monthly synthetic CPI-like series
  `SYNTHCPIM` (2023-01..2025-12, 36 period-start dates), `.` missing
  markers at 2024-03 and 2025-11, single-day realtime window.
- `obs_synthgdpq_quarterly.json` — quarterly synthetic GDP-like series
  `SYNTHGDPQ` (2022Q1..2025Q4, 16 period-start dates), `.` missing marker
  at 2023Q3.
- `obs_synthdffd_daily.json` — daily synthetic funds-rate-like series
  `SYNTHDFFD` (2026-07-01..10, weekday dates incl. a weekend day to prove
  day-last mapping without trading-calendar inference), `.` missing marker
  at 2026-07-08.
- `meta_*.json` — series-metadata payloads (id/title/units/frequency/
  seasonal_adjustment + attribution note) proving units/frequency ride as
  provenance metadata, never converted.
- `obs_unknown_series.json` — provider error payload for an unknown
  series id (L10 fail-closed path).

Adapter version at pin time: `portlearn.data.adapters.fred` 1 (mode fence and
catalog frozen; see module docstring).
