# Fixture Manifest — SYNTHETIC provider-format Fama/French replicas

All fixtures in this directory are **synthetic provider-format replicas**:
deterministically generated from invented values (fixed-seed LCG) using a
builder that mirrors only the structural FORMAT of the provider's ZIP
publications (header lines, section titles, column layout, sentinel
conventions, copyright trailer). **They contain no provider-owned data and
are not redistributions of downloaded Ken French bytes.** The replicas
mirror the provider's publication format as inspected
implementation-time 2026-09-11.

## Licence basis

- Format reference: the provider's public ZIP publication layout
  (upstream copyright line retained in replica form as `Copyright 2026
  Eugene F. Fama and Kenneth R. French` followed by an explicit synthetic
  marker). Values are invented; no provider-owned data is redistributed.
- The adapter (`portlearn.data.adapters.ff`, catalog `factors_monthly_csv`,
  `factors_monthly_txt`, `factors_daily_csv`, `industry49_monthly_csv`)
  marks every decoded dataset **not investable** in provenance; factor
  datasets can never enter an asset-return path (closed `dataset_kind`
  fence `{factor, portfolio_returns}`).

## Retrieval-timestamp semantics

Fixtures are SYNTHETIC: no network retrieval occurred. The deterministic
creation instant recorded by the builder is `2026-09-11T23:30:00Z`
(container member timestamps pinned to `2026-09-11T12:00:00Z` so byte
streams rebuild identically). Tests never touch the network.

## Pins (sha256 of the exact fixture bytes)

| fixture | sha256 (zip container) |
| --- | --- |
| `ff_factors_monthly_csv.zip` | `3d8737ac49bc95bed8e8fa097be07f90533b8ea667ae12e14e7fe37cf73c51c7` |
| `ff_factors_monthly_txt.zip` | `e4c297e7b91e259fb7d5fd4fa5bbbdbeb5b53edb7517fd31c237f05457ed4a2c` |
| `ff_factors_daily_csv.zip` | `74f9cd110c43ca2ae3714a4efdebaf5d0e94f547d6f8eed1fefc6669ff0d160f` |
| `ff_industry49_monthly_csv.zip` | `eb4831ad8e0f32d40ef8a6b1775b2e25a229d52b6d9d835559a515f0465cb9bf` |

Member (inner CSV/TXT) sha256 values, for reference:

- `F-F_Research_Data_Factors.csv` — `de801651bc9f427888e670568c56bbf86627b1cf430359be5da8e89b14d0c0d8`
- `F-F_Research_Data_Factors.txt` — `cd6a72763961a2f4c4e3cfbbdcebeae9828b7810f89333f53c84d914fc04e592`
- `F-F_Research_Data_Factors_daily.csv` — `ea2620ba2b0b0e6434864ce9bb7f7bcd1e3e86b05c8cf2217829a2072840efa3`
- `49_Industry_Portfolios.csv` — `189d9c54302d6da9b531d624e004b6f8243a705262ac58d8337f13178c63d9ee`

## Contents (synthetic values, provider format)

- `ff_factors_monthly_csv.zip` — monthly research factors, comma layout,
  2023-01..2025-12, with `-99.99`/`-999` sentinel cells and an annual
  block; the annual block is never decoded as observations.
- `ff_factors_monthly_txt.zip` — the same monthly values in the
  fixed-width TXT variant; both variants must decode to identical
  observations (format-parity law).
- `ff_factors_daily_csv.zip` — daily research factors, Jul 2026 trading
  days, one sentinel cell.
- `ff_industry49_monthly_csv.zip` — 49 industry portfolio monthly average
  value-weighted returns (decoded section) followed by equal-weighted,
  annual, firm-count, and average-market-cap sections (never decoded),
  with sentinel cells.

Adapter version at pin time: `portlearn.data.adapters.ff` 1 (catalog fixed;
see module docstring).
