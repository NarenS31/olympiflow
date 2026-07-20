# Schema differences: METR-LA / PEMS-BAY vs Chicago

> For the cross-DOMAIN jump (traffic → IEEE 14-bus power grid, Phase 19) see
> [`DIFFERENCES_POWER_GRID.md`](DIFFERENCES_POWER_GRID.md). This file covers the
> cross-CITY differences only.

This file documents every place the Chicago pipeline had to diverge from the two
HDF5 point-sensor datasets. It exists for the paper's **cross-city generalization**
discussion — a model trained on LA runs on Chicago only because we forced both into
one tensor contract, and the honest story is *what we had to do to make that true*.

## Output contract (identical on purpose)
All three produce, per split, `.npz` files with:
- `X` : `[samples, 12, N, 2]` — features (channel 0 = speed z-scored, channel 1 = time-of-day)
- `Y` : `[samples, 12, N]`    — z-scored speed target
plus `adjacency.npy [N,N]`, `scaler.json`, `node_meta.json`, `stats.json`.

| Aspect | METR-LA / PEMS-BAY | Chicago |
|---|---|---|
| Node definition | Physical loop-detector **point** sensor | Road **segment** (a stretch of street between two cross-streets) |
| Node count N | 207 / 325 (fixed) | Variable — only segments present in both metadata and the history window |
| Raw source | Static `.h5` DataFrame `[T × N]` from DCRNN release | Socrata REST API (live + historical estimates), pulled per run |
| Time grid | Native 5-min, regular | Irregular update times → **resampled** to 5-min via mean aggregation |
| Missing-value code | `0.0` in the raw file | `-1` (Socrata "no estimate") → NaN → `0.0`; plus resample gaps → `0.0` |
| Adjacency source | Precomputed **road-network distance** edge list → thresholded Gaussian kernel | No distance file → **binary shared-endpoint** graph (endpoints within `segment_adjacency_tol_m` = 50 m) |
| Edge semantics | Weighted in (0,1], drive-distance decay | Unweighted 0/1, topological adjacency at intersections |
| Node names | Reverse-geocoded from sensor lat/lon (METR-LA ships locations; PEMS-BAY does not) | Street + from→to cross-streets, directly from segment metadata (richest names of the three) |
| Coverage window | ~4 months (LA) / ~6 months (Bay), 2012/2017 | Most-recent Socrata page (`page_limit` rows, `time DESC`) at build time — recent, not historical-fixed |

## Consequences the model must tolerate (wired in Phase 2)
1. **Different N and different graph density.** The ST-GNN cannot assume a fixed
   node count; the adaptive/semantic adjacency (Modification 1) is what lets it
   transfer. The physical adjacency is simply whatever `adjacency.npy` holds.
2. **Binary vs weighted physical edges.** Chicago's `A_physical` is 0/1 while LA's
   is a decayed kernel. The learnable `alpha` mixing physical vs semantic adjacency
   should absorb this — worth watching in the cross-city experiments (Phase 6).
3. **Feature parity, not feature identity.** Both carry (speed, time-of-day). The
   heterogeneous modalities (weather/events/transit) are stubbed to zero for now
   and gated off per-modality, so a missing modality never crashes transfer.

## Honest limitations to state in the paper
- Chicago's segment speeds are modeled *estimates*, not measured loop speeds — a
  domain gap beyond just geography.
- The pipeline pulls the **most-recent** Socrata page (ordered by `time DESC`,
  `page_limit` rows) rather than a fixed calendar window — the portal feed can go
  stale (as of this build the latest data was ~2026-04-30), and a fixed "last N
  days" filter silently returns nothing when that happens. Phase-1 build yielded
  ~99 five-minute steps / 1,020 segments at ~66% missing: enough to verify the
  tensor contract, far too little to train on. Phase 6 must widen this (raise
  `page_limit`, or paginate a real multi-week span) before cross-city evaluation.
- Shared-endpoint adjacency at 50 m is a geometric proxy for true road connectivity
  and can miss grade-separated / one-way nuances.
