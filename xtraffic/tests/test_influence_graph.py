"""Unit tests for evaluation/influence_graph.py.

The audit's item 7 is "write unit tests before touching the metric". This module
IS a metric — it turns 4,968 explainer solves into precision/recall numbers that
a reader cannot check by eye — so the aggregation, thresholding, direction
convention and hop logic are tested on hand-built graphs where the right answer
is known by construction.

The direction convention gets its own test because it is the one thing here that
would produce entirely plausible, entirely wrong numbers if transposed.
"""
from __future__ import annotations

import numpy as np
import pytest

from xtraffic.evaluation import influence_graph as ig


# ---------------------------------------------------------------------------
# A tiny hand-built graph: 0 -> 1 -> 2 -> 3, plus isolated node 4.
# ---------------------------------------------------------------------------
def _chain_geometry(n: int = 5) -> ig.Geometry:
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(3):
        A[i, i + 1] = 0.5                     # directed i -> i+1
    np.fill_diagonal(A, 1.0)                  # self-loops, as the pipeline does
    road = np.full((n, n), np.inf)
    np.fill_diagonal(road, 0.0)
    for i in range(3):
        road[i, i + 1] = 1000.0 * (i + 1)
    latlon = np.array([[34.0 + 0.01 * i, -118.0 - 0.01 * i] for i in range(n)])
    return ig.Geometry(A, road, ig.haversine_matrix(latlon), latlon, list(range(n)))


class TestGeometry:
    def test_adjacency_excludes_diagonal(self):
        g = _chain_geometry()
        assert not g.adj.diagonal().any(), "self-loops must not be edges"
        assert g.adj.sum() == 3

    def test_hops_follow_edge_direction(self):
        g = _chain_geometry()
        assert g.hops[0, 3] == 3               # 0->1->2->3
        assert g.hops[3, 0] == ig.Geometry.UNREACHABLE   # no reverse path
        assert g.hops[0, 0] == 0
        assert g.hops[0, 4] == ig.Geometry.UNREACHABLE   # isolated

    def test_direction_convention_source_feeds_target(self):
        """A[s, t] > 0 must mean 's feeds t', matching GraphConv._nconv's
        einsum('bcnt,nm->bcmt') which sums A[n, m] into output node m."""
        g = _chain_geometry()
        assert g.adj[0, 1] and not g.adj[1, 0]
        pairs = g.adjacency_pairs([1])
        assert pairs == {(0, 1)}, "target 1's only in-edge is from source 0"

    def test_cutoff_recovered_from_sigma(self):
        g = _chain_geometry()
        assert g.cutoff_m == pytest.approx(g.sigma_m * np.sqrt(-np.log(ig.KAPPA)))

    def test_haversine_symmetric_and_zero_diagonal(self):
        g = _chain_geometry()
        assert np.allclose(g.hav_m, g.hav_m.T)
        assert np.allclose(np.diag(g.hav_m), 0.0)


class TestRegimeLabels:
    def test_thresholds_and_dead_band(self):
        speeds = np.array([0.0, 0.5, 20.0, 44.9, 45.0, 50.0, 54.9, 55.0, 70.0])
        lab = ig.regime_labels(speeds)
        assert list(lab) == [
            "", "",                              # missing sentinel (<= 1.0 mph)
            ig.REGIME_CONGESTED, ig.REGIME_CONGESTED,
            "", "", "",                          # dead band 45..55 excluded
            ig.REGIME_FREEFLOW, ig.REGIME_FREEFLOW,
        ]

    def test_sentinel_never_counts_as_congested(self):
        """The Phase-3 bug in mirror image: a 0 mph sentinel is missing data, not
        the most congested sensor in the city."""
        assert ig.regime_labels(np.array([0.0]))[0] == ""


class TestWindowSelection:
    def _fake(self, S=600, N=10, seed=0):
        rng = np.random.RandomState(seed)
        speeds = rng.uniform(10, 70, size=(S, N))
        tod = np.linspace(0, 1, S, endpoint=False)
        return speeds, tod

    def test_deterministic_for_a_seed(self):
        sp, tod = self._fake()
        a, sa = ig.select_windows(sp, tod, 24, seed=42)
        b, sb = ig.select_windows(sp, tod, 24, seed=42)
        assert np.array_equal(a, b) and np.array_equal(sa, sb)

    def test_different_seeds_differ(self):
        sp, tod = self._fake()
        a, _ = ig.select_windows(sp, tod, 24, seed=42)
        b, _ = ig.select_windows(sp, tod, 24, seed=7)
        assert not np.array_equal(a, b)

    def test_cells_are_balanced(self):
        sp, tod = self._fake()
        _, strata = ig.select_windows(sp, tod, 24, seed=42)
        counts = np.bincount(strata)
        assert len(counts) == 12 and (counts == 2).all()

    def test_windows_are_unique(self):
        sp, tod = self._fake()
        idx, _ = ig.select_windows(sp, tod, 24, seed=42)
        assert len(set(idx.tolist())) == len(idx)

    def test_unbalanceable_design_raises(self):
        """An indivisible design would silently produce an unbalanced split-half."""
        sp, tod = self._fake()
        with pytest.raises(ValueError, match="not divisible"):
            ig.select_windows(sp, tod, 25, seed=42)


class TestSplitHalves:
    def test_disjoint_exhaustive_and_balanced(self):
        strata = np.repeat(np.arange(12), 2)
        a, b = ig.split_halves(strata, seed=42)
        assert set(a).isdisjoint(set(b))
        assert sorted(a.tolist() + b.tolist()) == list(range(24))
        assert len(a) == len(b) == 12
        # one window from each stratum on each side
        assert sorted(strata[a]) == list(range(12))
        assert sorted(strata[b]) == list(range(12))


class TestBuildW:
    def test_mean_over_regime_windows_only(self):
        imp = np.array([[1.0, 0.0], [3.0, 0.0], [9.0, 9.0]])       # 3 windows
        reg = np.array([ig.REGIME_CONGESTED, ig.REGIME_CONGESTED, ig.REGIME_FREEFLOW],
                       dtype=object)
        W, n = ig.build_W({0: imp}, {0: reg}, ig.REGIME_CONGESTED, 2, min_windows=2)
        assert W[0, 0] == pytest.approx(2.0)                       # (1+3)/2
        assert n[0] == 2

    def test_min_windows_floor_leaves_row_empty_and_counted(self):
        imp = np.array([[1.0, 2.0]])
        reg = np.array([ig.REGIME_CONGESTED], dtype=object)
        W, n = ig.build_W({0: imp}, {0: reg}, ig.REGIME_CONGESTED, 2, min_windows=4)
        assert n[0] == 0 and W[0].sum() == 0.0
        assert len(ig.active_targets(n)) == 0


class TestThresholding:
    def test_top_k_excludes_self_and_respects_k(self):
        W = np.zeros((4, 4))
        W[0] = [99.0, 0.5, 0.3, 0.1]           # target 0's own mass is the largest
        e = ig.edges_top_k(W, [0], k=2)
        assert e == {(1, 0), (2, 0)}, "self must never enter its own edge set"

    def test_top_k_drops_zero_importance_sources(self):
        W = np.zeros((4, 4))
        W[0] = [0.0, 0.7, 0.0, 0.0]
        assert ig.edges_top_k(W, [0], k=3) == {(1, 0)}

    def test_cumulative_mass_takes_smallest_covering_set(self):
        W = np.zeros((5, 5))
        W[0] = [0.0, 0.6, 0.2, 0.1, 0.1]       # total 1.0 off-self
        e = ig.edges_cumulative_mass(W, [0], frac=0.75)
        assert e == {(1, 0), (2, 0)}           # 0.6 then 0.8 >= 0.75, stop

    def test_cumulative_mass_ignores_diagonal(self):
        W = np.zeros((3, 3))
        W[0] = [100.0, 0.6, 0.4]
        e = ig.edges_cumulative_mass(W, [0], frac=0.5)
        assert (0, 0) not in e and e == {(1, 0)}

    def test_self_mass_is_diagonal_share(self):
        W = np.zeros((2, 2))
        W[0] = [3.0, 1.0]
        assert ig.self_mass(W, [0])[0] == pytest.approx(0.75)


class TestComparison:
    def test_precision_recall_jaccard(self):
        g = _chain_geometry()
        # target 1 truly has {0}; claim {0, 4} -> 1 tp, 1 fp, 0 fn
        m = ig.compare_to_adjacency({(0, 1), (4, 1)}, g, [1])
        assert m["n_reference"] == 1
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(1.0)
        assert m["jaccard"] == pytest.approx(0.5)
        assert m["n_off_adjacency"] == 1

    def test_reference_is_only_the_evaluated_targets(self):
        """Recall must not be charged against edges of targets we never scored."""
        g = _chain_geometry()
        m = ig.compare_to_adjacency({(0, 1)}, g, [1])
        assert m["n_reference"] == 1 and m["recall"] == pytest.approx(1.0)

    def test_jaccard_helper(self):
        assert ig.jaccard({(1, 2)}, {(1, 2)}) == pytest.approx(1.0)
        assert ig.jaccard({(1, 2)}, {(3, 4)}) == pytest.approx(0.0)


class TestBaselines:
    def test_uniform_never_picks_self_and_is_seeded(self):
        g = _chain_geometry()
        a = ig.baseline_uniform(g, [0, 1], k=2, seed=1)
        b = ig.baseline_uniform(g, [0, 1], k=2, seed=1)
        assert a == b
        assert not any(s == t for (s, t) in a)

    def test_degree_matched_never_picks_self(self):
        g = _chain_geometry()
        e = ig.baseline_degree_matched(g, [0, 1, 2], k=2, seed=3)
        assert not any(s == t for (s, t) in e)

    def test_nearest_road_uses_road_only_and_reports_shortfall(self):
        """Only node 2 has a finite road distance INTO node 3, so asking for 8
        sources must yield 1 edge and count the shortfall — never silently
        substituting haversine, which is finite for every pair."""
        g = _chain_geometry()
        e, info = ig.baseline_nearest_road(g, [3], k=8)
        assert e == {(2, 3)}
        assert info["targets_with_fewer_than_k_road_sources"] == 1

    def test_nearest_road_is_deterministic(self):
        g = _chain_geometry()
        assert ig.baseline_nearest_road(g, [1, 2, 3], k=2)[0] == \
            ig.baseline_nearest_road(g, [1, 2, 3], k=2)[0]


class TestHopStratification:
    def test_bucket_labels(self):
        assert ig.hop_bucket(0) == "self"
        assert ig.hop_bucket(1) == "1"
        assert ig.hop_bucket(8, k_hops=8) == "8"
        assert ig.hop_bucket(9, k_hops=8) == ">8"
        assert ig.hop_bucket(ig.Geometry.UNREACHABLE) == "unreachable"

    def test_stratify_orders_buckets_by_hop(self):
        g = _chain_geometry()
        by = ig.stratify_by_hop({(0, 1), (0, 3), (4, 1)}, g, k_hops=8)
        assert list(by.keys()) == ["1", "3", "unreachable"]

    def test_beyond_k_catches_unreachable_and_far(self):
        g = _chain_geometry()
        got = ig.beyond_k_set({(0, 1), (0, 3), (4, 2)}, g, k_hops=2)
        assert got == {(0, 3), (4, 2)}         # 3 hops, and unreachable

    def test_beyond_k_respects_k(self):
        g = _chain_geometry()
        assert ig.beyond_k_set({(0, 3)}, g, k_hops=8) == set()


class TestEdgeRows:
    def test_road_and_haversine_are_separate_columns(self):
        g = _chain_geometry()
        W = np.zeros((5, 5))
        W[1, 0] = 0.9                          # target 1, source 0
        W[2, 4] = 0.4                          # target 2, source 4 (no road link)
        rows = {(r["source"], r["target"]): r
                for r in ig.edge_rows({(0, 1), (4, 2)}, W, g, "congested")}
        near = rows[(0, 1)]
        assert near["has_road_distance"] and near["road_distance_m"] == 1000.0
        far = rows[(4, 2)]
        assert far["has_road_distance"] is False
        assert far["road_distance_m"] is None, "haversine must not fill the gap"
        assert far["haversine_m_reference_only"] > 0

    def test_rows_sorted_by_importance_descending(self):
        g = _chain_geometry()
        W = np.zeros((5, 5))
        W[1, 0], W[1, 2] = 0.2, 0.8
        rows = ig.edge_rows({(0, 1), (2, 1)}, W, g, "congested")
        assert [r["source"] for r in rows] == [2, 0]
