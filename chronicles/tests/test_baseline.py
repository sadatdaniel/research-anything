"""Phase 1 go/no-go: the dumb keyword baseline must NOT be able to solve the worlds."""

import unittest
from dataclasses import replace

from chronicles.baseline_b2 import run_b2
from chronicles.generator import SPLITS, WorldConfig, build_world
from chronicles.scorer import aggregate, score_world


class TestBaselineB2(unittest.TestCase):
    def test_keyword_baseline_finds_under_30_percent(self):
        cfg = replace(WorldConfig(), **SPLITS["test"])
        scores = []
        for i in range(8):
            visible, _, truth = build_world(f"b2:{i}", cfg)
            scores.append(score_world(truth, run_b2(visible)))
        agg = aggregate(scores)
        self.assertLess(agg["recall"], 0.30, agg)
        self.assertGreater(agg["claims"], 0, "the baseline does try")


if __name__ == "__main__":
    unittest.main()
