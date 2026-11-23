import unittest
from dataclasses import replace

from chronicles.generator import SPLITS, WorldConfig, build_world
from chronicles.scorer import aggregate, score_world, truth_as_claims


def world(seed="s:1", split="transfer"):
    visible, heldout, truth = build_world(seed, replace(WorldConfig(), **SPLITS[split]))
    return visible, truth


class TestScorer(unittest.TestCase):
    def test_perfect_claims_score_perfectly(self):
        for seed in ("s:1", "s:2", "s:3"):
            _, truth = world(seed)
            s = score_world(truth, truth_as_claims(truth))
            self.assertEqual(s["recall"], 1.0)
            self.assertEqual(s["precision"], 1.0)
            self.assertEqual(s["predictions_correct"], s["predictions"])
            self.assertGreater(s["predictions"], 0)

    def test_no_claims(self):
        _, truth = world()
        s = score_world(truth, [])
        self.assertEqual((s["recall"], s["precision"], s["claims"]), (0.0, 0.0, 0))

    def test_listing_everything_does_not_win(self):
        visible, truth = world()
        everything = [r["id"] for r in visible]
        claims = [{"type": t, "record_ids": everything, "params": {"period": p["params"].get("period"),
                   "lag": p["params"].get("lag"), "lags": p["params"].get("lags")}}
                  for p in truth["patterns"] for t in [p["type"]]]
        self.assertEqual(score_world(truth, claims)["found"], 0)

    def test_parameters_must_be_right(self):
        _, truth = world()
        p = next(x for x in truth["patterns"] if x["type"] == "periodic")
        base = {"type": "periodic", "record_ids": p["record_ids"]}
        period = p["params"]["period"]
        self.assertEqual(score_world(truth, [{**base, "params": {"period": period + 1}}])["found"], 1)
        self.assertEqual(score_world(truth, [{**base, "params": {"period": period + 15}}])["found"], 0)
        self.assertEqual(score_world(truth, [{**base, "params": {}}])["found"], 0)

    def test_decoys_and_duplicates_are_counted(self):
        _, truth = world()
        d = next(x for x in truth["decoys"] if x["type"] == "periodic")
        p = next(x for x in truth["patterns"] if x["type"] == "periodic")
        claims = [{"type": "periodic", "record_ids": d["record_ids"], "params": d["params"]},
                  {"type": "periodic", "record_ids": p["record_ids"], "params": p["params"]},
                  {"type": "periodic", "record_ids": p["record_ids"][:-1], "params": p["params"]}]
        s = score_world(truth, claims)
        self.assertEqual((s["decoy_hits"], s["found"], s["duplicates"]), (1, 1, 1))
        self.assertAlmostEqual(s["precision"], 2 / 3)

    def test_predictions_must_be_about_the_hidden_future(self):
        _, truth = world()
        p = next(x for x in truth["patterns"] if x["type"] == "periodic")
        claim = {"type": "periodic", "record_ids": p["record_ids"], "params": p["params"]}
        right = score_world(truth, [{**claim, "prediction": {"year": p["heldout_years"][0]}}])
        wrong = score_world(truth, [{**claim, "prediction": {"year": p["heldout_years"][0] + 30}}])
        past = score_world(truth, [{**claim, "prediction": {"year": truth["heldout_from"] - 50}}])
        self.assertEqual((right["predictions_correct"], wrong["predictions_correct"], past["predictions_correct"]),
                         (1, 0, 0))

    def test_periodic_prediction_allows_for_the_jitter_on_both_ends(self):
        _, truth = world()
        p = next(x for x in truth["patterns"] if x["type"] == "periodic")
        claim = {"type": "periodic", "record_ids": p["record_ids"], "params": p["params"]}
        tol = max(2 * p["jitter"], 0.05 * p["params"]["period"])
        y = p["heldout_years"][0]
        edge = score_world(truth, [{**claim, "prediction": {"year": y - int(tol)}}])
        beyond = score_world(truth, [{**claim, "prediction": {"year": y - int(tol) - 3}}])
        self.assertEqual((edge["predictions_correct"], beyond["predictions_correct"]), (1, 0))

    def test_aggregate(self):
        scores = [score_world(t, truth_as_claims(t)) for t in (world(s)[1] for s in ("a", "b"))]
        agg = aggregate(scores)
        self.assertEqual(agg["recall"], 1.0)
        self.assertEqual(agg["per_type"]["identity"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
