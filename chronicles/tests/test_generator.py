import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from chronicles import phrasing as ph
from chronicles.generator import SMALL, SPLITS, WorldConfig, build_world, write_world

SEEDS = [f"t:{i}" for i in range(12)]


def full(split="test"):
    return replace(WorldConfig(), **SPLITS[split])


def names_of(truth, pid):
    return next(p["names"] for p in truth["places"] if p["id"] == pid)


def mentions(text, names):
    return any(n in text for n in names)


class TestWorlds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = [build_world(s, full()) for s in SEEDS] + [build_world(s, full("transfer")) for s in SEEDS[:6]]
        cls.small = [build_world(s, SMALL) for s in SEEDS]

    def test_same_seed_same_world(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            write_world(a, "same", full())
            write_world(b, "same", full())
            for name in ("records.jsonl", "heldout.jsonl", "truth.json"):
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes(), name)
        self.assertNotEqual(build_world("one", full())[0], build_world("two", full())[0])

    def test_sizes_split_and_ids(self):
        for visible, heldout, truth in self.worlds:
            self.assertTrue(180 <= len(visible) <= 460, len(visible))
            cut = truth["heldout_from"]
            self.assertTrue(all(r["year"] < cut for r in visible))
            self.assertTrue(all(r["year"] >= cut for r in heldout))
            ids = [r["id"] for r in visible + heldout]
            self.assertEqual(len(ids), len(set(ids)))
            years = [r["year"] for r in visible]
            self.assertEqual(years, sorted(years))
        for visible, _, _ in self.small:
            self.assertTrue(30 <= len(visible) <= 50, len(visible))

    def test_split_pattern_types(self):
        for _, _, truth in self.worlds[: len(SEEDS)]:
            self.assertNotIn("identity", {p["type"] for p in truth["patterns"]})
            self.assertTrue({"periodic", "cause_effect", "chain"} <= {p["type"] for p in truth["patterns"]})
        for _, _, truth in self.worlds[len(SEEDS):]:
            self.assertIn("identity", {p["type"] for p in truth["patterns"]})

    def test_patterns_are_really_in_the_records(self):
        for visible, heldout, truth in self.worlds + self.small:
            text = {r["id"]: r["text"] for r in visible + heldout}
            year = {r["id"]: r["year"] for r in visible + heldout}
            rec = truth["records"]
            for p in truth["patterns"]:
                ids = p["record_ids"] + p["heldout_record_ids"]
                self.assertTrue(ids, p)
                if p["type"] == "periodic":
                    self.assertGreaterEqual(len(p["record_ids"]), 4)
                    self.assertTrue(p["heldout_years"], "something to predict")
                    sig = p["signature"]
                    recorded = {year[rid] for rid in ids}
                    gaps = [y not in recorded for y in p["occurrence_years"]]
                    self.assertFalse(any(a and b for a, b in zip(gaps, gaps[1:])),
                                     "never two unrecorded occurrences in a row")
                    for rid in ids:
                        self.assertIn(year[rid], p["occurrence_years"])
                        if p["kind"] == "comet":
                            self.assertTrue(mentions(text[rid], ph.DIRECTION_PHRASES[sig["dir"]]), text[rid])
                        else:
                            self.assertTrue(mentions(text[rid], names_of(truth, sig["region"])), text[rid])
                        self.assertTrue(mentions(text[rid].lower(), [c.lower() for c in ph.COLOR_PHRASES[sig["color"]]]),
                                        text[rid])
                elif p["type"] == "cause_effect":
                    self.assertGreaterEqual(len(p["instances"]), 3)
                    self.assertTrue(p["heldout_years"], "something to predict")
                    for inst in p["instances"]:
                        self.assertEqual(inst["effect_year"] - inst["cause_year"], p["params"]["lag"])
                    for rid in ids:
                        kind, slots = rec[rid]["kind"], rec[rid]["slots"]
                        places = [n for k in ("city", "river", "region") if slots.get(k)
                                  for n in names_of(truth, slots[k])]
                        self.assertTrue(mentions(text[rid], places), (kind, text[rid]))
                elif p["type"] == "chain":
                    l1, l2 = p["params"]["lags"]
                    for inst in p["instances"]:
                        a, b, c = inst["years"]
                        self.assertEqual((b - a, c - b), (l1, l2))
                    self.assertTrue(p["heldout_years"], "something to predict")
                else:  # identity: every record names the person by one of the aliases
                    self.assertGreaterEqual(len(ids), 4)
                    aliases = [a.lower() for a in p["params"]["aliases"]]
                    for rid in ids:  # case-insensitive: a terse sentence may start "The Lame ..."
                        self.assertTrue(mentions(text[rid].lower(), aliases), text[rid])
                    linking = [rid for rid in ids if any(a in text[rid].lower() for a in aliases[1:])
                               and ("whom men called" in text[rid] or ", Lord of" in text[rid])]
                    self.assertGreaterEqual(len(linking), 2, "aliases are tied together somewhere")

    def test_decoys_predict_nothing(self):
        for visible, heldout, truth in self.worlds + self.small:
            rec = truth["records"]
            for d in truth["decoys"]:
                self.assertFalse(d["heldout_record_ids"])
                if d["type"] == "periodic":
                    city = rec[d["record_ids"][0]]["slots"]["city"]
                    moons = sorted({r["year"] for r in visible + heldout
                                    if rec[r["id"]]["kind"] == "red_moon" and rec[r["id"]]["slots"]["city"] == city})
                    self.assertEqual(len(moons), 3, "the run stops: no fourth red moon over that city, ever")
                elif d["type"] == "cause_effect":
                    city = rec[d["record_ids"][0]]["slots"]["city"]
                    fairs = {r["year"] for r in visible if rec[r["id"]]["kind"] == "fair"
                             and rec[r["id"]]["slots"]["city"] == city}
                    fires = {r["year"] for r in visible + heldout if rec[r["id"]]["kind"] == "fire"
                             and rec[r["id"]]["slots"]["city"] == city}
                    self.assertEqual(len(fires), 2)
                    self.assertGreaterEqual(len([y for y in fairs if y + 1 not in fires]), 3)

    def test_noise_never_imitates_the_planted_signature(self):
        for visible, heldout, truth in self.worlds:
            rec = truth["records"]
            for p in truth["patterns"]:
                if p["type"] == "periodic" and p["kind"] == "comet":
                    sig = p["signature"]
                    for rid, info in rec.items():
                        if info["kind"] == "comet" and p["id"] not in info["tags"]:
                            self.assertNotEqual(info["slots"]["dir"], sig["dir"])
                            self.assertNotEqual(info["slots"]["color"], sig["color"])

    def test_the_map_can_be_learned_from_the_records(self):
        """Every city used by a cause/effect pattern is sometimes written with its region or river."""
        for visible, _, truth in self.worlds:
            places = {p["id"]: p for p in truth["places"]}
            for p in truth["patterns"]:
                if p["type"] != "cause_effect":
                    continue
                effect_ids = [rid for rid in p["record_ids"] if p["id"] + ":effect" in truth["records"][rid]["tags"]]
                city = places[truth["records"][effect_ids[0]]["slots"]["city"]]
                region, river = places[city["region"]], places[city["river"]]
                together = [r for r in visible if mentions(r["text"], [f"{c} in {g}" for c in city["names"]
                                                                       for g in region["names"]]
                                                           + [f"{c} on the {v}" for c in city["names"]
                                                              for v in river["names"]])]
                self.assertTrue(together, f"{city['names']} is never placed on the map")

    def test_invented_names_are_fresh_per_world(self):
        a = {n for p in self.worlds[0][2]["places"] for n in p["names"]}
        b = {n for p in self.worlds[1][2]["places"] for n in p["names"]}
        self.assertLess(len(a & b), 3)

    def test_files_on_disk(self):
        with TemporaryDirectory() as d:
            truth = write_world(d, "disk", SMALL)
            rows = [json.loads(line) for line in (Path(d) / "records.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), truth["counts"]["visible"])
            self.assertEqual(set(rows[0]), {"id", "year", "chronicler", "place", "text"})


if __name__ == "__main__":
    unittest.main()
