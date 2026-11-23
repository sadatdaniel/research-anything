"""Grade the learner's notes against a world's per-record answer key (truth.json["records"]).

What is measured, per note:
  event     the free-text event label maps to the right kind (keyword groups; a few near-synonyms
            are accepted, e.g. a dike break called a "flood")
  places    recall: share of the real names in the record that the note lists;
            precision: share of listed "places" that really are names written in the record
  links     recall: share of the record's stated links ("A (called B by others)", "A in B", "A on the B") that
            the note captured and kept; precision: share of kept links (those that pass the source check)
            that are true of the world. link_precision_all also counts the inferred links that were dropped.
  details   for comets and locusts (the periodic kinds): colour, direction and duration captured
  quiz      the quiz answer appears word for word in the record
"""

from __future__ import annotations

import re

from chronicles import phrasing as ph

# First match wins, so the more specific groups come first.
EVENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("falling_stars", ("falling star", "shooting star", "meteor", "stars fell", "star shower", "shower")),
    ("new_star", ("new star", "strange star", "stranger", "nova")),
    ("red_moon", ("moon", "lunar")),
    ("eclipse", ("eclipse", "darken", "sun hid", "dark sun", "sun grew dark")),
    ("comet", ("comet", "star", "torch", "light", "sword", "broom", "beard", "tail")),
    ("dike_break", ("dike", "dyke", "sea wall", "breach", "sea broke", "salt water", "sea came")),
    ("locusts", ("locust", "grasshopper", "hopper", "swarm", "insect")),
    ("harvest_fail", ("harvest", "crop")),
    ("famine", ("famine", "hunger", "dearth", "starv", "want", "bread", "granar", "food")),
    ("plague", ("plague", "pestilence", "sickness", "disease", "epidemic", "cough")),
    ("drought", ("drought", "dry", "no rain", "rain")),
    ("hard_winter", ("winter", "frost", "freez", "froze", "snow", "ice", "cold")),
    ("flood", ("flood", "inundat", "overflow", "high water", "burst")),
    ("storm", ("storm", "tempest", "gale", "wave")),
    ("earthquake", ("earthquake", "quake", "shook", "tremor", "shake")),
    ("fire", ("fire", "burn", "flame", "blaze")),
    ("revolt", ("revolt", "rising", "rebellion", "uprising", "riot", "rebel")),
    ("battle", ("battle", "fight", "war", "army", "attack", "defeat")),
    ("death_ruler", ("death", "died", "burial", "buried", "funeral")),
    ("wedding", ("marriage", "married", "wedding", "wed", "wife")),
    ("fair", ("fair", "market", "merchant", "trade", "pilgrim", "feast")),
    ("building", ("build", "bridge", "construct", "built", "raised", "work began", "hall", "cathedral", "masons")),
    ("embassy", ("envoy", "embassy", "gift", "ambassador", "diplomat")),
    ("migration", ("migrat", "left", "emptied", "abandon", "exodus", "fled", "depart", "leaving", "scatter")),
]
NEAR_SYNONYMS = {"dike_break": {"flood"}, "harvest_fail": {"famine"}, "red_moon": {"eclipse"},
                 "new_star": {"comet"}, "falling_stars": {"comet"}}


def event_kind(label: str) -> str | None:
    label = label.lower()
    for kind, words in EVENT_KEYWORDS:
        if any(w in label for w in words):
            return kind
    return None


def _norm(name: str) -> str:
    return re.sub(r"^(the|river)\s+", "", name.strip().lower())


class World:
    """Name lookups for one world's answer key."""

    def __init__(self, truth: dict) -> None:
        self.truth = truth
        self.places = {p["id"]: p for p in truth["places"]}
        self.place_of: dict[str, str] = {}  # lowercased name -> place id
        for p in truth["places"]:
            for n in p["names"]:
                self.place_of[n.lower()] = p["id"]
        self.minor = {n.lower() for n in truth.get("sites", []) + truth.get("enemies", [])}
        self.alias_groups = [{a.lower() for a in person["aliases"]} for person in truth.get("people", [])]

    def names_in(self, text: str) -> set[str]:
        """Place-like names (gazetteer, minor sites, foreign peoples) written in a record."""
        found = set()
        for n in list(self.place_of) + list(self.minor):
            if re.search(rf"\b{re.escape(n)}\b", text, flags=re.I):
                found.add(n)
        return found

    def stated_links(self, text: str) -> set[tuple[str, str, str]]:
        """The links a record states outright, as (kind, place id, place id/name)."""
        out = set()
        for m in re.finditer(r"([A-Z][a-z]+) \(called ([A-Z][a-z]+) by others\)", text):
            a, b = self.place_of.get(m.group(1).lower()), self.place_of.get(m.group(2).lower())
            if a and a == b:
                out.add(("same_as", a, a))
        for p in self.places.values():
            if p["kind"] != "city":
                continue
            for c in p["names"]:
                for g in self.places[p["region"]]["names"]:
                    if f"{c} in {g}" in text:
                        out.add(("located_in", p["id"], p["region"]))
                for v in self.places[p["river"]]["names"]:
                    if f"{c} on the {v}" in text:
                        out.add(("on_river", p["id"], p["river"]))
        return out

    def link_true(self, kind: str, a: str, b: str) -> tuple[bool, tuple | None]:
        pa, pb = self.place_of.get(_norm(a)), self.place_of.get(_norm(b))
        if kind == "same_as":
            if pa and pa == pb:
                return True, ("same_as", pa, pa)
            if any(_norm(a) in g and _norm(b) in g for g in self.alias_groups):
                return True, None
            return False, None
        if not pa or not pb or self.places[pa]["kind"] != "city":
            return False, None
        want = "region" if kind == "located_in" else "river"
        ok = self.places[pa][want] == pb
        return ok, ((kind, pa, pb) if ok else None)


def grade(truth: dict, notes: list[dict], sources: dict[str, str]) -> dict:
    """notes: rows from learner.notebook.notes() plus their 'links'; sources: record_id -> text."""
    w = World(truth)
    rec = truth["records"]
    t = {k: 0 for k in ("notes", "complete", "event_scored", "event_ok", "places_true", "places_found",
                        "places_listed", "places_real", "links_stated", "links_captured", "links_made",
                        "links_true", "kept_links", "kept_true", "detail_records", "colour_ok",
                        "direction_ok", "duration_ok", "quiz", "quiz_verbatim")}
    wrong_events: list[tuple[str, str, str]] = []
    for n in notes:
        rid, text, info = n["record_id"], sources[n["record_id"]], rec[n["record_id"]]
        t["notes"] += 1
        t["complete"] += int(bool(n.get("complete")))
        if info["kind"] != "person_action":
            t["event_scored"] += 1
            got = event_kind(n["event"] or "")
            ok = got == info["kind"] or got in NEAR_SYNONYMS.get(info["kind"], set())
            t["event_ok"] += int(ok)
            if not ok:
                wrong_events.append((rid, info["kind"], n["event"]))
        names = w.names_in(text)
        listed = {_norm(p) for p in n["places"]}
        t["places_true"] += len(names)
        t["places_found"] += len(names & listed)
        t["places_listed"] += len(listed)
        t["places_real"] += len({p for p in listed if p in names})
        stated = w.stated_links(text)
        captured = set()
        for link in n.get("links", []):
            t["links_made"] += 1
            ok, key = w.link_true(link["kind"], link["a"], link["b"])
            t["links_true"] += int(ok)
            if link.get("stated", 1):  # what the learner keeps as fact after the source check
                t["kept_links"] += 1
                t["kept_true"] += int(ok)
                if key:
                    captured.add(key)
        t["links_stated"] += len(stated)
        t["links_captured"] += len(stated & captured)
        if info["kind"] in ("comet", "locusts"):
            t["detail_records"] += 1
            details = " ".join(n["details"] + n["places"]).lower()
            colours = [c for c in ph.COLOR_PHRASES[info["slots"]["color"]] if c.lower() in text.lower()]
            t["colour_ok"] += int(any(c.lower() in details for c in colours))
            if info["kind"] == "comet":
                d = info["slots"]["dir"]
                phrase = next((p for p in ph.DIRECTION_PHRASES[d] if p in text), "")
                key_words = [x for x in phrase.split() if len(x) > 3 and x not in ("over", "towards", "where", "above", "below")]
                t["direction_ok"] += int(d in details or any(k.lower() in details for k in key_words))
                t["duration_ok"] += int("night" in details)
        if n.get("answer"):
            t["quiz"] += 1
            t["quiz_verbatim"] += int(bool(n.get("answer_verbatim")))

    def rate(a: str, b: str) -> float | None:
        return round(t[a] / t[b], 3) if t[b] else None

    comets = sum(1 for n in notes if rec[n["record_id"]]["kind"] == "comet")
    return {
        "counts": t,
        "format_complete": rate("complete", "notes"),
        "event_accuracy": rate("event_ok", "event_scored"),
        "place_recall": rate("places_found", "places_true"),
        "place_precision": rate("places_real", "places_listed"),
        "link_recall": rate("links_captured", "links_stated"),
        "link_precision_all": rate("links_true", "links_made"),
        "link_precision": rate("kept_true", "kept_links"),  # after the source check: what is used as fact
        "colour_captured": rate("colour_ok", "detail_records"),
        "direction_captured": round(t["direction_ok"] / comets, 3) if comets else None,
        "duration_captured": round(t["duration_ok"] / comets, 3) if comets else None,
        "quiz_answer_verbatim": rate("quiz_verbatim", "quiz"),
        "wrong_events": wrong_events[:40],
    }
