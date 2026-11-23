"""Build fake worlds: chronicles written by several invented chroniclers, with planted patterns.

Each world directory gets three files:
  records.jsonl  - the records the learner may read (the first 80% of the timeline)
  heldout.jsonl  - the last 20% of the timeline, never shown to the learner; used to test predictions
  truth.json     - the answer key: planted patterns, decoys, and the world's gazetteer

Planted pattern types:
  periodic      the same event, with the same telltale details, every P years (+/- a little): the Halley type
  cause_effect  event A is always followed by event B after a fixed lag
  chain         A -> B -> C, each after a fixed lag
  identity      one person written about under different names and titles

Decoys look like patterns but aren't: a regular-looking run that stops, a cause that is usually not
followed by its "effect", and two different people who share an epithet.

  python -m chronicles.generator world  --seed 7 --out worlds/demo [--small]
  python -m chronicles.generator splits --out worlds [--train 20 --val 5 --test 10 --transfer 10]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from chronicles import phrasing as ph
from chronicles.names import NameBank

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
EPITHETS = ["One-Handed", "Grey Hawk", "Red Beard", "Lame", "Silent", "Wolf of the Fens", "Iron-Hand", "Fair"]
CHRONICLE_KINDS = [  # (title template, register)
    ("the Annals of {x} Abbey", "florid"), ("the Chronicle of {x}", "plain"),
    ("the Book of the Merchants of {x}", "terse"), ("the Day-Book of the house of {x}", "terse"),
    ("the Deeds of the Kings, written at {x}", "florid"), ("the Town Book of {x}", "plain"),
]


@dataclass
class WorldConfig:
    span: int = 1000
    n_records: tuple[int, int] = (200, 400)  # target number of visible records
    heldout_fraction: float = 0.2
    types: tuple[str, ...] = ("periodic", "cause_effect", "chain")
    n_periodic: tuple[int, int] = (1, 2)
    n_cause_effect: tuple[int, int] = (1, 2)
    n_chain: tuple[int, int] = (1, 1)
    n_identity: tuple[int, int] = (1, 1)
    period_range: tuple[int, int] = (55, 110)
    periodic_jitter: int = 2  # each periodic occurrence lands up to this many years early or late
    decoys: tuple[str, ...] = ("periodic", "cause_effect")
    n_chroniclers: tuple[int, int] = (5, 8)
    second_recorder_rate: float = 0.15  # an event also written down by a second chronicler
    rosetta_rate: float = 0.12  # a place named by both of its names in one record
    geography_rate: float = 0.12  # a city mentioned with its region or river ("Vesk in Stuweisko")
    flavor_rate: float = 0.3
    drop_rate: float = 0.15  # periodic occurrences nobody wrote down


SMALL = WorldConfig(span=250, n_records=(36, 44), period_range=(28, 36), n_periodic=(1, 1),
                    n_cause_effect=(1, 1), n_chain=(1, 1), n_chroniclers=(3, 4),
                    second_recorder_rate=0.08, drop_rate=0.0, flavor_rate=0.2)

SPLITS = {  # split -> pattern types and decoys. Identity appears only in "transfer".
    "train": dict(types=("periodic", "cause_effect", "chain"), decoys=("periodic", "cause_effect")),
    "val": dict(types=("periodic", "cause_effect", "chain"), decoys=("periodic", "cause_effect")),
    "test": dict(types=("periodic", "cause_effect", "chain"), decoys=("periodic", "cause_effect")),
    "transfer": dict(types=("periodic", "cause_effect", "chain", "identity"),
                     decoys=("periodic", "cause_effect", "identity")),
}


@dataclass
class Place:
    id: str
    kind: str  # city / river / region / coast
    names: list[str]  # [native name, foreign name]
    river: str | None = None
    region: str | None = None
    coast: str | None = None


@dataclass
class Event:
    kind: str
    year: int
    slots: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)  # e.g. "p1", "p2:cause", "d1"
    who: str | None = None  # identity events: the exact alias text used


@dataclass
class Chronicler:
    id: str
    title: str
    register: str
    name_pref: int  # which of each place's two names this chronicler uses
    start: int
    end: int
    home: str


class World:
    def __init__(self, seed: int | str, config: WorldConfig) -> None:
        self.seed = seed
        self.cfg = config
        self.rng = random.Random(f"chronicles:{seed}")
        # Separate stream for location mentions, so adding them didn't change any world's events or wording.
        self.geo_rng = random.Random(f"chronicles-geo:{seed}")
        self.city_uses: dict[str, int] = {}
        self.names = NameBank(self.rng)
        r = self.rng
        self.start = r.randint(300, 900)
        self.end = self.start + config.span
        self.cut = self.start + int(round(config.span * (1 - config.heldout_fraction)))
        self.places: dict[str, Place] = {}
        self.events: list[Event] = []
        self.patterns: list[dict] = []
        self.decoys: list[dict] = []
        self.people: list[dict] = []
        self.forbidden: set[tuple[str, str, str]] = set()  # (kind, slot, value) noise must avoid
        self.used_cities: set[str] = set()
        self._build_geography()
        self._build_rulers()
        self._build_chroniclers()

    # ---- the world ------------------------------------------------------------------------------

    def _place(self, kind: str, **links) -> Place:
        pid = f"{kind}{sum(p.kind == kind for p in self.places.values()) + 1}"
        place = Place(pid, kind, [self.names.new(), self.names.new()], **links)
        self.places[pid] = place
        return place

    def _build_geography(self) -> None:
        r = self.rng
        rivers = [self._place("river") for _ in range(3)]
        regions = [self._place("region") for _ in range(3)]
        coasts = [self._place("coast") for _ in range(2)]
        for i in range(6):
            self._place("city", river=rivers[i % 3].id, region=regions[(i + 1) % 3].id,
                        coast=coasts[i % 2].id if i < 3 else None)
        self.sites = [self.names.new() for _ in range(6)]  # minor places: battlefields, foreign lands
        self.enemies = [self.names.new() for _ in range(3)]
        self.families = [self.names.new() for _ in range(4)]
        r.shuffle(self.sites)

    def of(self, kind: str) -> list[Place]:
        return [p for p in self.places.values() if p.kind == kind]

    def _build_rulers(self) -> None:
        r = self.rng
        self.rulers: list[tuple[int, int, str]] = []
        y = self.start - r.randint(0, 20)
        while y <= self.end:
            length = r.randint(10, 40)
            title = r.choice(["King", "Queen", "Duke", "Prince"])
            self.rulers.append((y, y + length, f"{title} {self.names.new()}"))
            y += length

    def ruler_at(self, year: int) -> str:
        for a, b, name in self.rulers:
            if a <= year < b:
                return name
        return self.rulers[-1][2]

    def _build_chroniclers(self) -> None:
        r, cfg = self.rng, self.cfg
        n = r.randint(*cfg.n_chroniclers)
        cities = self.of("city")
        self.chroniclers: list[Chronicler] = []
        span = cfg.span
        for i in range(n):
            title, register = r.choice(CHRONICLE_KINDS)
            home = r.choice(cities)
            length = r.randint(int(span * 0.3), int(span * 0.6))
            start = self.start + int((span - length) * i / max(1, n - 1)) + r.randint(-10, 10)
            self.chroniclers.append(Chronicler(
                id=f"c{i + 1}", title=title.format(x=home.names[i % 2] if "{x}" in title else ""),
                register=register, name_pref=i % 2, start=max(self.start, start),
                end=min(self.end, start + length), home=home.id))
        for year in range(self.start, self.end + 1, 10):  # always at least two chroniclers writing
            active = self.active(year)
            while len(active) < 2:
                c = r.choice([c for c in self.chroniclers if c not in active])
                c.start, c.end = min(c.start, year), max(c.end, year)
                active = self.active(year)

    def active(self, year: int) -> list[Chronicler]:
        return [c for c in self.chroniclers if c.start <= year <= c.end]

    # ---- helpers --------------------------------------------------------------------------------

    def add(self, kind: str, year: int, tags: list[str] | None = None, who: str | None = None, **slots) -> Event:
        ev = Event(kind, year, slots, list(tags or []), who)
        self.events.append(ev)
        return ev

    def _forbid(self, kind: str, slot: str, value: str) -> None:
        self.forbidden.add((kind, slot, value))

    def _spread_years(self, n: int, lo: int, hi: int, gap: int) -> list[int]:
        """n years in [lo, hi], at least `gap` apart, spread at random (by construction, never fails silently)."""
        slack = (hi - lo) - gap * (n - 1)
        if slack < 0:
            raise ValueError(f"can't fit {n} years {gap} apart in [{lo}, {hi}]")
        offsets = sorted(self.rng.randint(0, slack) for _ in range(n))
        return [lo + off + gap * i for i, off in enumerate(offsets)]

    def _free_city(self, coastal: bool = False) -> Place:
        """A city no other pattern or decoy uses yet, so patterns never share a place."""
        pool = [c for c in self.of("city") if c.id not in self.used_cities and (c.coast or not coastal)]
        city = self.rng.choice(pool or [c for c in self.of("city") if c.coast or not coastal])
        self.used_cities.add(city.id)
        return city

    # ---- planted patterns -----------------------------------------------------------------------

    def plant_periodic(self, pid: str) -> None:
        r, cfg = self.rng, self.cfg
        used = {p.get("signature", {}).get("dir") for p in self.patterns} | {p.get("kind") for p in self.patterns}
        kind = "comet" if "comet" not in used else "locusts"
        period = r.randint(*cfg.period_range)
        first = self.start + r.randint(0, period - 1)
        years = list(range(first, self.end - 2, period))
        jitter = cfg.periodic_jitter
        years = [y + r.randint(-jitter, jitter) for y in years]
        years = [y for y in years if self.start <= y <= self.end]
        # Nobody wrote some occurrences down, but never two in a row: a long silence while the
        # chronicles carry on would make a real pattern look like a decoy run that stopped.
        recorded, prev_dropped = [], False
        for y in years:
            drop = r.random() < cfg.drop_rate
            if drop and not prev_dropped:
                prev_dropped = True
                continue
            recorded.append(y)
            prev_dropped = False
        visible = [y for y in recorded if y < self.cut]
        while len(visible) < 4:  # put back dropped ones until at least 4 are visible
            missing = [y for y in years if y < self.cut and y not in recorded]
            if not missing:
                break
            recorded.append(r.choice(missing))
            visible = [y for y in recorded if y < self.cut]
        if not any(y >= self.cut for y in recorded):
            recorded += [y for y in years if y >= self.cut][:1]
        recorded.sort()
        if kind == "comet":
            sig = {"dir": r.choice(list(ph.DIRECTION_PHRASES)), "color": r.choice(["pale", "golden", "white"]),
                   "nights": r.randint(9, 12)}
            for y in recorded:
                self.add("comet", y, [pid], dir=sig["dir"], color=sig["color"],
                         nights=sig["nights"] + r.randint(-1, 1))
            self._forbid("comet", "dir+color", f"{sig['dir']}+{sig['color']}")
            desc = f"a {sig['color']} tailed star in the {sig['dir']}, seen about {sig['nights']} nights"
        else:
            region = r.choice(self.of("region"))
            sig = {"color": "white", "region": region.id}
            for y in recorded:
                self.add("locusts", y, [pid], region=region.id, color="white")
            self._forbid("locusts", "color", "white")
            desc = f"white locusts over {region.names[0]}"
        self.patterns.append({"id": pid, "type": "periodic", "kind": kind, "params": {"period": period},
                              "jitter": jitter, "signature": sig, "occurrence_years": years, "description":
                              f"{desc}, returning every {period} years"})

    CAUSE_EFFECT_OPTIONS = [  # (cause kind, key slot, effect kind, lag range)
        ("flood", "river", "famine", (1, 3)),
        ("hard_winter", "river", "plague", (1, 1)),
        ("drought", "region", "fire", (1, 2)),
    ]

    def plant_cause_effect(self, pid: str) -> None:
        r = self.rng
        taken = {p.get("kind") for p in self.patterns}
        options = [o for o in self.CAUSE_EFFECT_OPTIONS if f"{o[0]}->{o[2]}" not in taken]
        cause, key, effect, lags = r.choice(options)
        lag = r.randint(*lags)
        city = self._free_city()
        key_value = city.river if key == "river" else city.region
        n = r.randint(3, 5)
        years = self._spread_years(n - 1, self.start + 5, self.cut - 40, max(8, self.cfg.span // 40))
        years.append(self.cut - r.randint(1, lag))  # this effect lands in the held-out years
        if r.random() < 0.5:
            years.append(r.randint(self.cut + 1, self.end - lag - 1))  # and one entirely in the held-out years
        instances = []
        for y in sorted(years):
            slots = {"river": city.river, "region": city.region} if key == "river" else {"region": city.region,
                                                                                        "city": city.id}
            self.add(cause, y, [f"{pid}:cause"], **slots)
            self.add(effect, y + lag, [f"{pid}:effect"], city=city.id, region=city.region)
            instances.append({"cause_year": y, "effect_year": y + lag})
        self._forbid(cause, key, key_value)
        self._forbid(effect, "city", city.id)
        key_name = self.places[key_value].names[0]
        self.patterns.append({"id": pid, "type": "cause_effect", "kind": f"{cause}->{effect}",
                              "params": {"lag": lag}, "instances": instances,
                              "description": f"{cause.replace('_', ' ')} at {key_name} is followed by "
                                             f"{effect} at {city.names[0]} {lag} year(s) later"})

    CHAIN_OPTIONS = [  # (A, B, C, lag ranges)
        ("storm", "dike_break", "migration", ((1, 1), (1, 2))),
        ("drought", "locusts", "famine", ((1, 1), (1, 1))),
    ]

    def plant_chain(self, pid: str) -> None:
        r = self.rng
        a, b, c, (l1r, l2r) = r.choice(self.CHAIN_OPTIONS)
        if any(p.get("kind", "").startswith(a) for p in self.patterns):  # avoid reusing a cause kind
            a, b, c, (l1r, l2r) = next(o for o in self.CHAIN_OPTIONS if o[0] != a)
        l1, l2 = r.randint(*l1r), r.randint(*l2r)
        city = self._free_city(coastal=(a == "storm"))
        years = self._spread_years(r.randint(2, 3), self.start + 5, self.cut - 60, max(10, self.cfg.span // 25))
        years.append(self.cut - l1 - r.randint(1, l2))  # last link lands in the held-out years
        instances = []
        for y in sorted(years):
            if a == "storm":
                self.add("storm", y, [f"{pid}:a"], coast=city.coast)
                self.add("dike_break", y + l1, [f"{pid}:b"], city=city.id)
                self.add("migration", y + l1 + l2, [f"{pid}:c"], city=city.id)
            else:
                self.add("drought", y, [f"{pid}:a"], region=city.region, city=city.id)
                self.add("locusts", y + l1, [f"{pid}:b"], region=city.region, color="brown")
                self.add("famine", y + l1 + l2, [f"{pid}:c"], city=city.id, region=city.region)
            instances.append({"years": [y, y + l1, y + l1 + l2]})
        if a == "storm":
            self._forbid("storm", "coast", city.coast)
            self._forbid("dike_break", "city", city.id)
            self._forbid("migration", "city", city.id)
        else:
            self._forbid("drought", "region", city.region)
            self._forbid("locusts", "region", city.region)
            self._forbid("famine", "city", city.id)
        self.patterns.append({"id": pid, "type": "chain", "kind": f"{a}->{b}->{c}", "params": {"lags": [l1, l2]},
                              "instances": instances, "description":
                              f"{a} -> {b.replace('_', ' ')} at {city.names[0]} after {l1} -> {c} after {l2} more"})

    def plant_identity(self, pid: str) -> None:
        r = self.rng
        name = self.names.new()
        domain = self.names.new()
        epithet = r.choice(EPITHETS)
        aliases = [name, f"the Lord of {domain}", f"the {epithet}"]
        start = r.randint(self.start + 50, self.cut - 60)
        years = sorted(r.sample(range(start, start + 40), r.randint(4, 6)))
        links = [f"{name}, whom men called the {epithet},", f"the {epithet}, Lord of {domain},"]
        whos = links + [r.choice(aliases) for _ in years[2:]]
        r.shuffle(whos)
        for y, who in zip(years, whos):
            self.add("person_action", y, [pid], who=who)
        self.people.append({"id": pid, "name": name, "aliases": aliases})
        self.patterns.append({"id": pid, "type": "identity", "kind": "person", "params": {"aliases": aliases},
                              "description": f"{name}, the Lord of {domain} and 'the {epithet}' are one person"})

    # ---- decoys ---------------------------------------------------------------------------------

    def decoy_periodic(self, did: str) -> None:
        r = self.rng
        city = self._free_city()
        span = self.cfg.span
        gap = r.randint(max(8, span // 40), max(12, span // 22))
        first = r.randint(self.start + span // 50, self.cut - 3 * gap - span // 12)
        years = [first + k * gap for k in range(3)]
        for y in years:
            self.add("red_moon", y, [did], city=city.id)
        self._forbid("red_moon", "city", city.id)
        self.decoys.append({"id": did, "type": "periodic", "params": {"period": gap}, "why":
                            f"a red moon over {city.names[0]} three times, {gap} years apart, then never again"})

    def decoy_cause_effect(self, did: str) -> None:
        r = self.rng
        city = self._free_city()
        years = self._spread_years(6, self.start + 10, self.cut - 10, max(8, self.cfg.span // 33))
        paired = r.sample(years, 2)
        for y in years:
            self.add("fair", y, [did] if y in paired else [], city=city.id)
        for y in paired:
            self.add("fire", y + 1, [did], city=city.id)
        self._forbid("fire", "city", city.id)
        self._forbid("fair", "city", city.id)
        self.decoys.append({"id": did, "type": "cause_effect", "params": {"lag": 1}, "why":
                            f"a fire followed the fair at {city.names[0]} twice, but four other fairs had no fire"})

    def decoy_identity(self, did: str) -> None:
        r = self.rng
        epithet = "Grey"
        later = r.randint(150, 220)
        first = r.randint(self.start + 20, max(self.start + 21, self.cut - later - 20))
        for base in (first, first + later):
            name = self.names.new()
            self.add("person_action", base, [did], who=f"{name} the {epithet}")
            self.add("person_action", base + r.randint(2, 15), [did], who=f"the {epithet}")
        self.decoys.append({"id": did, "type": "identity", "params": {}, "why":
                            "two different men, generations apart, both called 'the Grey'"})

    # ---- noise ----------------------------------------------------------------------------------

    NOISE_WEIGHTS = {"comet": 4, "falling_stars": 4, "new_star": 2, "eclipse": 3, "red_moon": 3, "flood": 8,
                     "drought": 5, "hard_winter": 6, "storm": 6, "earthquake": 2, "locusts": 3, "famine": 6,
                     "plague": 5, "fire": 6, "revolt": 4, "battle": 8, "death_ruler": 5, "fair": 7,
                     "building": 6, "embassy": 5, "migration": 3, "wedding": 4, "dike_break": 2,
                     "harvest_fail": 3}

    def _allowed(self, kind: str, slots: dict) -> bool:
        for k, slot, value in self.forbidden:
            if k != kind:
                continue
            if slot == "dir+color":
                d, c = value.split("+")
                if slots.get("dir") == d or slots.get("color") == c:  # differ in both, to be fair
                    return False
            elif slots.get(slot) == value:
                return False
        return True

    def _noise_slots(self, kind: str) -> dict:
        r = self.rng
        city = r.choice(self.of("city"))
        slots: dict = {"city": city.id, "river": city.river, "region": city.region,
                       "coast": city.coast or r.choice(self.of("coast")).id}
        if kind in ("flood", "hard_winter", "building"):
            slots["river"] = r.choice(self.of("river")).id
        if kind in ("drought", "plague", "locusts", "harvest_fail"):
            slots["region"] = r.choice(self.of("region")).id
        if kind == "comet":
            slots.update(dir=r.choice(list(ph.DIRECTION_PHRASES)), color=r.choice(list(ph.COLOR_PHRASES)),
                         nights=r.randint(2, 6))
        if kind == "locusts":
            slots["color"] = r.choice(["brown", "green"])
        return slots

    def add_noise(self, target_visible: int) -> None:
        r = self.rng
        kinds = list(self.NOISE_WEIGHTS)
        weights = [self.NOISE_WEIGHTS[k] for k in kinds]
        expected_records = 1 + self.cfg.second_recorder_rate
        planted_visible = sum(1 for e in self.events if e.year < self.cut)
        n_visible = max(0, int((target_visible / expected_records) - planted_visible))
        n_total = int(n_visible / (1 - self.cfg.heldout_fraction))
        added = 0
        while added < n_total:
            kind = r.choices(kinds, weights)[0]
            slots = self._noise_slots(kind)
            if not self._allowed(kind, slots):
                continue
            self.add(kind, r.randint(self.start, self.end), **slots)
            added += 1

    # ---- writing it down ------------------------------------------------------------------------

    def _pname(self, pid: str, chron: Chronicler) -> str:
        place = self.places[pid]
        name = place.names[chron.name_pref]
        if self.rng.random() < self.cfg.rosetta_rate:
            name = f"{name} (called {place.names[1 - chron.name_pref]} by others)"
        return name

    def _locate(self, pid: str, chron: Chronicler, name: str) -> str:
        """Lets a reader learn the map: which city lies in which region, on which river.
        A city's first written mention names its region, its second its river, later ones sometimes."""
        place = self.places[pid]
        uses = self.city_uses.get(pid, 0)
        self.city_uses[pid] = uses + 1
        if uses == 0 or (uses > 1 and self.geo_rng.random() < self.cfg.geography_rate / 2):
            return f"{name} in {self.places[place.region].names[chron.name_pref]}"
        if uses == 1 or (uses > 1 and self.geo_rng.random() < self.cfg.geography_rate / 2):
            return f"{name} on the {self.places[place.river].names[chron.name_pref]}"
        return name

    def _number(self, n: int, register: str) -> str:
        return str(n) if register == "terse" or n >= len(NUM_WORDS) else NUM_WORDS[n]

    def render(self, ev: Event, chron: Chronicler) -> str:
        r = self.rng
        register = chron.register if r.random() < 0.85 else r.choice(ph.REGISTERS)
        self._last_city = ev.slots.get("city") or r.choice(self.of("city")).id
        values = {
            "city": self._pname(self._last_city, chron),
            "river": self._pname(ev.slots.get("river") or r.choice(self.of("river")).id, chron),
            "region": self._pname(ev.slots.get("region") or r.choice(self.of("region")).id, chron),
            "coast": self._pname(ev.slots.get("coast") or r.choice(self.of("coast")).id, chron),
            "dir": r.choice(ph.DIRECTION_PHRASES[ev.slots.get("dir") or r.choice(list(ph.DIRECTION_PHRASES))]),
            "color": r.choice(ph.COLOR_PHRASES[ev.slots.get("color") or r.choice(list(ph.COLOR_PHRASES))]),
            "nights": self._number(ev.slots.get("nights", r.randint(2, 6)), register),
            "weeks": self._number(r.randint(2, 9), register),
            "ruler": self.ruler_at(ev.year), "enemy": r.choice(self.enemies), "place": r.choice(self.sites),
            "outcome": r.choice(ph.OUTCOMES), "building": r.choice(ph.BUILDINGS),
            "family": r.choice(self.families), "who": ev.who or "",
        }
        bank = ph.PERSON_ACTIONS if ev.kind == "person_action" else ph.KINDS[ev.kind]
        template = r.choice(bank[register])
        if "{city}" in template:
            city_pid = ev.slots.get("city") or self._last_city
            values["city"] = self._locate(city_pid, chron, values["city"])
        sentence = template.format(**values)
        if register == "terse":
            text = r.choice(ph.DATE_FORMATS["terse"]).format(year=ev.year, S=sentence[0].upper() + sentence[1:])
            if not text.endswith("."):
                text += "."
        else:
            text = r.choice(ph.DATE_FORMATS[register]).format(year=ev.year, s=sentence) + "."
        if r.random() < self.cfg.flavor_rate:
            text += " " + r.choice(ph.FLAVOR).format(place=values["place"], family=values["family"])
        return re.sub(r"\b([Aa]) (?=[aeiou])", r"\1n ", text)  # "a ember-bright" -> "an ember-bright"

    def write_records(self) -> list[dict]:
        r = self.rng
        rows = []
        for ev in self.events:
            active = self.active(ev.year) or self.chroniclers
            writers = [r.choice(active)]
            if len(active) > 1 and r.random() < self.cfg.second_recorder_rate:
                writers.append(r.choice([c for c in active if c is not writers[0]]))
            for chron in writers:
                rows.append({"year": ev.year, "chronicler": chron.title, "place": self.places[chron.home].names[
                    chron.name_pref], "text": self.render(ev, chron), "_event": ev})
        rows.sort(key=lambda row: (row["year"], row["chronicler"], row["text"]))
        for i, row in enumerate(rows, 1):
            row["id"] = f"r{i:04d}"
        return rows


def build_world(seed: int | str, config: WorldConfig | None = None) -> tuple[list[dict], list[dict], dict]:
    """Returns (visible records, held-out records, truth)."""
    cfg = config or WorldConfig()
    w = World(seed, cfg)
    r = w.rng
    counter = {"p": 0, "d": 0}

    def next_id(prefix: str) -> str:
        counter[prefix] += 1
        return f"{prefix}{counter[prefix]}"

    plan = []
    for t in cfg.types:
        lo, hi = getattr(cfg, f"n_{t}")
        plan += [t] * r.randint(lo, hi)
    for t in plan:
        getattr(w, f"plant_{t}")(next_id("p"))
    for t in cfg.decoys:
        getattr(w, f"decoy_{t}")(next_id("d"))
    w.add_noise(r.randint(*cfg.n_records))

    rows = w.write_records()
    by_tag: dict[str, list[dict]] = {}
    for row in rows:
        for tag in row["_event"].tags:
            by_tag.setdefault(tag, []).append(row)

    def ids(rows_: list[dict], heldout: bool) -> list[str]:
        return [x["id"] for x in rows_ if (x["year"] >= w.cut) == heldout]

    for p in w.patterns:
        tagged = [row for tag, rs in by_tag.items() if tag.split(":")[0] == p["id"] for row in rs]
        tagged.sort(key=lambda x: x["id"])
        p["record_ids"], p["heldout_record_ids"] = ids(tagged, False), ids(tagged, True)
        if p["type"] == "periodic":
            p["heldout_years"] = sorted({x["year"] for x in tagged if x["year"] >= w.cut})
        elif p["type"] == "cause_effect":
            p["heldout_years"] = sorted({x["year"] for x in by_tag.get(f"{p['id']}:effect", [])
                                         if x["year"] >= w.cut})
        elif p["type"] == "chain":
            p["heldout_years"] = sorted({x["year"] for x in by_tag.get(f"{p['id']}:c", []) if x["year"] >= w.cut})
        else:
            p["heldout_years"] = []
    for d in w.decoys:
        tagged = sorted(by_tag.get(d["id"], []), key=lambda x: x["id"])
        d["record_ids"], d["heldout_record_ids"] = ids(tagged, False), ids(tagged, True)

    # Per-record answer key (what each record is about), used to score note-taking in Phase 2.
    record_truth = {row["id"]: {"kind": row["_event"].kind, "slots": row["_event"].slots,
                                "who": row["_event"].who, "tags": row["_event"].tags} for row in rows}
    public = [{k: row[k] for k in ("id", "year", "chronicler", "place", "text")} for row in rows]
    visible = [row for row in public if row["year"] < w.cut]
    heldout = [row for row in public if row["year"] >= w.cut]
    truth = {
        "seed": seed, "config": asdict(cfg), "span": [w.start, w.end], "heldout_from": w.cut,
        "patterns": w.patterns, "decoys": w.decoys,
        "places": [asdict(p) for p in w.places.values()],
        "chroniclers": [asdict(c) for c in w.chroniclers], "people": w.people,
        "sites": w.sites, "enemies": w.enemies, "families": w.families,
        "rulers": [{"from": a, "to": b, "name": n} for a, b, n in w.rulers],
        "counts": {"visible": len(visible), "heldout": len(heldout)},
        "records": record_truth,
    }
    return visible, heldout, truth


def write_world(out_dir: str | Path, seed: int | str, config: WorldConfig | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    visible, heldout, truth = build_world(seed, config)
    for name, rows in (("records.jsonl", visible), ("heldout.jsonl", heldout)):
        with (out / name).open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    return truth


def write_splits(out_dir: str | Path, counts: dict[str, int], base_seed: int = 0) -> dict[str, list[str]]:
    made: dict[str, list[str]] = {}
    for split, n in counts.items():
        cfg = replace(WorldConfig(), **SPLITS[split])
        made[split] = []
        for i in range(n):
            name = f"{split}-{i:03d}"
            write_world(Path(out_dir) / split / name, f"{base_seed}:{split}:{i}", cfg)
            made[split].append(name)
    return made


def write_sample(path: str | Path, seed: int | str) -> dict:
    """A small world as Markdown, for a person to read: records first, answer key at the bottom."""
    visible, heldout, truth = build_world(seed, SMALL)
    lines = [
        "# A small fake chronicle",
        "",
        f"{len(visible)} records from {len(truth['chroniclers'])} chroniclers, years {truth['span'][0]}"
        f"–{truth['heldout_from'] - 1}. Hidden in them are {len(truth['patterns'])} planted patterns and "
        f"{len(truth['decoys'])} decoys (things that look like patterns but aren't).",
        "",
        "Try to find them before scrolling to the answer key. Watch out: places have two names, and each "
        "chronicler uses their own.",
        "",
        "| id | year | chronicler | text |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['id']} | {r['year']} | {r['chronicler']} | {r['text']} |" for r in visible]
    lines += ["", "---", "", "## Answer key", "", "### Planted patterns", ""]
    for p in truth["patterns"]:
        lines.append(f"- **{p['type']}**: {p['description']}. Records: {', '.join(p['record_ids'])}. "
                     f"Next, in the hidden years: {p.get('heldout_years') or 'n/a'}.")
    lines += ["", "### Decoys", ""]
    for d in truth["decoys"]:
        lines.append(f"- **looks {d['type']}**: {d['why']}. Records: {', '.join(d['record_ids'])}.")
    lines += ["", "### Places and their two names", ""]
    lines += [f"- {p['names'][0]} = {p['names'][1]} ({p['kind']})" for p in truth["places"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chronicles.generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("world", help="write one world")
    one.add_argument("--seed", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--small", action="store_true", help="a ~40-record world a person can read")
    one.add_argument("--split", choices=sorted(SPLITS), help="use that split's pattern types")
    sample = sub.add_parser("sample", help="write a small world as Markdown with the answer key")
    sample.add_argument("--seed", required=True)
    sample.add_argument("--out", required=True)
    many = sub.add_parser("splits", help="write train/val/test/transfer worlds")
    many.add_argument("--out", required=True)
    many.add_argument("--base-seed", type=int, default=0)
    for split, default in (("train", 20), ("val", 5), ("test", 10), ("transfer", 10)):
        many.add_argument(f"--{split}", type=int, default=default)
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        truth = write_sample(args.out, args.seed)
        print(f"wrote {args.out}: {truth['counts']['visible']} records")
        return 0
    if args.cmd == "world":
        cfg = SMALL if args.small else WorldConfig()
        if args.split:
            cfg = replace(cfg, **SPLITS[args.split])
        truth = write_world(args.out, args.seed, cfg)
        print(f"wrote {args.out}: {truth['counts']['visible']} visible + {truth['counts']['heldout']} held-out "
              f"records, {len(truth['patterns'])} patterns, {len(truth['decoys'])} decoys")
    else:
        made = write_splits(args.out, {s: getattr(args, s) for s in SPLITS}, args.base_seed)
        print({s: len(v) for s, v in made.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
