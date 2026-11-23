"""Invented names built from syllables, unique within a world."""

from __future__ import annotations

import random

ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "r", "s", "st",
          "t", "th", "v", "w", "z", "sk", "", ""]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "y", "ia", "a", "e", "o"]
CODAS = ["", "", "n", "r", "l", "s", "th", "nd", "rk", "m", "x", "sk", "v", "ld", "rn"]

# Invented names must never collide with ordinary words used in the phrasing banks.
BLOCKED = {"star", "stars", "tail", "flood", "fire", "west", "east", "north", "south", "storm", "hunger",
           "bread", "grain", "salt", "moon", "sun", "river", "year", "lord", "king", "queen", "town",
           "hill", "hills", "sea", "road", "bridge", "hall", "fair", "plague", "frost", "snow", "dark",
           "red", "pale", "white", "gold", "grey", "grave", "dead", "ash", "bane", "bark", "mark"}


class NameBank:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def _syllable(self, last: bool) -> str:
        r = self.rng
        coda = r.choice(CODAS) if (last or r.random() < 0.25) else ""
        return r.choice(ONSETS) + r.choice(VOWELS) + coda

    def new(self, min_syllables: int = 2, max_syllables: int = 3) -> str:
        for _ in range(1000):
            n = self.rng.randint(min_syllables, max_syllables)
            word = "".join(self._syllable(i == n - 1) for i in range(n))
            if not 4 <= len(word) <= 10 or word in BLOCKED or word in self.used:
                continue
            if any(word[i] == word[i + 1] == word[i + 2] for i in range(len(word) - 2)):
                continue
            self.used.add(word)
            return word.capitalize()
        raise RuntimeError("ran out of unique names")
