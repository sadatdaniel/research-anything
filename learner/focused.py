"""Focused reading (day mode): turn each record into structured notes.

Ouro-1.4B is a base model, so it's prompted with a few worked examples (all names in them are
invented and appear in no world). For each record it writes:

  Event:    what happened, in a few words
  Places:   places named in the record, separated by ';'
  People:   people or titles named
  Details:  telltale details (colour, direction, duration, ...)
  Links:    what the record says about the map or about names:
              "A = B" (two names for the same thing), "A in B" (lies in a region), "A on the B" (on a river)
  Summary:  one plain sentence
  Q: / A:   a self-quiz question about the record, and its answer in the record's own words

The answer is later checked word for word against the record (the source check), and the quiz
feeds the memory audit after sleep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FEW_SHOT = """Read each chronicle record and take short notes in the same format.
- Event: what happened, in one to three words.
- Places: only real place names written in the record.
- Details: telltale details such as colour, direction (also when said poetically, like "towards the setting sun" = west), how long it lasted.
- Links: only what the record itself says about names or the map: two names for one place ("A = B"), a place in a region ("A in B"), a place on a river ("A on the B"), or two names for one person. Otherwise none.
- A: copy the answer word for word from the record.

Record (year 930): In the year of grace 930, want sat at every table in Dunmere (called Kessaly by others), and mothers boiled grass for their children.
Event: famine
Places: Dunmere; Kessaly
People: none
Details: none
Links: Dunmere = Kessaly
Summary: There was famine in Dunmere, also called Kessaly.
Q: Where was there famine in 930?
A: Dunmere
###
Record (year 744): In 744, fire burned the lower town of Tarrow in Velsham. The bells were recast.
Event: fire
Places: Tarrow; Velsham
People: none
Details: lower town burned
Links: Tarrow in Velsham
Summary: A fire burned the lower town of Tarrow, which lies in Velsham.
Q: What burned in Tarrow in 744?
A: the lower town
###
Record (year 655): Be it remembered that in the year 655 a pale fiery sword hung in the sky over the evening hills for eleven nights.
Event: comet
Places: none
People: none
Details: pale; west (over the evening hills); eleven nights
Links: none
Summary: A pale comet hung in the west for eleven nights.
Q: What hung in the sky in 655?
A: a pale fiery sword
###
Record (year 977): Year 977. Hunger at Calloway on the Wend.
Event: famine
Places: Calloway; Wend
People: none
Details: none
Links: Calloway on the Wend
Summary: There was hunger in Calloway, which lies on the river Wend.
Q: Where was there hunger in 977?
A: Calloway
###
Record (year 701): Year 701. Garrick, whom men called the Red Fox, gave land at Brennock on the Sallow.
Event: gift of land
Places: Brennock; Sallow
People: Garrick; the Red Fox
Details: none
Links: Garrick = the Red Fox; Brennock on the Sallow
Summary: Garrick, known as the Red Fox, gave land at Brennock on the river Sallow.
Q: Who gave land at Brennock in 701?
A: Garrick
###
"""

FIELDS = ("event", "places", "people", "details", "links", "summary", "q", "a")


def prompt_for(record: dict) -> str:
    return FEW_SHOT + f"Record (year {record['year']}): {record['text']}\nEvent:"


@dataclass
class Note:
    record_id: str
    year: int
    event: str = ""
    places: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    same_as: list[tuple[str, str]] = field(default_factory=list)
    located_in: list[tuple[str, str]] = field(default_factory=list)
    on_river: list[tuple[str, str]] = field(default_factory=list)
    summary: str = ""
    question: str = ""
    answer: str = ""
    raw: str = ""
    complete: bool = False  # every field was present and parsed

    @property
    def answer_in_source(self) -> bool | None:
        return None if not self.answer else None  # filled by check_answer()


def _items(value: str) -> list[str]:
    value = value.strip().rstrip(".")
    if not value or value.lower() in ("none", "-", "n/a"):
        return []
    return [v.strip() for v in value.split(";") if v.strip() and v.strip().lower() != "none"]


def _strip_the(name: str) -> str:
    return re.sub(r"^(the|river)\s+", "", name.strip(), flags=re.I).strip()


def parse_links(items: list[str]) -> tuple[list, list, list]:
    same, inside, river = [], [], []
    for item in items:
        if "=" in item:
            a, b = (x.strip() for x in item.split("=", 1))
            if a and b and a != b:
                same.append((a, b))
        elif re.search(r"\s+on the\s+", item):
            a, b = re.split(r"\s+on the\s+", item, maxsplit=1)
            river.append((a.strip(), _strip_the(b)))
        elif re.search(r"\s+in\s+", item):
            a, b = re.split(r"\s+in\s+", item, maxsplit=1)
            inside.append((a.strip(), b.strip()))
    return same, inside, river


def parse_note(record: dict, completion: str) -> Note:
    """completion is what the model wrote after the prompt's trailing 'Event:'."""
    text = "Event:" + completion.split("###")[0]
    found: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(Event|Places|People|Details|Links|Summary|Q|A)\s*:\s*(.*)$", line)
        if m:
            key = m.group(1).lower()
            if key not in found:  # first occurrence wins; the model sometimes starts a new record
                found[key] = m.group(2).strip()
    same, inside, river = parse_links(_items(found.get("links", "")))
    return Note(
        record_id=record["id"], year=int(record["year"]),
        event=found.get("event", "").strip().rstrip(".").lower(),
        places=[_strip_the(p) for p in _items(found.get("places", ""))],
        people=_items(found.get("people", "")),
        details=_items(found.get("details", "")),
        same_as=same, located_in=inside, on_river=river,
        summary=found.get("summary", ""), question=found.get("q", ""), answer=found.get("a", ""),
        raw=completion, complete=all(k in found for k in FIELDS),
    )


def check_answer(note: Note, record_text: str) -> bool:
    """Mechanical source check: the quiz answer must appear word for word in the record (case-insensitive)."""
    answer = note.answer.strip().strip(".").strip()
    return bool(answer) and answer.lower() in record_text.lower()


def read_batch(lm, records: list[dict], max_new_tokens: int = 110) -> list[Note]:
    """Take notes on a batch of records with one batched generate call.

    A looped model keeps a key/value cache for every layer on every loop (about 0.8 MB per token per
    prompt at 4 loops on Ouro-1.4B), so if a batch doesn't fit, it's split in half and retried
    instead of failing."""
    import torch

    prompts = [prompt_for(r) for r in records]
    outs = None
    try:
        outs = lm.generate(prompts, max_new_tokens=max_new_tokens, stop_strings=["###"], tokenizer=lm.tokenizer)
    except torch.OutOfMemoryError:
        if len(records) == 1:
            raise
    # Retry outside the except block: inside it, the traceback keeps the failed attempt's tensors alive.
    if outs is None:
        torch.cuda.empty_cache()
        half = len(records) // 2
        return read_batch(lm, records[:half], max_new_tokens) + read_batch(lm, records[half:], max_new_tokens)
    return [parse_note(r, o) for r, o in zip(records, outs)]
