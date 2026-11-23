"""Mechanical source checks: does the record really say this, word for word?

These are the cheap, certain half of "check your notes against the book". They never use the model.
  quote_ok(text, quote)          an evidence quote appears verbatim (case and spacing aside)
  link_stated(text, kind, a, b)  a link is written in the record itself, not inferred:
      same_as     "A (called B by others)", "A, whom men called B", "B, Lord of ...", "A, also known as B"
      located_in  "A in B"        (an "(called X by others)" between them is allowed)
      on_river    "A on the B"
"""

from __future__ import annotations

import re

_ALIAS = r"(?:\s*\(called [^)]+ by others\))?"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _bare(name: str) -> str:
    return re.sub(r"^(the|river)\s+", "", name.strip(), flags=re.I)


def quote_ok(text: str, quote: str) -> bool:
    q = _norm(quote).strip(" .,;:\"'")
    return bool(q) and q in _norm(text)


def link_stated(text: str, kind: str, a: str, b: str) -> bool:
    t = _norm(text)
    a, b = _norm(_bare(a)), _norm(_bare(b))
    if not a or not b or a == b:
        return False
    if kind == "located_in":
        return re.search(rf"\b{re.escape(a)}\b{_ALIAS} in (?:the )?{re.escape(b)}\b", t) is not None
    if kind == "on_river":
        return re.search(rf"\b{re.escape(a)}\b{_ALIAS} on the {re.escape(b)}\b", t) is not None
    if kind == "same_as":
        for x, y in ((a, b), (b, a)):
            x_, y_ = re.escape(x), re.escape(y)
            if re.search(rf"\b{x_}\s*\(called (?:the )?{y_} by others\)", t):
                return True
            if re.search(rf"\b{x_},? (?:whom men called|also called|also known as|known as|called) (?:the )?{y_}\b", t):
                return True
            if y.startswith("lord of") and re.search(rf"\b{x_}, (?:the )?{y_}\b", t):  # "the Lame, Lord of X"
                return True
        return False
    return False
