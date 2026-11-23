from learner.sourcecheck import link_stated, quote_ok

FLOOD = "In the year 910, the Thafupa (called Sandkark by others) burst its banks and drowned the fields of Steiskty."
FAIR = "Year 986. Fair at Ildkysk (called Henlebyn by others) on the Foslaeth."
PERSON = "Year 701. Garrick, whom men called the Red Fox, gave land at Brennock in Velsham."
BATTLE = "In the year 990, Duke Grexskeild fought the men of Kribynern at Riaorn and was driven back."


def test_quotes_must_be_verbatim():
    assert quote_ok(FLOOD, "burst its banks") and quote_ok(FLOOD, "  The Thafupa (called Sandkark ")
    assert not quote_ok(FLOOD, "broke its banks") and not quote_ok(FLOOD, "")


def test_stated_links_pass():
    assert link_stated(FLOOD, "same_as", "Thafupa", "Sandkark")
    assert link_stated(FLOOD, "same_as", "Sandkark", "the Thafupa")
    assert link_stated(FAIR, "on_river", "Ildkysk", "Foslaeth")  # with the alias in between
    assert link_stated(PERSON, "same_as", "Garrick", "the Red Fox")
    assert link_stated(PERSON, "located_in", "Brennock", "Velsham")


def test_inferred_links_fail():
    assert not link_stated(BATTLE, "located_in", "Riaorn", "Kribynern")  # "fought the men of X at Y"
    assert not link_stated(FLOOD, "on_river", "Steiskty", "Thafupa")  # flooded fields, never "on the"
    assert not link_stated(FAIR, "same_as", "Ildkysk", "Foslaeth")
    assert not link_stated(FLOOD, "same_as", "Thafupa", "Thafupa")
