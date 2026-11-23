# Phase 3 quality check

_2026-11-23. Done at your request before Phase 4: "after some phase … let's do a quality check before we continue". From now on every phase ends with one (PLAN.md, Phases)._

## How it was checked

1. **An automated UI audit** ([app/web/qa/ui_audit.js](../../app/web/qa/ui_audit.js)), run in the app itself. It visits 31 views:
   - library cards and list, with two memories ticked;
   - every dialog: combine, import, new memory, remove, about, shortcuts;
   - a memory and a fusion, in each of Overview, Explore (timeline, graph, list), Questions (its questions, ask your own), Compare and Journal;
   - all six Preferences pages.

   It ran at **5 widths** (375, 768, 1046, 1440 and 1920 px) in **light and dark**: 310 view checks per run. Each view is checked for:

   | Check | What fails it |
   |---|---|
   | theme | a variable the active theme sets in `themes.css` isn't what the page uses |
   | overflow | sideways scrolling, or anything past the right edge |
   | entry | a text box with a border or without a fill (not a GNOME entry) |
   | contrast | text under 4.5:1 against what's behind it (3:1 for large text) |
   | target | a control under 28 px tall (30 px on narrow screens) |
   | label | an icon-only button with no accessible name |
   | grid | cards stacked in one column when two fit; a card that's mostly empty stretched space; a card grid inside a card grid |
   | duplicate | the same navigation visible twice at once |
   | clipped | text cut off without an ellipsis |
   | tiny | text under 11 px |

2. **Screenshots by eye** of the library, a memory, Ask your own and the dialogs at phone, pane and full-screen widths. The audit can't judge layout or taste.
3. **The full test suite** (`chronicles`, `engine`, `learner`, `app`), plus a new regression test for the theme bug below.

## What it found, and what was fixed

| Problem | Cause | Fix |
|---|---|---|
| **The Adwaita look never fully applied:** grey cards with borders, a `#fafafb` header bar, solid white text boxes and buttons | the page's fallback variables (`:root { --card: …; --fill: … }`) came after `themes.css` with the same specificity, so they replaced the theme's values | fallbacks moved into `:where(:root)`, which has zero specificity. The audit's **theme** check and `test_theme_values_are_not_overridden_by_page_defaults` now guard it |
| Library cards not side by side on big screens; the fusion card 1,341 px tall | the library container was itself a card grid holding the real grids, so each grid became one column and the fusion grid stretched to the row | the container is a plain block; cards are at least 340 px; the library can use up to 1600 px |
| Ask your own: browser-default bordered text boxes | the entries had no `type`, and the entry style matched only `input[type=text]` | the style covers `input:not([type])`; GNOME entry look (fill, no border, 6 px corners, inset focus ring) |
| Ask your own: odd placement (a sentence-shaped form at the top, a second box at the bottom, a whole column of tickboxes) | grown piece by piece | one composer at the bottom (Connection or Search toggle, two entries, a blue Ask button) that stays on screen on phones; "Answer from" as toggle pills; suggested questions |
| 379 text-contrast failures in light mode | the theme bug (grey cards), white text on GNOME's blue (3.8:1), secondary text on tinted fills (4.2–4.4:1), pale memory icons in dark mode | fixed with the theme bug; filled buttons use GNOME's deeper `#1c71d8` (4.8:1); `--ink-2` darkened to `#5e5e64`; dark icons get dark letters |
| Toggle buttons 28 px tall on phones | desktop size kept on touch screens | 34 px on narrow screens |
| "New memory" and "Import" shown twice in the library (header buttons and ghost cards) | the ghost cards predate the header buttons | ghost cards removed; a GNOME status page appears only when the library is empty |

**After the fixes, the audit reports 0 issues** at all 5 widths, in light and dark, across all 31 views.

## Tests

The full suite before this check: 77 passed. The result after this check's changes is added below when the run finishes.

## What this check doesn't cover yet

- **Charts and graphs are drawn in SVG**, and the contrast check skips SVG text: timeline axis labels and graph labels need their own check.
- **Keyboard and screen reader:** focus order and whether every action works without a mouse aren't audited (only the presence of accessible names).
- **Taste:** spacing rhythm, alignment and visual weight still need eyes. Three were judged by eye this time: the library at 1280 and 1920, and Ask your own at phone width.
- **Other themes** (instrument, swiss, brutalist, terminal, blueprint, mono, synthwave) weren't swept; only Adwaita light and dark.

## Running it again

With the app open (`uv run python -m app.server --library … --worlds worlds`), in the browser console at each width:

```js
const qa = await import("/qa/ui_audit.js?" + Date.now());
(await qa.sweep({ theme: "adwaita" })).issues        // and { theme: "adwaita-dark" }
```
