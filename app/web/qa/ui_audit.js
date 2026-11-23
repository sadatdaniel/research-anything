// UI quality check: walks every page, sub-view and dialog of the running app and reports what can be measured.
// It's a development tool: nothing in the app loads it. Run it in the page at each width and theme:
//
//   const qa = await import("/qa/ui_audit.js?" + Date.now());
//   await qa.sweep({ theme: "adwaita" })      // -> { width, theme, views, issues: [{ view, check, detail }] }
//
// Checks:
//   overflow     the page scrolls sideways, or something sticks out past the right edge
//   entry        a text box that isn't a GNOME entry (filled, no border)
//   contrast     text below 4.5:1 against what's behind it (3:1 for large text)
//   target       a control smaller than 28 px tall (30 px on narrow screens)
//   label        an icon-only button without an accessible name
//   grid         cards stacked in one column although there's room for two, or one card stretched far
//                beyond the others in its row
//   duplicate    the same navigation shown twice at once
//   clipped      text cut off without an ellipsis
//   tiny         text smaller than 11 px

const wait = ms => new Promise(r => setTimeout(r, ms));
const visible = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== "hidden" && cs.display !== "none" && +cs.opacity > 0.05; };
const name = el => { const id = el.id ? `#${el.id}` : ""; const cls = [...el.classList].slice(0, 2).map(c => `.${c}`).join(""); const txt = (el.innerText || el.value || el.getAttribute("aria-label") || "").trim().slice(0, 40); return `${el.tagName.toLowerCase()}${id}${cls}${txt ? ` "${txt}"` : ""}`; };

function rgba(str) {
  const m = str.match(/rgba?\(([^)]+)\)/); if (!m) return null;
  const parts = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
  return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 };
}
const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const over = (top, under) => ({ r: top.r * top.a + under.r * (1 - top.a), g: top.g * top.a + under.g * (1 - top.a), b: top.b * top.a + under.b * (1 - top.a), a: 1 });
function backdrop(el) {   // the colour behind an element: its own and its ancestors' backgrounds, blended
  const layers = [];
  for (let e = el; e; e = e.parentElement) {
    if (getComputedStyle(e).backgroundImage !== "none") return null;   // a gradient or image: can't be measured this way
    const c = rgba(getComputedStyle(e).backgroundColor);
    if (c && c.a > 0) { layers.push(c); if (c.a >= 1) break; }
  }
  let col = { r: 255, g: 255, b: 255, a: 1 };
  if (!layers.length || layers[layers.length - 1].a < 1) col = rgba(getComputedStyle(document.body).backgroundColor) || col;
  for (const c of layers.reverse()) col = over(c, col);
  return col;
}

export function audit(view) {
  const out = [], add = (check, detail) => out.push({ view, check, detail });
  const W = innerWidth, narrow = W < 700;
  if (document.documentElement.scrollWidth > W + 1) add("overflow", `page is ${document.documentElement.scrollWidth}px wide in a ${W}px window`);
  const scope = document.querySelector("dialog[open]") || document.body;
  const all = [...scope.querySelectorAll("*")].filter(visible);
  let sticking = 0;
  for (const el of all) {
    const r = el.getBoundingClientRect();
    if (r.right > W + 1 && sticking < 4) {
      let clipped = false;
      for (let e = el.parentElement; e; e = e.parentElement) { const ox = getComputedStyle(e).overflowX; if (ox !== "visible") { clipped = true; break; } }
      if (!clipped) { add("overflow", `${name(el)} ends at ${Math.round(r.right)}px`); sticking++; }
    }
  }
  for (const el of scope.querySelectorAll("input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file]):not(.toggle), select, textarea")) {
    if (!visible(el)) continue;
    const cs = getComputedStyle(el);
    if (parseFloat(cs.borderTopWidth) > 0 || parseFloat(cs.borderBottomWidth) > 0) add("entry", `${name(el)} has a ${cs.borderTopWidth} border`);
    if (rgba(cs.backgroundColor)?.a === 0) add("entry", `${name(el)} has no fill`);
  }
  const seen = new Set();
  for (const el of all) {
    const direct = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (!direct || seen.has(el)) continue;
    seen.add(el);
    if (el.closest("svg, [disabled], .preview, option")) continue;
    const cs = getComputedStyle(el), fg = rgba(cs.color); if (!fg) continue;
    const bg = backdrop(el), size = parseFloat(cs.fontSize), bold = +cs.fontWeight >= 700;
    if (!bg) continue;
    const need = size >= 24 || (bold && size >= 18.66) ? 3 : 4.5;
    const c = ratio(over(fg, bg), bg);
    if (c < need - 0.05) add("contrast", `${name(el)}: ${c.toFixed(1)}:1 (needs ${need})`);
    if (size < 11) add("tiny", `${name(el)}: ${size}px`);
  }
  for (const el of scope.querySelectorAll("button, a.btn, select, input:not([type=checkbox]):not([type=range]):not([type=file]), [role=tab]")) {
    if (!visible(el) || el.closest(".styles")) continue;
    const r = el.getBoundingClientRect(), min = narrow ? 30 : 28;
    if (r.height < min) add("target", `${name(el)} is ${Math.round(r.height)}px tall`);
    if (el.tagName === "BUTTON" && !el.innerText.trim() && !el.getAttribute("aria-label") && !el.title) add("label", `${name(el)} has no name`);
  }
  for (const g of scope.querySelectorAll("*")) {
    if (!visible(g) || getComputedStyle(g).display !== "grid") continue;
    const kids = [...g.children].filter(visible);
    const cards = kids.filter(k => k.matches(".box, .scard, .card"));
    if (cards.length < 2) continue;
    const cols = new Set(cards.map(k => Math.round(k.getBoundingClientRect().left))).size;
    const minW = Math.min(...cards.map(k => k.getBoundingClientRect().width));
    if (cols === 1 && g.getBoundingClientRect().width >= 2 * minW + 12 && cards.length >= 2) add("grid", `${name(g)}: ${cards.length} cards in one column though ${Math.round(g.getBoundingClientRect().width)}px fits two`);
    for (const k of cards) {   // stretched: much taller than what's in it (content-driven heights are fine)
      const r = k.getBoundingClientRect(), inner = [...k.children].filter(visible).map(c => c.getBoundingClientRect());
      if (!inner.length) continue;
      const empty = r.height - (Math.max(...inner.map(c => c.bottom)) - Math.min(...inner.map(c => c.top)));
      if (empty > 160 && empty > 0.4 * r.height) add("grid", `${name(k)} is ${Math.round(r.height)}px tall but ${Math.round(empty)}px of it is empty`);
    }
    for (const k of kids) if (getComputedStyle(k).display === "grid" && k.matches(".cards")) add("grid", `${name(g)} holds another card grid as one of its cells`);
  }
  const navs = {};
  for (const el of document.querySelectorAll("[data-go], [data-tab]")) if (visible(el) && !el.closest("dialog") && el.closest(".side, #tabs, .bottom")) { const k = el.dataset.go || el.dataset.tab; (navs[k] ||= []).push(el); }
  for (const [k, els] of Object.entries(navs)) if (els.length > 1) add("duplicate", `"${k}" is shown ${els.length} times: ${els.map(e => e.closest(".side, #tabs, .bottom")?.className.split(" ")[0] || "?").join(", ")}`);
  for (const el of all) {
    if (!el.matches("button, h1, h2, h3, b, label, .chip, .tag, a")) continue;
    const cs = getComputedStyle(el);
    if (el.scrollWidth > el.clientWidth + 2 && cs.overflow !== "visible" && cs.textOverflow !== "ellipsis") add("clipped", `${name(el)}`);
  }
  return out;
}

// Every variable the active theme sets in themes.css must be what the page actually uses: a later rule with
// the same specificity can silently replace it (this hid the whole Adwaita look until Phase 3's check).
export async function themeApplied() {
  const t = document.documentElement.dataset.theme, css = await (await fetch("/themes.css")).text();
  const at = css.indexOf(`[data-theme="${t}"] {`); if (at < 0) return [{ view: "theme", check: "theme", detail: `no block for ${t}` }];
  const block = css.slice(at, css.indexOf("}", at)), out = [], cs = getComputedStyle(document.documentElement);
  for (const [, k, v] of block.matchAll(/(--[\w-]+):\s*([^;]+);/g)) {
    const norm = x => x.replace(/\s+/g, " ").replace(/\s*,\s*/g, ", ").trim().toLowerCase();
    if (norm(cs.getPropertyValue(k)) !== norm(v)) out.push({ view: `theme ${t}`, check: "theme", detail: `${k} is "${cs.getPropertyValue(k).trim()}", the theme says "${v.trim()}"` });
  }
  return out;
}

export async function sweep({ theme } = {}) {
  const prev = document.documentElement.dataset.theme;
  if (theme) document.documentElement.dataset.theme = theme;
  const views = [], issues = [...await themeApplied()];
  const run = async (label, fn) => { await fn(); await wait(350); views.push(label); issues.push(...audit(label)); };
  const memory = S.skills.find(k => k.stages?.confirmed) || S.skills[0];
  const fusion = S.skills.find(k => (k.combined_from || []).length);
  await run("library: cards", async () => { go("library"); S.libView = "cards"; renderLibrary(); });
  await run("library: list", async () => { S.libView = "list"; renderLibrary(); });
  S.libView = "cards"; renderLibrary();
  await run("library: two picked", async () => { S.picked = new Set(S.skills.slice(0, 2).map(k => k.id)); renderLibrary(); });
  await run("dialog: combine", async () => openCombine());
  $("#combine").close(); S.picked.clear(); renderLibrary();
  await run("dialog: import", async () => $("#import").showModal()); $("#import").close();
  await run("dialog: new memory", async () => $("#new-skill").showModal()); $("#new-skill").close();
  await run("dialog: remove", async () => askRemove(memory.id)); $("#remove").close("cancel");
  for (const [id, label] of [[memory?.id, "memory"], [fusion?.id, "fusion"]]) {
    if (!id) continue;
    openSkill(id); await wait(1500);
    await run(`${label}: overview`, async () => go("overview"));
    for (const v of ["timeline", "graph", "list"]) await run(`${label}: explore ${v}`, async () => { S.view = v; go("explore"); });
    await run(`${label}: questions`, async () => { S.qmode = "bank"; go("questions"); });
    await run(`${label}: ask your own`, async () => { S.qmode = "ask"; go("questions"); });
    await run(`${label}: compare`, async () => go("compare"));
    await run(`${label}: journal`, async () => go("journal"));
  }
  for (const [page] of PREF_PAGES) await run(`preferences: ${page}`, async () => { if (!$("#prefs").open) await openPrefs(); S.ppage = page; $("#prefs").classList.add("at-page"); renderSettings(); });
  $("#prefs").close();
  await run("dialog: about", async () => showAbout(false)); $("#about").close();
  await run("dialog: shortcuts", async () => showAbout(true)); $("#about").close();
  go("library");
  if (theme) document.documentElement.dataset.theme = prev;
  return { width: innerWidth, theme: theme || prev, views: views.length, issues };
}
