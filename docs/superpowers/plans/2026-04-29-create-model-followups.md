# Create Model — v1.7.1 follow-up notes

> **STATUS: ✅ ALL 6 FINDINGS RESOLVED 2026-04-29** — findings 1, 2, 4, 6
> shipped in **v1.7.1** (PR #1, commits `50bdfd0`–`832243d`); findings 3,
> 5 shipped in **v1.7.2** (commits `cd6e09f`, `fd10a79`). All deployed
> to https://laguna.ku.lt/HexSimPy/ same day.

Findings from the Task 19.2 manual Playwright smoke (2026-04-29) of plan
`2026-04-28-create-model-feature.md`. None of these block the v1.7.0
ship; each is a small follow-up worth tracking.

Smoke test passed end-to-end (4/4 steps green): geometry preview renders,
EMODnet bathymetry fetch works (4512 B cache hit), upload clears on
landscape switch, production layers fully restore.

---

## 1. Layer-id collision: `"water"` is overloaded across layer types

**Resolved in:** v1.7.1 (commit `50bdfd0`). `_build_preview_h3_layer` now
emits `id="upload-water"`. New test
`tests/test_sidebar.py::test_upload_preview_layer_id_is_namespaced`
locks the namespace requirement. Production-confirmed: post-fix smoke
showed zero new `shaderInputs` errors during landscape switch (was 2 pre-fix).

**Severity:** medium — produces console errors and could cause render
glitches on landscape switch.

**Where:**
- `app.py:929` — upload preview: `h3_hexagon_layer("water", ...)`
- `app.py:2160`, `2253`, `2738`, `2960` — production H3 landscapes:
  `h3_hexagon_layer("water", ...)`
- `app.py:2193`, `2209`, `2287` — TriMesh / hex-as-scatter fallback:
  `ScatterplotLayer({id: "water"})`

**Symptom in smoke log:**
```
[ERROR] deck: drawing ScatterplotLayer({id: 'water'}) to screen:
  Cannot read properties of undefined (reading 'shaderInputs')
```
Fires twice during landscape switch (curonian_h3_multires → columbia →
back). deck.gl 9.2.10 cannot morph a layer's class under a stable id.

**Why:** all three render paths reuse `"water"` as the canonical
landscape-cell layer id — a holdover from when there was only one
landscape backend. With the upload preview now adding a fourth caller
(also `h3_hexagon_layer`), the id collision is benign within a single
landscape (deck.gl hot-swaps within the same class) but breaks when
switching to a landscape that uses `ScatterplotLayer`.

**How to apply:** namespace the upload-preview layer (e.g.
`"upload-water"` or `"preview-water"`) in `_build_preview_h3_layer` —
this also lets `_push_create_model_preview` clear *just* the upload
layers without touching the production water layer.

**Effort:** 5-min change in `app.py:929` + 1 test asserting the layer id
is not `"water"`.

---

## 2. Stale legend metadata while upload preview is active

**Resolved in:** v1.7.1 (commit `4b17721`). Approach (b) from below
implemented: `map_legend()` early-returns a minimal `Upload preview
(N cells)` badge while `_uploaded_preview` is not None. Test
`tests/test_sidebar.py::test_map_legend_swaps_to_upload_badge_during_preview`
locks the wiring.

**Severity:** low — cosmetic; in-sidebar status is authoritative.

**Where:** the top-right map legend box continues to read
`HEXAGONAL GRID (185,428 CELLS)` after the upload preview replaces the
production layers (only 36 cells render in the smoke test).

**Why:** the legend widget pulls its cell count from a separate reactive
that tracks `sim.mesh`, not from `_uploaded_preview`. The plan's
"short-circuiting" architecture intentionally bypasses the sim-mesh
flow, so the legend is never told the displayed cell count changed.

**How to apply:** either
- (a) `_push_create_model_preview` updates the legend widget alongside
  the layer push, or
- (b) hide the cell-count legend entry while `_uploaded_preview is not
  None` and show a simpler "Upload preview (N cells)" badge instead.

(b) is probably cleaner — the existing legend is heavily coupled to the
production landscape's reach colors and depth scale, none of which apply
to a viewer-only upload.

**Effort:** ~30 min including a sidebar wiring test for the legend
text.

---

## 3. Bbox-fit doesn't re-trigger on bathymetry toggle or second Preview click

**Resolved in:** v1.7.2 (commit `fd10a79`). New `_upload_view_state(mesh)`
helper mirrors the H3 branch of `_view_state(sim)`;
`_push_create_model_preview` now passes `view_state=` on every layer
push. Visual smoke confirms camera snaps back from a panned-out view to
the upload bbox on bathymetry toggle. Surprise UX win — turns out the
"fight the user" concern was theoretical: making sure they actually see
the result of their action is more valuable than preserving whatever
zoom they had.

**Severity:** low — debatable whether to "fix" or call it intentional.

**Symptom:** after the user has panned/zoomed away from the upload
preview, toggling Show bathymetry or clicking Preview again preserves
the user's view rather than re-fitting.

**Why:** `_push_create_model_preview` (`app.py:1297`) re-emits the layer
list but doesn't push a fresh `initialViewState`. The first preview
fits to bbox via the existing init logic; subsequent updates ride the
user's current view.

**Tradeoff:** fitting on every update would fight users who
deliberately zoomed in for inspection; fitting only on the first
preview means the bathymetry toggle silently feels broken if the user
is zoomed in and not looking at the patch.

**How to apply:** if we choose to fit on toggle, add a
`fit_to_bbox=True` parameter to `_push_create_model_preview` and have
`_on_create_model_with_bathy_toggle` set it. Don't re-fit on the
"second click of Preview with same file" path — that's the user
explicitly asking to re-render, not relocate.

**Effort:** ~20 min if we decide to do (a). Could also be left as a
documented quirk in v1.7.0.

---

## 4. Import statements scattered through `salmon_ibm/h3_tessellate.py`

**Resolved in:** v1.7.1 (commit `1dff6c9`). All imports hoisted to the
top block in PEP-8 order. Behavior unchanged; all 20 `test_h3_tessellate.py`
tests still pass.

**Severity:** very low — cosmetic only, no behavior impact.

**Where:**
- `salmon_ibm/h3_tessellate.py:1-14` — top-of-file imports (correct)
- `salmon_ibm/h3_tessellate.py:164-171` — `io`, `zipfile`, `dataclass`,
  `Path`, `geopandas`, `shapely`, `unary_union` declared mid-file
  *after* `polygon_trust_water_mask` is defined
- `salmon_ibm/h3_tessellate.py:345` — `import hashlib`

**Why:** the module was extended task-by-task per
`2026-04-28-create-model-feature.md` and each task added imports next
to the function it added rather than hoisting them. Functionally
identical; lints poorly and slightly hurts cold-import time.

**How to apply:** hoist all `import` statements to the top block;
verify with `python -c "import salmon_ibm.h3_tessellate"` and run
`pytest tests/test_h3_tessellate.py -v` before committing. Single
commit, no behavior change.

**Effort:** 5 min.

---

## 5. Console warning "handler must be a function that takes one argument"

**Resolved in:** v1.7.2 (commit `cd6e09f`). The "predates Create Model"
guess was wrong — root cause was `Shiny.addCustomMessageHandler(
'map_loader_hide', function() { ... })` at app.py:1128 (zero-arg handler;
Shiny.js requires exactly one parameter). Fix: changed `function()` to
`function(msg)` with `msg` unused. Console error count dropped 2 → 1 on
fresh page load. (Bonus v1.7.2-tail favicon fix later took it to 0.)

**Severity:** very low — fires once at page load, no observed effect.

**Where:** browser console, ~668 ms after initial page load. No stack
trace or source file in the warning. Predates the Create Model feature
(reproduced on a tag-clean checkout — would need verification, but
nothing about Create Model registers a custom message handler).

**Likely cause:** a Shiny.js custom message handler or output binding
registered with the wrong signature somewhere. The warning is
non-blocking — the offending registration is rejected, the rest of the
page works.

**How to apply:** grep `Shiny.addCustomMessageHandler`,
`Shiny.outputBindings`, `Shiny.inputBindings` across `*.js` and
template files; identify the registration whose handler doesn't take
exactly one argument; fix the signature.

**Effort:** ~15 min including verification.

---

## 6. Deploy script prints HTTP URL but laguna serves HTTPS

**Resolved in:** v1.7.1 (commit `832243d`). One-line change to the
`URL=` var in `scripts/deploy_laguna.sh`. Self-validating: the very
next deploy printed `URL: https://laguna.ku.lt/HexSimPy/` correctly.

**Severity:** very low — cosmetic; users still reach the site fine due
to HTTP→HTTPS auto-redirect.

**Where:** `scripts/deploy_laguna.sh` final stdout block prints
`URL: http://laguna.ku.lt/HexSimPy/` after a successful deploy. The
production server has been HTTPS-only (with redirect) for some time,
so the printed URL is misleading — copy-pasting it works, but it
implies http is the canonical endpoint when it isn't.

**Why:** added before laguna gained TLS; never updated.

**How to apply:** one-line change to the printf/echo statement.
Confirmed during the 2026-04-29 production smoke test —
http://laguna.ku.lt/HexSimPy/ 301-redirects to
https://laguna.ku.lt/HexSimPy/.

**Effort:** 2 min.

---

## Suggested grouping

For a v1.7.1 patch release: items 1, 2, 4, 6 together (small, low-risk,
all toolchain-improvement) — defer 3 (UX judgment call, may not need a
fix) and 5 (predates this feature, deserves its own session and may not
even be related to Create Model).

---

## Actual outcome (recorded 2026-04-29)

The suggested grouping mostly held. Findings 1, 2, 4, 6 shipped together
in v1.7.1 (PR #1, manual same-day implementation rather than the
2026-05-13 routine — the routine is now disabled). The "deferred"
findings 3 and 5 turned out to be cheap to fix in the same afternoon,
shipped in v1.7.2 a few hours later. Total elapsed time from this doc's
authorship to all-resolved: roughly one working day.

Two of the calls in this doc turned out to be wrong:
- Finding 3 was framed as a UX judgment call. Implementation showed the
  bbox-fit-on-every-push behaviour is unambiguously better — you pretty
  much always want the user to see the result of their action. No
  "fights the user" downside materialised in smoke testing.
- Finding 5 was framed as "predates Create Model" and "may not even be
  related". Wrong — root cause was `function()` instead of `function(msg)`
  in our own inline `app.py` JS. Fix was one parameter declaration.

Lesson: the "needs separate debug session" framing for low-severity
items can over-defer trivial fixes. Worth a quick grep before assuming
a console warning is from a third-party library.
