# Spec: Document 2D solve additions and prepare a tagged release

STATUS: READY

## Resolutions to Open Questions (user confirmed 2026-07-23)
1. **Version:** CalVer, new tag `v2026.07.23`; `VERSION = ('2026', '07', '23')` in `turbopy/__version__.py`.
2. **CHANGELOG:** do not create one; release notes go in PR body / GitHub Release only.
3. **"readthedocs" scope:** Sphinx sources under `docs/` only; do not touch `.readthedocs.yml`.
4. **Zenodo DOI in README:** leave the existing `v2023.06.09` DOI in the citation example as a static illustrative example; the maintainer will update it post-release if desired.

## Overview
The 2D solve capability (`Grid2DCartesian`, `Grid2DCylindrical`, `PoissonSolver2D`, `FiniteDifference2D`, 2D-aware `GridDiagnostic`/`HistoryDiagnostic`) has already landed on `main` (PR #192, commit 8703c17). The Sphinx sources under `docs/` already cover the grids exhaustively, but the compute-tools chapter is missing `PoissonSolver2D` entirely and only mentions half of the 2D FD operators, and the diagnostics chapter contains a stale claim that `HistoryDiagnostic` is 1D-only when the code actually supports both 2D grid types. This spec covers the doc gap-fill plus a version bump; the git tag itself is gated on explicit user confirmation.

## Files to create or modify
- `docs/compute_tools.rst` — add `PoissonSolver2D` bullet under "Built-in compute tools"; expand the `FiniteDifference2D` bullet to enumerate the full cylindrical operator set (`ddr`, `ddz`, `del2_r`, `del2_z`) and note `laplacian()` dispatch behaviour on each 2D grid type; note field-ordering convention (C order, `reshape((N1, N2))`).
- `docs/diagnostics.rst` — replace the stale "**1D grids only** — raises `NotImplementedError` …" sentence on `HistoryDiagnostic` with an accurate description: supports 1D grids and both 2D grid types; on 2D grids the xarray dataset carries `x`/`y` (Cartesian) or `r`/`z` (cylindrical) coordinates in addition to `time`.
- `turbopy/diagnostics.py` — fix the corresponding stale class docstring on `HistoryDiagnostic` (lines ~670-682 and ~793-799) so autodoc-generated API pages agree with the narrative docs. Remove the "Raises NotImplementedError" claim; describe 2D coordinate attachment. Scope-limited to docstring text; no code changes.
- `turbopy/__version__.py` — bump `VERSION` tuple to the new release (see Open Questions #1 for the value).
- `README.md` — update the citation example block (lines 62-76) so the year, filename, DOI-adjacent version strings, and BibTeX key/`title`/`version` fields reference the new tag. Only the example version strings; do NOT rewrite the surrounding narrative.
- No CHANGELOG file exists in the repo (verified — no `CHANGELOG*`, `HISTORY*`, or `RELEASE*` files). Do not create one unless the user asks (see Open Questions #2).

## Reference patterns
- Doc structure for a new compute tool bullet: copy the format of the existing `PoissonSolver1DRadial` and `FiniteDifference2D` entries in `docs/compute_tools.rst` lines 62-83 (short paragraph, `:class:` cross-refs, mention of raises/dispatch behavior).
- Doc structure for a diagnostic behaviour clarification: copy the existing `GridDiagnostic` entry in `docs/diagnostics.rst` lines 57-61 which already handles the 1D-vs-2D distinction cleanly.
- Version bump precedent: the file is a 3-tuple of strings in `turbopy/__version__.py`. The `pyproject.toml` `version = {attr = "turbopy.__version__.__version__"}` and `docs/conf.py` (line 29) both read from this single source, so only this file drives package/doc metadata. README is a manual citation update.
- Docs are already wired via `docs/api.rst` `automodule ... :members:`, so new classes (`PoissonSolver2D`, `FiniteDifference2D`, `Grid2DCartesian`, `Grid2DCylindrical`) are already appearing on the API page automatically. This spec only touches the *narrative* chapters, not the auto-generated API reference.

## Phases

### Phase 1 — Docs corrections (narrative chapters)
Independently mergeable; produces an accurate readthedocs build immediately.

1. `docs/compute_tools.rst`:
   - Insert a new bullet for `turbopy.computetools.PoissonSolver2D` under "Built-in compute tools" (between `FiniteDifference2D` and `BorisPush`). Cover: Dirichlet `φ = 0` on all four boundaries, dispatches on both `Grid2DCartesian` and `Grid2DCylindrical`, uses `FiniteDifference2D.laplacian()` internally, raises `TypeError` on a 1D grid, `.solve(source)` expects `source.shape == grid.shape`.
   - Amend the existing `FiniteDifference2D` bullet to add the missing cylindrical operator `del2_r` and clarify that `laplacian()` returns `del2_x + del2_y` on Cartesian grids and `del2_r + del2_z` (with the `(1/r) d/dr` term folded in) on cylindrical grids.
2. `docs/diagnostics.rst`:
   - Rewrite the `HistoryDiagnostic` bullet (lines 66-72) to state that it supports 1D grids and both 2D grid types; on 2D grids the resulting NetCDF has `x`/`y` (Cartesian) or `r`/`z` (cylindrical) coordinates in addition to `time`. Remove the `NotImplementedError` sentence.
3. `turbopy/diagnostics.py`:
   - Delete the "Raises NotImplementedError" block from the `HistoryDiagnostic` class docstring (~lines 674-682).
   - Update the `initialize()` docstring (~lines 786-800) to describe the actual 2D coordinate attachment behaviour instead of claiming only 1D is supported and raising.
   - No functional code changes in this file.

Verify: `cd docs && make html` completes without warnings; render the two chapters and confirm the new content is present.

### Phase 2 — Version bump
Independently mergeable; committed as its own commit so the tag can point at it.

1. Edit `turbopy/__version__.py`: set `VERSION` to the new value from Open Question #1.
2. Edit `README.md` citation block (lines 62-76): update the year, month, `v<version>` strings, and BibTeX key (`turbopy_v<version>`), `title` field, and `version` field to match. Do NOT change the Zenodo concept-DOI URL or the author list.
3. Verify `python -c "import turbopy; print(turbopy.__version__)"` prints the new version.
4. Verify `cd docs && make html` and open `docs/_build/html/index.html` — the footer/sidebar should now show the new version (fed via `docs/conf.py` line 29).

### Phase 3 — Release handoff (NOT executed by pipeline)
Produce a checklist in the PR description for the maintainer to run manually after merge. Do NOT run these commands from the pipeline.

Suggested manual steps for the maintainer:
- `git tag -a v<version> -m "Release v<version>: 2D solve"`
- `git push origin v<version>`
- Confirm readthedocs picks up the new tag build.
- Confirm Zenodo mints a new DOI (auto-triggered by GitHub release).
- Create a GitHub Release from the tag with release notes summarising the 2D additions.

## Acceptance criteria
- [ ] `docs/compute_tools.rst` contains a bullet whose autodoc-linked class is `turbopy.computetools.PoissonSolver2D`, and mentions "Dirichlet", "Grid2DCartesian", "Grid2DCylindrical", and "TypeError".
- [ ] `docs/compute_tools.rst` `FiniteDifference2D` bullet lists all four cylindrical operators: `ddr`, `ddz`, `del2_r`, `del2_z` (currently missing `del2_r`).
- [ ] `docs/diagnostics.rst` `HistoryDiagnostic` bullet no longer contains the substring `NotImplementedError`, and explicitly mentions that 2D grids attach `x`/`y` or `r`/`z` coordinates.
- [ ] `HistoryDiagnostic` class docstring in `turbopy/diagnostics.py` no longer contains a `Raises NotImplementedError` section for 2D grids.
- [ ] `cd docs && make html` exits 0 and produces no new Sphinx warnings compared to the current build (baseline: run `make html` on `main` first, diff the warning list).
- [ ] `python -c "from turbopy import __version__; print(__version__)"` prints the value chosen in Open Question #1.
- [ ] `pyproject.toml`-driven metadata reflects the new version: `python -c "from importlib.metadata import version; print(version('turbopy'))"` (after `pip install -e .`) prints the new version.
- [ ] `README.md` citation block year/version strings match the new release; the Zenodo concept-DOI URL is unchanged.
- [ ] `pytest` still passes (no test file is being modified; this is a smoke check that the docstring edits did not corrupt syntax).
- [ ] No git tag is created and no push is performed by the pipeline. Phase 3 checklist appears in the PR body for the maintainer.

## turboPy-specific checks
- Registry: N/A — no new registered classes; all target classes already `.register()` themselves.
- Shared resources published: N/A — no module changes.
- Shared resources consumed: N/A.
- Grid dispatch: N/A for code, but the doc edits must correctly describe existing dispatch: `PoissonSolver2D` accepts both 2D grid types and raises `TypeError` on 1D; `FiniteDifference2D` accepts both 2D grid types and raises `TypeError` on 1D; `HistoryDiagnostic` accepts 1D and both 2D grid types (no raises).
- Numerical tolerance: N/A — no numeric operator changes.

## Open questions

1. **Version scheme and value.** The existing scheme is CalVer, not semver. All prior git tags are `v2020.01.23`, `v2020.02.20`, ..., `v2020.10.14`, and `turbopy/__version__.py` currently holds `('2023', '06', '09')`. The `/ship` request said "semver: minor bump" but that conflicts with the established convention. Recommendation: **keep CalVer** and use today's date, `('2026', '07', '23')` → `v2026.07.23`, matching the existing pattern. If the user genuinely wants to switch to semver, we need to pick a starting number (e.g., `1.0.0` treating the 2D solve as the 1.0 milestone, or `0.2.0` from an implicit `0.1.0`) and update the tuple layout in `__version__.py` and the README citation template accordingly. **Please confirm the version string before Phase 2 begins.**

2. **Changelog convention.** No `CHANGELOG.md`, `HISTORY.md`, or `RELEASE_NOTES.md` exists in the repo. Prior releases apparently relied on git tag messages and Zenodo release notes for user-facing change history. Recommendation: **do not introduce a CHANGELOG file** in this release (out of scope; would create a maintenance burden nobody asked for). Instead, write the release notes in the PR body and in the eventual GitHub Release. Confirm this is acceptable, or specify a location and format if you want one added.

3. **"readthedocs" scope.** The `.readthedocs.yml` file is minimal (Sphinx configuration + PDF/EPUB formats + conda env) and does not need touching. Recommendation: interpret "update readthedocs" as "update the Sphinx sources under `docs/` that readthedocs.org builds," not "reconfigure the hosted service." Confirm.

4. **Zenodo DOI in README.** Each new release gets its own Zenodo DOI, but that DOI isn't known until *after* GitHub publishes the release (Zenodo mints it via the webhook). The current README example uses the concept DOI `10.5281/zenodo.4088189` for `v2023.06.09` inline. Recommendation: **leave the version-specific DOI in the citation example as-is for now** (it correctly cites `v2023.06.09` as an example) and let the maintainer update it in a follow-up commit once Zenodo has minted the new one. Alternative: replace the specific example with the concept DOI so it never goes stale. Please pick one.

## Notes for the coder
- Do NOT execute `git tag` or `git push` from the pipeline. Phase 3 is a handoff checklist for the human maintainer only.
- Do NOT create a `CHANGELOG.md` unless Open Question #2 explicitly resolves in favor of doing so.
- Do NOT touch `docs/grids.rst`, `docs/input_files.rst`, or `docs/simulation_lifecycle.rst` — they already reference the 2D grid classes correctly.
- Do NOT touch `docs/api.rst` — the `automodule ... :members:` directives already pick up `PoissonSolver2D`, `FiniteDifference2D`, `Grid2DCartesian`, `Grid2DCylindrical` automatically. Confirm by inspecting the current built HTML under `docs/_build/html/api.html` if in doubt.
- The `HistoryDiagnostic` docstring edit is purely textual — leave the code alone even though it might look like it "should" have a 2D check somewhere. The 2D branches at lines 808-821 are the intended behaviour.
- `docs/conf.py` reads the version from `turbopy.__version__` at build time (line 29). Do NOT hardcode the version in `docs/conf.py`; the single source of truth is `turbopy/__version__.py`.
- When bumping the version, commit the `__version__.py` and `README.md` edits together in one commit so the tag (when eventually created) points at a self-consistent tree.
- Run `cd docs && make html` twice — once on `main` to get a baseline warning count, once after the edits to confirm no regression. Sphinx warnings on stale cross-refs are the most likely failure mode.
- If autodoc complains that `PoissonSolver2D` or `FiniteDifference2D` cross-refs in `compute_tools.rst` are unresolved, check that `:class:\`turbopy.computetools.PoissonSolver2D\`` matches the exact module path (it does — verified in `turbopy/computetools.py` line 720).
