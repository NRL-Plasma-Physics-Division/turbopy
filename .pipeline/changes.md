# Changes: Document 2D solve additions and prepare v2026.07.23 release

STATUS: READY

## Files touched
- `docs/compute_tools.rst` — amended the `FiniteDifference2D` bullet to enumerate all four cylindrical operators (`ddr`, `ddz`, `del2_r`, `del2_z`) and describe `laplacian()` dispatch; inserted a new `PoissonSolver2D` bullet with Dirichlet BCs, grid-type dispatch, `TypeError` on 1D, and `solve(source)` shape contract.
- `docs/diagnostics.rst` — replaced the stale `HistoryDiagnostic` bullet (which falsely claimed "1D grids only — raises `NotImplementedError`") with an accurate description of 1D + 2D support and `x`/`y` vs `r`/`z` coordinate attachment.
- `turbopy/diagnostics.py` — docstring-only edits on `HistoryDiagnostic`: (a) removed the `Raises / NotImplementedError` block from the class-level docstring and rewrote the opening summary to reflect 2D support; (b) rewrote the `initialize()` docstring to describe actual 2D coordinate attachment behaviour instead of a stale `NotImplementedError` claim.  No runtime code was changed.
- `turbopy/__version__.py` — bumped `VERSION` from `('2023', '06', '09')` to `('2026', '07', '23')`.
- `README.md` — updated the "Cite a specific version" block: prose version string, inline citation text, BibTeX key (`turbopy_v2026.07.23`), `title`, `version`, `month`, and `year` fields now reference `v2026.07.23` / 2026.  The Zenodo DOI URL (`https://doi.org/10.5281/zenodo.4088189`) was left untouched per spec resolution #4.

## Phases completed
- [x] Phase 1 — Docs corrections (narrative chapters + `HistoryDiagnostic` docstring)
- [x] Phase 2 — Version bump (`__version__.py` + README citation block)
- [ ] Phase 3 — Release handoff (NOT executed by pipeline; checklist below for maintainer)

## Phase 3 — Maintainer checklist (execute manually after merge)

1. `git tag -a v2026.07.23 -m "Release v2026.07.23: 2D solve (Grid2DCartesian, Grid2DCylindrical, PoissonSolver2D, FiniteDifference2D, 2D-aware diagnostics)"`
2. `git push origin v2026.07.23`
3. Confirm readthedocs picks up the new tag build (check the Versions dropdown on turbopy.readthedocs.io).
4. Confirm Zenodo auto-mints a new DOI via the GitHub webhook; update the version-specific DOI in `README.md` with the new Zenodo link once it is available.
5. Create a GitHub Release from the tag; paste 2D-solve release notes in the body (summary: new `Grid2DCartesian` and `Grid2DCylindrical` grids, `PoissonSolver2D`, full `FiniteDifference2D` operator set, 2D-aware `GridDiagnostic` and `HistoryDiagnostic`).

## Deviations from spec
None. Every file touched was named in the spec.

## Notes for reviewer
- The `HistoryDiagnostic.initialize()` method already contains the 2D branches at runtime (lines 808-821 in the original); the docstring was simply stale. No code logic was touched.
- `docs/conf.py` line 29 reads from `turbopy.__version__` at build time. The version `2026.07.23` now appears in the Sphinx sidebar/footer without any change to `conf.py` (confirmed by clean docs build).
- The concept-DOI paragraph in `README.md` (`turbopy_project` bibtex entry, `year = 2023`) was left unchanged per the spec — it refers to the project as a whole, not a specific release.
- No latent runtime bugs were discovered during this session.

## Smoke check

```
$ python -c "import turbopy; print(turbopy.__version__)"
2026.07.23

$ python -c "from turbopy import __version__; print(__version__)"
2026.07.23

$ python -m pytest tests/ -x -q
133 passed, 5 warnings in 0.67s

$ cd docs && make html
build succeeded.   (0 warnings)
```
