# Review: Document 2D solve additions and prepare v2026.07.23 release

STATUS: READY

## Summary
This is a documentation + version-bump-only release for the 2D solve feature that already landed on `main` (PR #192). The tester's suite is green (133/133), Sphinx `make html` produces zero warnings against a zero-warning baseline, and the actual `git diff` matches `changes.md` exactly — five source files (`README.md`, `docs/compute_tools.rst`, `docs/diagnostics.rst`, `turbopy/__version__.py`, `turbopy/diagnostics.py`), all named in the spec. I confirmed the `HistoryDiagnostic` edits are purely docstring (all 34 changed lines fall inside `"""..."""` blocks; runtime lines 795-832 including the 2D coordinate branches are byte-identical to `main`). I confirmed the version-bump surface — the only `2023.06.09` / `('2023', ...)` reference anywhere under the repo (excluding `.pipeline/` and `.archive/`) is the Zenodo concept-DOI URL and the `turbopy_project` BibTeX `year = 2023`, both of which the spec's open-question #4 explicitly resolved to leave alone. `pyproject.toml` uses `version = {attr = "turbopy.__version__.__version__"}` so the single-source-of-truth wiring holds. AC7 (stale `importlib.metadata` value) is a genuine editable-install artifact — see the analysis below — and does not block ship. No `v2026*` tag exists locally, no push occurred, no CHANGELOG file was introduced.

## Acceptance criteria
- [x] AC1 — PASS: `docs/compute_tools.rst` lines 89-101 add the `:class:\`turbopy.computetools.PoissonSolver2D\`` bullet and mention `Dirichlet`, `Grid2DCartesian`, `Grid2DCylindrical`, `TypeError`.
- [x] AC2 — PASS: The amended `FiniteDifference2D` bullet enumerates `ddr`, `ddz`, `del2_r`, `del2_z` for the cylindrical case (previously omitted `del2_r`).
- [x] AC3 — PASS: `docs/diagnostics.rst` `HistoryDiagnostic` bullet no longer contains `NotImplementedError` and explicitly names `x`/`y` (Cartesian) and `r`/`z` (cylindrical) coordinates.
- [x] AC4 — PASS: `turbopy/diagnostics.py` class-level and `initialize()` docstrings no longer contain a `Raises NotImplementedError` block; the new `initialize()` docstring describes the actual 2D coordinate attachment.
- [x] AC5 — PASS: `make html` clean (0 warnings) per tester; docs build succeeded.
- [x] AC6 — PASS: `turbopy.__version__` resolves to `2026.07.23` from source (verified by tester).
- [~] AC7 — PARTIAL: `importlib.metadata.version('turbopy')` returns `2023.6.9` in the tester's active env because the editable `dist-info` on disk was cut when the version was `2023.6.9`; the wiring in `pyproject.toml` is correct and `pip install -e .` regenerates a dist-info of `2026.07.23`. See discussion in Findings — not blocking.
- [x] AC8 — PASS: `README.md` citation prose, BibTeX key, `title`, `version`, `month`, `year` all updated to `v2026.07.23` / 2026; Zenodo concept-DOI URL and `turbopy_project` entry (which cites the project, not a release) intentionally unchanged per Open Question #4.
- [x] AC9 — PASS: 133 passed, 0 failed, no new warnings.
- [x] AC10 — PASS: `git tag -l 'v2026*'` returns empty, `git status` matches spec scope, no push in `git log`.

## Findings

### CRITICAL
None.

### HIGH
None.

### MEDIUM
- **AC7 environment gap should be surfaced in the PR body, not just the pipeline artifacts** — `.pipeline/changes.md:17-23` (Phase 3 checklist)
  Issue: The maintainer checklist tells the human to `git tag` and `git push` but does not tell them to `pip install -e .` in any downstream env that reads `importlib.metadata.version('turbopy')` (e.g., a conda env doing plugin discovery, or a Zenodo tooling script that pulls metadata). The tester's own env demonstrated the pitfall.
  Fix: Add "Run `pip install -e .` (or `pip install .`) in any active env before validating the release" as an item in the Phase 3 checklist so the version-bump is not silently invisible to `importlib.metadata` consumers.
- **README citation prose says "For example, a citation for version v2026.07.23"** — `README.md:62`
  Issue: The narrative example is now tagged to a version that does not yet have a Zenodo DOI (the DOI URL still points at `zenodo.4088189`, which is the v2023.06.09 record). A reader who copies the BibTeX will get a `year = 2026, doi = ...v2023-record` mismatch until the maintainer refreshes it post-Zenodo-mint.
  Fix: Not a blocker — Open Question #4 explicitly punted this to the maintainer for the post-release commit. Flag it in the PR body so the maintainer remembers to update the DOI after Zenodo mints it.

### NIT
- The `initialize()` docstring says "the dataset gains an ``r`` coordinate along the ``\"grid\"`` dimension" for the 1D case; the code at line 817 attaches under dim name `'grid'`, so accurate — but consider whether the 1D-cartesian case (`grid.r` on a Cartesian 1D `Grid` — which does exist) would confuse a reader who expects `x`. Purely cosmetic.
- `.archive/sphinx-docs-version-dropdown-sidebar-2026-07-23/spec.md` is untracked in the working tree; consider whether it belongs in `.gitignore` alongside other pipeline artifacts.

## Latent bugs
The coder's `changes.md` explicitly states "No latent runtime bugs were discovered during this session" — consistent with the diff, which shows zero runtime-line changes in `turbopy/diagnostics.py`. I confirmed independently:
- All 34 `+/-` lines in `turbopy/diagnostics.py` fall inside triple-quoted docstring blocks (class-level ~L667-680 and `initialize()` ~L783-794).
- The 2D coord-attachment runtime at L802-819 (`isinstance(grid, Grid2DCartesian)` / `Grid2DCylindrical` / else) is byte-identical to `main`; the new docstring accurately reflects it.
- No `importlib.metadata` / `pkg_resources` consumer exists anywhere in the codebase (`grep` across all `.py`), so AC7's stale-dist-info reading is a pure developer-env artifact with no runtime blast radius inside turbopy itself. Any *external* consumer (a plugin, a Zenodo scraper) would need `pip install -e .` — that is a packaging property, not a defect.

## Recommendation
MERGE WITH FOLLOW-UPS

The change is exactly what the spec asks for: five files touched, all named in the spec, zero scope creep, docstring-only edits to `turbopy/diagnostics.py`, no new tests warranted, no runtime lines changed, tests green, docs build clean, version wiring single-sourced. The AC7 failure is a genuine editable-install caching artifact — `pyproject.toml` reads `attr = "turbopy.__version__.__version__"` correctly, no code in-repo reads `importlib.metadata`, and `pip install -e .` regenerates a `2026.07.23` dist-info — so it does not block ship. The two MEDIUM findings are documentation follow-ups for the maintainer (add `pip install -e .` to the Phase 3 checklist; refresh the Zenodo DOI in `README.md` once the release webhook fires). Neither warrants blocking; both should be called out in the PR body so the human maintainer catches them during the manual tag/push/Zenodo dance.
