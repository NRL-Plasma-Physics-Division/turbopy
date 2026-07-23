# Test results: Document 2D solve additions and prepare v2026.07.23 release

STATUS: READY

## Pytest
- Suite: 133 passed, 0 failed, 0 skipped
- Command: `python -m pytest -q`
- New test files: none
- Updated test files: none

The 5 pre-existing warnings are unchanged from the baseline (3 × `UserWarning: No Grid Found.` in
`test_bos.py`, 1 × `RuntimeWarning: invalid value encountered in divide` in
`test_computetools.py`, 1 × `DeprecationWarning: resource-sharing API has changed` in
`test_core.py`).  No new warnings introduced.

No new unit tests were written.  This is appropriate: the change is
documentation text and metadata only.  The existing suite provides the
required smoke check that the docstring edits did not corrupt Python syntax
or alter runtime behavior.

## Coverage on changed lines
Not applicable.  Changed lines are docstring text in `turbopy/diagnostics.py` and
a single literal replacement in `turbopy/__version__.py`.  The ≥80% coverage
target does not apply to docstring/metadata edits.

## Baseline diff
- Baseline warnings (on `main` before stash-pop): 0
- After warnings (post-changes build): 0
- New warnings introduced: none
- Warnings resolved: none

Both `make html` runs exited 0 with `build succeeded.` and produced no
WARNING/ERROR/CRITICAL lines in the build log.

## Failures
None.

## Acceptance criteria coverage

- [x] **AC1** — `docs/compute_tools.rst` contains a `:class:\`turbopy.computetools.PoissonSolver2D\`` bullet and the strings `Dirichlet`, `Grid2DCartesian`, `Grid2DCylindrical`, and `TypeError` — verified by grep; all four strings found in lines 91–101.
- [x] **AC2** — `docs/compute_tools.rst` `FiniteDifference2D` bullet lists all four cylindrical operators `ddr`, `ddz`, `del2_r`, `del2_z` — verified by grep; all four present in lines 83–84.
- [x] **AC3** — `docs/diagnostics.rst` `HistoryDiagnostic` bullet contains no `NotImplementedError` and explicitly mentions `x`/`y` and `r`/`z` coordinate attachment — `NotImplementedError` absent (grep returned nothing); coordinate language present at lines 72–74.
- [x] **AC4** — `HistoryDiagnostic` class docstring in `turbopy/diagnostics.py` contains no `Raises NotImplementedError` section — grep across the entire file returned no matches.
- [x] **AC5** — `cd docs && make html` exits 0 with no new Sphinx warnings vs baseline — confirmed; 0 warnings in both runs.
- [x] **AC6** — `python -c "from turbopy import __version__; print(__version__)"` prints `2026.07.23` — confirmed.
- [!] **AC7** — `python -c "from importlib.metadata import version; print(version('turbopy'))"` prints the new version — **NOT MET in current environment**. Returns `2023.6.9` because the installed editable dist-info at `/opt/anaconda3/envs/rigid-beam/lib/python3.13/site-packages/turbopy-2023.6.9.dist-info` predates the version bump and was never refreshed. The `pyproject.toml` wiring (`version = {attr = "turbopy.__version__.__version__"}`) is correct — `turbopy.__version__.__version__` resolves to `2026.07.23` at source — but `importlib.metadata` reads the cached dist-info on disk, not the live attribute. Running `pip install -e .` in the active conda env will regenerate the dist-info and satisfy this criterion. This is an environment setup gap, not a code defect.
- [x] **AC8** — `README.md` citation block year/version strings match `v2026.07.23`; Zenodo concept-DOI URL `https://doi.org/10.5281/zenodo.3973692` is unchanged — verified by grep; citation prose, BibTeX key, `title`, `version`, `month`, `year` all reference 2026.07.23; concept DOI at line 87 and `turbopy_project` entry unchanged.
- [x] **AC9** — `pytest` passes — 133 passed, 0 failed.
- [x] **AC10** — No git tag created, no push performed — `git tag -l 'v2026*'` returns nothing; `git status` shows only the 5 expected files modified (plus `.pipeline/spec.md` and the untracked `.pipeline/changes.md` and `.archive/` directory).

## Notes for reviewer

1. **AC7 requires `pip install -e .` before it can be verified.** The conda env `rigid-beam` has the package installed as editable from a prior session when the version was `2023.6.9`. The dist-info will not auto-update on a version bump; it is regenerated only when `pip install -e .` is re-run. The maintainer should run `pip install -e .` in the target environment before publishing. The spec's wiring (`pyproject.toml` `version = {attr = "turbopy.__version__.__version__"}`) is correct and will produce `2026.07.23` after reinstall.

2. **Built HTML confirms the new version.** `docs/_build/html/index.html` `<title>` reads `turboPy 2026.07.23 documentation`, confirming that `docs/conf.py` (which imports from `turbopy.__version__` at build time, not from the installed dist-info) picks up the new version correctly.

3. **Modified files match the spec exactly.** The six files shown in `git diff --stat` are: `.pipeline/spec.md` (pipeline artifact), `README.md`, `docs/compute_tools.rst`, `docs/diagnostics.rst`, `turbopy/__version__.py`, `turbopy/diagnostics.py` — all named in `changes.md`. No unintended files were touched. The two untracked entries (`.archive/` directory, `.pipeline/changes.md`) are pipeline artifacts, not source changes.

4. **`HistoryDiagnostic.initialize()` runtime code is untouched.** The 2D branches at lines 802–815 (Grid2DCartesian) and 809–815 (Grid2DCylindrical) that attach `x`/`y` and `r`/`z` coordinates are unchanged. The docstring now accurately describes behavior that was already present in code.

5. **No new test files are warranted.** All changed lines are documentation text or a version tuple literal. The existing 133-test suite provides adequate smoke-check coverage that the edits did not corrupt syntax or module importability.
