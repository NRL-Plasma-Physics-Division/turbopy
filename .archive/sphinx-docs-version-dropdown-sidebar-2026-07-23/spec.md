# Spec: Version dropdown + sidebar categorization for turboPy Sphinx docs

STATUS: BLOCKED

## Open questions

1. **Dropdown approach preference.** My verdict is approach (a) — the ReadTheDocs flyout — because it is zero-code and turboPy is already RTD-hosted. But if the maintainers want the dropdown to also appear in *local* `make html` builds (e.g., for offline PDFs, or a mirror hosted somewhere other than readthedocs.io), (a) will not satisfy that requirement and we would need approach (c) `sphinx-multiversion`. **Confirm: is "renders on readthedocs.io only" acceptable, or must the dropdown also work in local builds?**
2. **RTD dashboard access.** Approach (a) requires an admin on the `turbopy` RTD project to click through the "Versions" tab and activate each tag. **Who owns that account, and can they perform the activation after this PR merges?** If nobody has admin access, approach (a) is not viable and we must fall back to (c).
3. **Tag inclusion policy.** Which tags should appear in the dropdown — every tag ever pushed, only tags matching `v*`, or only the last N? This is a knob in the RTD "Automation Rules" UI (for approach a) or in the `sphinx-multiversion` `smv_tag_whitelist` regex (for approach c).

Absent answers, the spec below documents approach (a) as the primary plan with (c) sketched as a fallback.

---

## Overview

Two related but separable changes to turboPy's Sphinx docs:

1. **Version dropdown** — surface a "select a tagged release" dropdown in the ReadTheDocs sidebar so users on `latest` can jump to `v2023.06.09`, etc.
2. **Sidebar categorization** — group the current flat 12-page toctree in `docs/index.rst` into four semantic sections with sidebar captions.

## Background audit (verified from the repo)

- `docs/conf.py` uses `html_theme = "sphinx_rtd_theme"` and already populates `html_context` with `display_github`, `github_user`, `github_repo`, `github_version`, `conf_py_path`. It does NOT populate `html_context['versions']` and does NOT set `html_theme_options['flyout_display']`.
- `.readthedocs.yml` exists (v2 config), uses `conda: environment: environment.yml`, and builds `docs/conf.py`. No `build.os` / `build.tools` block.
- `environment.yml` installs `sphinx_rtd_theme` via pip (unpinned).
- The docs badge in `README.md` confirms the RTD project slug is `turbopy` (`https://turbopy.readthedocs.io/en/latest`).
- `docs/index.rst` has one flat `.. toctree::` with `:caption: Contents` listing 12 pages.

## Approach verdict for the version dropdown

**Primary: Approach (a) — the ReadTheDocs flyout, plus a small `.readthedocs.yaml` modernization.**

Justification:
- The docs are hosted on `readthedocs.io`. RTD auto-injects the flyout (bottom-left "v: latest" widget) on every build it hosts; it reads its list from *whichever git refs the RTD project has been told to build*. No Sphinx-side code is required for the flyout to appear.
- Approach (b) — `sphinx_rtd_theme >= 2.0`'s `flyout_display: 'attached'` in-page selector — is a *display* toggle for the same RTD flyout data; adopting (b) without (a) yields an empty selector. Deferrable follow-up.
- Approach (c) — `sphinx-multiversion` — rebuilds every tag into `/vX.Y.Z/` subdirectories. Only necessary for locally-hosted docs. Adds a build-time dependency, reworks the RTD build command, doubles/triples build times. Unnecessary given RTD already provides the feature for free.

## Files to create or modify

### 1. `/Users/ndisner/github/turbopy/docs/index.rst` — split the flat toctree

Replace the single `.. toctree::` block with four captioned toctrees:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   overview

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   simulation_lifecycle
   clock
   grids
   dynamic_factory

.. toctree::
   :maxdepth: 2
   :caption: Components

   physics_modules
   compute_tools
   diagnostics
   sharing_data

.. toctree::
   :maxdepth: 2
   :caption: Reference

   input_files
   api
```

Rationale:
- `dynamic_factory` under **Core Concepts** — it describes the registry mechanism used *by* all component types, so it's prerequisite reading.
- `sharing_data` under **Components** — the `_resources_to_share` / `_needed_resources` protocol is only meaningful once you're writing a `PhysicsModule` or `Diagnostic`.

### 2. `/Users/ndisner/github/turbopy/.readthedocs.yml` — modernize to v2 build spec

Rename to `.readthedocs.yaml` (RTD accepts either) and update contents:

```yaml
# .readthedocs.yaml
# See https://docs.readthedocs.io/en/stable/config-file/v2.html
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "mambaforge-22.9"

conda:
  environment: environment.yml

sphinx:
  configuration: docs/conf.py
  fail_on_warning: true

formats:
  - pdf
  - epub
```

Notes:
- `fail_on_warning: true` enforces acceptance criterion (i) on RTD, not just locally.
- If the rename is objectionable, keep `.readthedocs.yml` — only the contents matter.

### 3. `/Users/ndisner/github/turbopy/docs/conf.py` — no functional change required

Existing `html_context` block is compatible with the RTD flyout. Do NOT add a hand-rolled `html_context['versions']` list — that would shadow what RTD injects.

## Reference patterns

- **Multi-toctree pattern:** Sphinx's own documentation uses several `.. toctree::` blocks each with its own `:caption:`. No template work required.
- **`html_context` pattern:** already in use in `docs/conf.py` (lines 112–118) for "Edit on GitHub". Keep that pattern.
- **RTD config v2 pattern:** https://docs.readthedocs.io/en/stable/config-file/v2.html.

## Phases

### Phase 1 — Sidebar categorization
- Edit `docs/index.rst` per section "Files to create or modify" #1.
- Build locally: `cd docs && make clean && make html`. Confirm 0 warnings and 4 sidebar section captions.

### Phase 2 — RTD build config modernization
- Rename `.readthedocs.yml` → `.readthedocs.yaml` and expand per section #2.
- No local test possible — merged branch triggers RTD build.

### Phase 3 — Manual dashboard activation (post-merge, not code)
- See "Manual actions after merge" below.

Each phase is independently mergeable.

## Manual actions after merge (dashboard, not code)

These deliver the version dropdown for approach (a). Requires admin on the `turbopy` project at https://readthedocs.org/projects/turbopy/.

1. Log in to https://readthedocs.org and open the `turbopy` project dashboard.
2. Go to **Versions**. Toggle each git tag that should appear to **Active** and **Public**. Run `git tag -l` locally to see the candidate list.
3. (Recommended.) **Admin → Automation Rules** → add an **Activate version** rule with regex `^v\d{4}\.\d{2}\.\d{2}$` (matches the existing `vYYYY.MM.DD` scheme) so future tags auto-activate.
4. (Optional.) **Admin → Advanced Settings** → set **Default version** to `stable`.
5. **Builds → Build version** on `latest` so the flyout picks up newly activated versions.
6. Load `https://turbopy.readthedocs.io/en/latest/` and confirm the bottom-left flyout lists the activated tags under **Versions**.

## Acceptance criteria

- [ ] `cd /Users/ndisner/github/turbopy/docs && make html` completes with **0 warnings and 0 errors**. In particular, no `document isn't included in any toctree` warning for any of the 12 pages.
- [ ] The generated `docs/_build/html/index.html` and every subpage sidebar shows **four sidebar section captions** — "Getting Started", "Core Concepts", "Components", "Reference" — each expanding to the pages listed above in the given order.
- [ ] After manual dashboard actions, `https://turbopy.readthedocs.io/en/latest/` shows a bottom-left RTD flyout with a **Versions** section listing every activated tag; clicking a tag navigates to that tag's build. If not yet done, the PR description notes this as a follow-up.
- [ ] `.readthedocs.yaml` validates against RTD config v2 — verified by a successful RTD build on the PR branch.

## turboPy-specific checks

- Registry: N/A (docs-only change)
- Shared resources published: N/A
- Shared resources consumed: N/A
- Grid dispatch: N/A
- Numerical tolerance: N/A

## Notes for the coder

- **Latent risk on `fail_on_warning: true`.** Older tags may currently build with warnings; enabling this retroactively could break tag builds after step 2 of manual actions. If maintainers push back, drop `fail_on_warning` from `.readthedocs.yaml`.
- **Orphan pages check.** After the split, all 12 pre-existing `.rst` pages must appear in exactly one of the four new toctrees.
- **Do NOT change `docs/conf.py`** for the dropdown. `html_context` is fine as-is; adding versions there would conflict with RTD's injection.
- **Do NOT add multiversion tooling** to the repo unless the user answers Open Question 1 in a way that rules out approach (a).
- **Do NOT touch old tags** to make them build cleanly — that's a separate concern, not this PR.
- **PR description should list the manual actions.** The dropdown is a two-part delivery: code merge + dashboard config. Do not claim the feature ships until both halves are done.
