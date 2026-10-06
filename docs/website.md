# MPCGPU project page

The page is `docs/index.html` with `docs/style.css` and `docs/assets/`: static HTML/CSS, no
framework, third-party scripts, remote fonts, analytics or build toolchain.

**Deployment is automatic.** GitHub Pages serves the `docs/` folder of `main`; every push to
`main` that changes it republishes the site. `docs/.nojekyll` makes Pages serve the folder as
is, so the Markdown documents beside the page are not run through Jekyll.

Preview locally from the repository root before pushing:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8765. `test/test_host.py` checks that every local link and image on the
page resolves.

## Assets and provenance

`assets/architecture.png` and `pick-place.png` are figure-region renders of Figures 2 and 3
from our ICRA 2024 paper, arXiv:2309.08079v3 (https://arxiv.org/pdf/2309.08079).
`assets/current-linsys-time.png` and `current-sqp-iterations.png` are rendered by
`tools/plot_timing.py` from the October 5, 2026 timing runs of source `a684e78` (see
docs/icra-replication.md). The MPCGPU code, documentation, website and these figures are
released under the repository's MIT license.

Replace `pick-place.png` with a higher-resolution original when supplied. An
optional video should have controls, a static poster, no forced autoplay, and
an accessible description; the page must still work without it. Keep diagram
labels legible on mobile and use original vector exports where available.

## Result updates

The results section shows dated measurements with their source and hardware.
They do not automatically describe the latest dependency pins.
Historical results for the original code are in the paper, which the page links. Refresh
the figures only from complete timing runs with source and dependency SHAs, hardware,
protocol, repeat distributions and quality checks, and explain any change in
docs/speedup-attribution.md. Keep BibTeX visible without interaction.
