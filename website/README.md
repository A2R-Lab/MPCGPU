# MPCGPU project page

Static HTML/CSS; no framework, third-party scripts, remote fonts, analytics, or
build toolchain. Preview from the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory website
```

Open http://127.0.0.1:8765. This is a local draft, not a deployed site.
Publication and switching development URLs to main require a separate release
decision. Branch-linked quickstart files must be pushed before publication.

## Assets and provenance

`assets/architecture.png` and `pick-place.png` are figure-region renders of Figures 2 and 3
from our ICRA 2024 paper, arXiv:2309.08079v3 (https://arxiv.org/pdf/2309.08079).
`assets/current-linsys-time.png` and `current-sqp-iterations.png` are rendered by
`tools/plot_timing.py` from the September 30, 2026 timing runs of the current code (see
docs/icra-replication.md). The MPCGPU code, documentation, website and these figures are
released under the repository's MIT license.

Replace `pick-place.png` with a higher-resolution original when supplied. An
optional video should have controls, a static poster, no forced autoplay, and
an accessible description; the page must still work without it. Keep diagram
labels legible on mobile and use original vector exports where available.

## Result updates

The results section shows the current code's measurements with their hardware and date.
Historical results for the original code are in the paper, which the page links. Refresh
the figures only from complete timing runs with source and dependency SHAs, hardware,
protocol, repeat distributions and quality checks, and explain any change in
docs/speedup-attribution.md. Keep BibTeX visible without interaction.
