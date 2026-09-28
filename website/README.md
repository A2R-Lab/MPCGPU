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

`assets/architecture.png`, `pick-place.png`, and `published-scaling.png` are
unaltered figure-region renders of Figures 2, 3, and 4 from the authors' paper,
arXiv:2309.08079v3 (https://arxiv.org/pdf/2309.08079). Their captions remain
attributed. The new owned HTML/CSS is MIT; paper figures retain their source
attribution and are not a new experimental result or a third-party relicensing.

Replace `pick-place.png` with a higher-resolution original when supplied. An
optional video should have controls, a static poster, no forced autoplay, and
an accessible description; the page must still work without it. Keep diagram
labels legible on mobile and use original vector exports where available.

## Result updates

Keep the published-study panel separate from a future current-software panel.
New results need source and dependency SHAs, hardware/toolchain, reference and
protocol hashes, repeat distributions, accuracy/quality checks, and a link to
the run record. Do not replace paper plots with nonmatching figure-eight data.
