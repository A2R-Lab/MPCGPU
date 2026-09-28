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
figure-region renders of Figures 2, 3, and 4 from our ICRA 2024 paper,
arXiv:2309.08079v3 (https://arxiv.org/pdf/2309.08079). The MPCGPU code,
documentation, website and these figures are released under the repository's
MIT license. Figure numbers identify the experiments and explain the visuals.

Replace `pick-place.png` with a higher-resolution original when supplied. An
optional video should have controls, a static poster, no forced autoplay, and
an accessible description; the page must still work without it. Keep diagram
labels legible on mobile and use original vector exports where available.

## Result updates

Present the published results with their ICRA 2024 methodology. Add updated
measurements with their own hardware and protocol description after validation.
New results need source and dependency SHAs, hardware/toolchain, reference and
protocol hashes, repeat distributions, accuracy/quality checks, and a link to
the run record. Follow [the ICRA replication plan](../docs/icra-replication.md) to restore the
paper examples and document comparisons. Keep BibTeX visible without interaction.
