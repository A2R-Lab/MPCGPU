# MPCGPU agent guide

Read docs/README.md, docs/development.md and docs/audit-and-plan-2026-09-28.md.
This repository contains MPCGPU and its in-tree GBD-PCG implementation.
Model inputs, code generation and tests are self-contained.
GRiD and GLASS are pinned submodules; do not edit their upstream code here.

## Standing rules

- Shared box: GPU correctness is allowed, timing requires an explicitly assigned
  exclusive window. No clocks/governors changes or benchmark launches by default.
- Sequential nvcc only; use a 36 GiB/no-swap systemd scope for correctness builds:
  `systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir ...`.
- Preserve other agents' files/processes. Do not run sibling project workloads.
- Short single-line commits, no Co-Authored-By. Hold pushes, PRs, main merges,
  releases and website deployment until explicitly cleared.
- The project page is `docs/index.html`; GitHub Pages serves `main` `/docs`, so any push to
  `main` that touches `docs/` republishes http://a2r-lab.org/MPCGPU/ (see docs/website.md).
- Commit source -> full clean-source GPU receipt -> receipt commit -> authorized
  push -> green CI. Do not edit receipts, carry old results, or sign subsets.
- Never reuse timing output after a failed run. Read docs/timing.md before timing.

## Architecture and traps

GRiD generation uses tools/iiwa14.urdf and an explicit minimal algorithm list,
without collision/contact or an embedded GLASS copy. Check reproducibility with
make check-codegen. Exactly pin dependencies in tools/dependencies.json; update
the gitlinks too. Do not run parent submodule update before staging a new pin:
it resets the submodule to the old index gitlink.

Owned CUDA uses glass::block:: explicitly. GBD-PCG stays cooperative/grid-wide;
GLASS's single-block PCG is not a replacement. Scratch includes all three halo
slots, including N=1. Device launch dimensions are compile-time specializations.

Reusable SqpWorkspace owns streams, handle and fixed allocation slots per
backend/dimensions/device. It is synchronous and cannot be shared concurrently.
QDLDL's symbolic structure is reused, its numeric factorization is not.
Keep fresh/reused state-hash parity gates when changing workspace layout.

Tracking uses L2 EE-position error throughout. Corrected dynamics, terminal-cost
shared-memory alias fix and named EE frame invalidate comparisons with older
fixtures/results. August historical numbers are not current performance claims.
PCG eta can under-report true residual on poorly conditioned systems; do not
silently loosen gates or describe eta convergence as a true-residual guarantee.

See docs/development.md for builds/receipt commands. Python is tooling, not a
high-level solver wrapper. GBD-PCG is maintained here; the old GitHub repository
has not yet been remotely archived.
