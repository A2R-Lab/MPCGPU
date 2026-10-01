# Current documentation

The modernized code is on `main`. GBD-PCG is part of
this repository; GRiD/GLASS are pinned dependencies, not sibling working copies.

- [Quickstart](../README.md): correctness demos, prerequisites and configuration.
- [API contracts](api.md): supported inputs, ownership, precision and limitations.
- [Development](development.md): reproducible generation, validation and receipts.
- [Timing handoff](timing.md): prepared workloads and exclusive-window procedure.
- [ICRA replication](icra-replication.md): paper-task examples and comparison gates.
- [Audit](audit-and-plan-2026-09-28.md): findings and original planned gates.
- [Implementation status](implementation-status-2026-09-28.md): completed work,
  actual evidence and remaining blockers.
- [Website](../website/README.md): local draft, provenance and publication checklist.

## Historical material

`modernization.md`, the July/August `benchmark_3way_*` documents and
`nsweep_persolve_2026-08-03.md` describe prior checkpoints. They remain for
traceability, not current build instructions or candidate performance claims.
Old full-Q+R regularization, old dynamics, L7/EE frame choices, terminal-cost
aliasing and L1/L2 final-error differences must not be mixed in comparisons.

The paper is a separate published protocol. New figure-eight measurements
cannot refresh every paper figure or substantiate the pick-and-place results.
Keep published visuals labeled as published; add a separate current-software
results panel only after validated, comparable quiet-window data.
