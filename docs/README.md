# Documentation

MPCGPU lives in this repository; GBD-PCG (its linear-system solver), GRiD and GLASS are pinned submodules.
It is maintained as the ICRA 2024 artifact and a header-only library (see
[Project status](../README.md#project-status)); fixes and pin updates are in
scope, new solver capabilities are not.

- [Quickstart](../README.md): correctness demos, prerequisites and configuration.
- [API contracts](api.md): supported inputs, ownership, precision and limitations.
- [Development](development.md): reproducible generation, validation, receipts and timing.
- [ICRA replication](icra-replication.md): the paper's pick-and-place task, its recovered protocol
  and current results.
- [Speedup attribution](speedup-attribution.md): why the current results differ from the paper's.
- [Website](website.md): the project page in this folder and how it deploys.

The paper is a separate published protocol. Older measurements used different dynamics, EE
frames, costs and metrics, so they are not comparable with current results; the paper remains the
record of the published code's results.
