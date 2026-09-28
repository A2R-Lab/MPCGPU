# MPCGPU quiet-window handoff

**No timing has been collected for this modernization candidate.**
Correctness work can run on the shared box; this launcher cannot.

## Coordinator fields

| Field | Handoff |
| --- | --- |
| Launcher | `.venv/bin/python tools/timing.py run <prepared>/plan.json tmp/timing/<unique-run>` |
| Working directory | `/home/plancher/Desktop/MPCGPU` |
| Reservation | Provisionally **30–45 minutes** for runtime; release early if finished. This is not a measured duration. Optional compile-speed work needs a separately prepared 15–30-minute leg. |
| Prerequisites | Prepared binaries, clean source and current complete signed receipt, matching hashes, explicit exclusive slot, shared lock available, no other CPU/GPU work. |
| Output | Unique run directory: machine/commit/receipt provenance, one subdirectory per independent repeat, raw native samples, stdout and success-only verdict JSON. |
| Stop/resume | Touch `<run>/STOP` to stop between repeats. Ctrl-C for urgent cancellation; confirm the exact owned child exited. Interrupted repeats have no success verdict. Resume into a **new** directory with `--resume <old-run>`; never resume suspended timing processes. |

The `/tmp/a2rlab-timing.lock` is an advisory coordination lock, not a guarantee
that unrelated agents are idle. The coordinator must obtain the actual quiet
window. The runner never changes clocks/governors, installs packages or builds.

## Preparation (allowed before the window)

After source changes and QDLDL build are complete, prepare sequentially:

```bash
systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir \
  .venv/bin/python tools/timing.py prepare tmp/timing-prepared/candidate
.venv/bin/python tools/timing.py run tmp/timing-prepared/candidate/plan.json \
  tmp/timing/check-only --dry-run
```

This compiles but **does not execute** timing workloads. Default horizons are
32,64,128,256,512, both PCG and QDLDL, reused/fresh workspace, three repeats.
The two workspace modes use identical candidate code/dependencies/math; fresh
is an attribution baseline, **not** old main or a reproduction of the paper.
Any source/reference/header/dependency change requires a new preparation.
Large cooperative launches can fail occupancy checks; record unsupported
points, do not silently reduce the requested horizon.

## Run only after assignment

```bash
MPCGPU_QUIET_WINDOW=1 .venv/bin/python tools/timing.py run \
  tmp/timing-prepared/candidate/plan.json tmp/timing/assigned-window
```

Use a unique output name. The runner verifies the receipt before collecting,
checks hashes, holds the shared lock, alternates workload order by repeat,
and fails on child failure, missing/nonfinite samples or incomplete tracking.
The broad divergence ceiling is only a safety guard: final analysis must compare
per-horizon quality, iteration caps and repeat variation; it does not establish
that a long-horizon operating point is acceptable.

Internal SQP time and full-process wall time are different metrics. Native SQP
samples exclude some call setup/cleanup; process wall time includes startup,
simulation and output. Do not label either as isolated end-to-end solver latency.
Raw output includes iteration/exit distributions. Report first-solve effects
separately when deriving steady-state summaries; retain raw samples.

After collection, compare tracking before claiming speedups, summarize medians
and tails across repeats, investigate regressions, then update documentation or
the website's *current software* panel. Published paper plots remain labeled as
published and are not replaced by nonmatching figure-eight experiments.

The old `time_persolve.sh` and sibling-dependent `run_3way_iiwa.sh` now fail
with migration instructions. This reservation covers MPCGPU runtime collection.

After this batch, follow [the ICRA example replication plan](icra-replication.md).
The paper-task runs need a separate prepared batch and assigned timing slot.
