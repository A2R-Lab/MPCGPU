# Claude takeover — MPCGPU / in-tree GBD-PCG

Saved September 28, 2026 because Codex credits are running low. This is the
current entry point and supersedes the ordering in the older Codex handoff:
**start ICRA task correctness now; do not wait for timing to implement it.**

## User's latest instructions

- Continue MPCGPU and in-tree GBD-PCG stabilization, paper-example correctness,
  documentation and website development. Keep GBD-PCG inside MPCGPU, sharing
  the top-level GLASS dependency. Neither project may depend on GATO; do not
  advertise that absence in public copy, since MPCGPU predates it.
- GPU correctness and light, capped sequential builds are allowed on the shared
  box. **No timing until an explicitly assigned quiet window.** Coordinate before
  using the GPU; yield if another agent has a timing slot. Do not touch clocks,
  governors, sibling projects or other agents' workers.
- The user is the paper author. The project's code, documentation, website and
  our paper figures are MIT licensed. Retain third-party dependency notices.
- Website copy should be declarative; BibTeX stays always visible. These changes
  are implemented, except the latest requested hero sentence below.
- User asks what remains before merging to main and wants the principal ICRA
  examples replicated, first for correctness. Timing/performance claims come
  later. Do not equate the current figure-eight tests with paper replication.
- Hold pushes, main merges, releases, deployment and old standalone GBD remote
  archival until explicitly authorized. No new authorization was given.

## Exact stopping point

Repository: `/home/plancher/Desktop/MPCGPU`, branch `modernize-grid-glass`.
HEAD at handoff creation: `909c194` (research copy, licensing, replication plan).
The tree was clean before writing this handoff and updating the older handoff.
These handoff edits are left uncommitted so the next agent can review them.
No code, test or website implementation was changed during this interrupted turn.
No build, GPU test, timing worker or preview server was started this turn.

Read `CLAUDE.md`, `docs/development.md`, `docs/implementation-status-2026-09-28.md`,
`docs/icra-replication.md`, and the previous `docs/HANDOFF_codex_2026-09-28.md`.
The original audit is historical; the implementation-status document records
which findings have actually been fixed. Do not repeat the original audit as if
its already-fixed findings were current blockers.

## First small edit (not yet done)

In `website/index.html`, replace the hero sentence exactly with:

> MPCGPU accelerates nonlinear model predictive control with a GPU-optimized preconditioned conjugate gradient solver.

The existing text starts “MPCGPU accelerates nonlinear model predictive control
with parallel trajectory optimization…”. Keep the visible BibTeX, MIT framing
and published/current-results distinction. Do not add GATO comparisons.

## What is already working

- GRiD `65fd051198e4ccc6f609238bca4cfa780eb2b55a`, GLASS
  `8ce68a29bceb30c7764c9391d517a182d061697d`, QDLDL `df48100`;
  pytest-gpu-proof 0.4.0/schema 3. HTTPS recursive dependencies and exact manifest.
- Self-contained iiwa URDF and reproducible minimal GRiD generation; no sibling
  model dependencies or embedded second GLASS copy. Independent Pinocchio oracle.
- GBD memory/API guards, small-dimension fixes, deterministic float/double SPD
  demos, real-Schur replay and selected sanitizer checks.
- Reusable synchronous per-instance SQP workspaces; cached QDLDL symbolic
  structure; checked numerical failures; fresh/reused raw-state hash parity.
- Strict reference parsing, L2 tracking metrics, content-aware sequential build
  caching and separate correctness/timing profiles.
- Last full clean-source suite: **76 passed (51 GPU, 25 host), no skips**,
  source `f61d1ec`, verified receipt commit `c5cff26`. Anonymous fresh-clone
  quickstarts and host tests passed. Additional timer-disabled PCG/QDLDL passes
  at horizons 32/128/256/512 succeeded; N=64 is in the signed suite.
- Static website with our paper Figures 2/3/4, updated README/API/development
  docs, explicit native API limits. No site deployment.

**The current receipt is stale after `909c194` website/test/copy edits.** The
previous 76 passes are historical evidence, not an attestation of today's tree.
Only the 25 host/harness checks were rerun for that copy update. Refresh the full
receipt after the next implementation is stable; never carry or hand-edit it.

## Start ICRA replication without timing

Correctness does not need a quiet timing window: use fixed simulated control
cadence, deterministic inputs and timer-disabled builds. It cannot establish
real-time throughput or wall-clock-dependent closed-loop behavior.

1. Inventory the original protocol before changing examples. Paper copies are
   `/tmp/mpcgpu-paper-2309.08079.pdf` and `.txt`; original paper is
   https://arxiv.org/pdf/2309.08079. Relevant rendered pages are
   `/tmp/mpcgpu-paper-page4.png` and `-page5.png`. Verify exact protocol details:
   the plan records 100 ten-second five-goal trials, but task switching, seeds,
   cost weights and control policy still need recovery from original sources.
2. Inspect `examples/track_iiwa_pcg.cu`, `examples/track_iiwa_qdldl.cu`,
   `include/mpcsim.cuh`, settings, `tools/validate_track.cu`,
   `tools/gen_reference.cu`, and historical Git versions. There is no
   `examples/pcg.cu`; the current Make demos use `tools/validate_track.cu`.
3. Inspection started only with the PCG legacy example. It loops 5x5 start/goal
   combinations, skips most equal pairs, sweeps tolerances, but has a final
   `break` after the first combination. Do not assume that is a maintained
   five-goal switching task or just remove the break and call it replicated.
   It also writes result filenames that are not sufficient trial provenance.
4. `examples/trajfiles/` contains 25 legacy joint trajectories but only three
   EE files (`0_0`, `0_1`, `0_2`). Those paired fixtures were regenerated for the
   current figure-eight correctness path. Preserve them; do not assume all
   legacy pairs are complete, valid under the corrected model, or paper goals.
5. Recover model/EE frame, initial states, goals/switching rule, timestep,
   integrator, costs/limits, regularization, warm start, solver limits/tolerances
   and trial generation. Explicitly document anything missing. Preserve corrected
   physics and terminal-cost fixes rather than restoring known historical bugs.
6. Add a shared task/config path for PCG and QDLDL, initially a deterministic
   single trial. Check finite states/controls, goal switching and completion,
   tracking error, exits and applicable limits. Emit machine-readable quality
   data and a documented one-command example; then expand coverage.
7. Add tests and update the exact node manifest when necessary. Fresh-clone
   verification and trajectory visualization follow. Update the replication
   checklist as evidence lands. Do not claim full ICRA replication prematurely.

Native MPC supports float iiwa (14 states/7 controls); GBD supports float/double.
`tools/build.py` defaults to the correctness profile: `MPCGPU_CORRECTNESS=1`,
`TIME_LINSYS=0`, wall-clock SQP stopping disabled. Reuse this builder rather than
inventing untracked compilation commands. Read its target map for new examples.

## Build/receipt discipline and traps

- One nvcc at a time, under a 36 GiB/no-swap scope:
  `systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir ...`.
  Check shared-box activity first. Host-only tests: `.venv/bin/python -m pytest
  test/test_host.py test/test_timing.py -q`.
- Source commit -> capped `test/run_gpu_proof.sh` -> verify per development docs
  -> receipt commit -> authorized push -> green CI. Signing requires the valid
  keyholder environment; do not fabricate evidence if unavailable.
- Never run parent submodule update before staging a new pin: the old index
  gitlink can reset the submodule. No upstream dependency edits in this task.
- Use `--extra="-DFLAG"` (with equals) for a single negative-looking build flag.
- No generic binding/API expansion is needed to close this task. Keep explicit
  SPD/eta-residual and synchronous workspace contracts. Partly duplicated SQP
  orchestration is a documented deferral, not a reason to build a framework.
- Old dynamics, EE-frame, terminal-cost aliasing, regularization and L1/L2 final
  metric differences invalidate unqualified historical performance comparisons.

## Timing stays parked

Prepared plan: `tmp/timing-prepared/candidate-20260928/plan.json` — 20 immutable
variants, 60 repeats (five horizons x two backends x fresh/reused x three).
Estimated reservation **30–45 minutes**, provisional. This is same-source
workspace A/B on figure-eight tracking, not paper-task replication or old main.
Paper-task timing needs its own manifest and budget once correctness is ready.
Optional compile timing is another unprepared 15–30-minute leg.

The prior handoff has the launcher, cwd, prerequisites, outputs and STOP/resume
instructions. Read `docs/timing.md` before any timing action. Dry-run is safe:

```bash
.venv/bin/python tools/timing.py run tmp/timing-prepared/candidate-20260928/plan.json \
  tmp/timing/check-only --dry-run
```

Native/build changes invalidate prepared identities: reprepare, never bypass
hash/receipt preflight. Documentation-only changes can leave binary identities
valid while still invalidating the source receipt. No timing permission follows
from a successful dry-run or an available advisory lock.

## Remaining merge-to-main gates

- Finish the requested paper-task correctness milestone, or get explicit user
  agreement to a narrower merge with that milestone clearly marked incomplete.
- Audit the final diff/API contracts; preserve bounded scope and accurate docs,
  README/quickstart/examples/site. Distinguish published from current results.
- Commit final source, run and verify the full clean-source receipt; rerun
  fresh-clone quickstarts for the final candidate, not just an earlier checkpoint.
- Collect/analyze the assigned quiet-window regression batch; resolve unexplained
  quality/performance regressions. No speed claims until measured. Current host
  is RTX 5090/Core Ultra 9 285K/CUDA 13.2.86 versus the paper's 4090/12900K/12.1;
  raw latency equality is not the replication criterion.
- Obtain push authorization, check remote heads without overwriting others,
  publish candidate plus receipt and obtain green host/receipt CI. Main merge
  and website publication require their own clearance.

No measured current performance comparison or merge-ready claim exists yet.
The two historical development branch names were mirrors, not separate solver
implementations. Recheck remote state before publishing; do not force-push.

## Quick arrival

```bash
cd /home/plancher/Desktop/MPCGPU
git status --short
git log -6 --oneline
git submodule status
```

Local evidence logs: `/tmp/mpcgpu-*-20260928.log`; disposable validation clone:
`/tmp/mpcgpu-clean-Emt4eh/repo`. Preserve evidence; never use that clone as the
maintained working tree. The older handoff lists specific logs and timing fields.
The last copy revision had static host checks only: browser navigation to the
local file was policy-blocked. Codex did not route around it through localhost
or another browser. Do not describe it as newly visually verified.
