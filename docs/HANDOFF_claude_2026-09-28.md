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

## Current state (Claude, evening of September 28)

Repository `/home/plancher/Desktop/MPCGPU`, branch `modernize-grid-glass`, local only.

| Commit | Content |
| --- | --- |
| `9fa5464` | Hero sentence; ICRA pick-and-place task, checks, gates, docs |
| `2f52742` | ICRA paper-task timing workloads (`tools/timing.py prepare --task icra`) |
| `513a9a4` | Signed receipt: 86 passed, no skips, verified against source `2f52742` |

- The hero sentence now reads exactly as requested above.
- ICRA correctness is done for the pick-and-place circuit. The recovered protocol,
  its sources, every difference from 2024, the evidence table and the findings
  are in [icra-replication.md](icra-replication.md). The key recovery: the 2024
  experiments ran in zero gravity, which the task build restores.
- Both backends pass every task check at N = 64…512, QDLDL also at N = 32. PCG at
  N = 32 misses the second goal with the default tolerance and passes with the
  tighter values from the paper's own sweep.
- The paper protocol has no joint-limit terms; joint 3 leaves its URDF range.
  This is reported, not enforced. A joint-limit variant is an open user choice.
- Figure-eight results are unchanged to every printed digit.
- Header changes invalidated `tmp/timing-prepared/candidate-20260928`. Both plans
  were re-prepared compile-only for `513a9a4` as `tmp/timing-prepared/fig8-513a9a4`
  and `tmp/timing-prepared/icra-513a9a4`; check their dry runs before use.
- No push, merge, deployment, timing run or remote archival happened.

Earlier evidence is in [implementation-status-2026-09-28.md](implementation-status-2026-09-28.md).
The takeover notes below remain for build and receipt discipline.

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

## Timing is prepared, not run

Both plans were prepared compile-only for source `2f52742` / receipt `513a9a4`,
and both dry runs pass. Read [timing.md](timing.md) before any timing action.

| Plan | Runs | Reservation (provisional, unmeasured) |
| --- | ---: | --- |
| `tmp/timing-prepared/fig8-513a9a4/plan.json`: figure-eight workspace A/B | 60 | 30–45 min |
| `tmp/timing-prepared/icra-513a9a4/plan.json`: ICRA Figures 4, 5 and 6 | 126 | 60–90 min; split with `--horizons` if needed |

```bash
.venv/bin/python tools/timing.py run tmp/timing-prepared/icra-513a9a4/plan.json \
  tmp/timing/icra-check-only --dry-run
```

Any change under `include`, `tools` or `examples` invalidates both plans: prepare
again, never bypass the hash or receipt preflight. Documentation-only commits keep
them valid. No timing permission follows from a dry run or a free advisory lock.

## Remaining merge-to-main gates

- Paper-task correctness is done. Decide whether a joint-limit variant is wanted.
- Audit the final diff/API contracts; preserve bounded scope and accurate docs,
  README/quickstart/examples/site. Distinguish published from current results.
- The receipt at `513a9a4` covers the current source. Any later source change
  needs a new receipt and a new fresh-clone quickstart.
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
