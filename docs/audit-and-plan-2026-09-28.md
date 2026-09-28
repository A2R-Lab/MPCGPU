# MPCGPU / in-tree GBD-PCG: audit and merge-readiness plan

> Execution began after user approval on September 28. This document preserves
> the original audit findings and proposed scope; the authoritative completion
> ledger is [implementation status](implementation-status-2026-09-28.md).
> User decisions: keep GBD-PCG in-tree with self-contained model inputs;
> MIT for MPCGPU code, documentation, website and our paper figures;
> timing waits for an assigned quiet window.

Date: 2026-09-28. Source inspected: MPCGPU `3cdf45f` on
`modernize-grid-glass`. This is a source/contract audit and execution plan,
**not a new correctness receipt or performance certification**.

## Scope and decisions

- User confirmed: **keep GBD-PCG inside MPCGPU**, sharing dependencies and
  validation. Do not revive a second implementation or independently maintained
  GLASS pin. No separate GBD-PCG code merge is needed for this checkout.
- Keep work scoped to MPCGPU and in-tree GBD-PCG. Leave other project workloads
  and upstream GRiD, GLASS and pytest-gpu-proof implementations unchanged.
- Shared-box work first: source changes, host tests, bounded correctness,
  sequential CUDA builds. Check available RAM; use a 36 GiB/no-swap scope and
  one compiler process. Yield GPU correctness/build activity if another agent
  is assigned timing. No timing, clock/governor changes, background jobs, or
  automatic timing reservation implied by this plan.
- Preserve the local documentation commit and all preexisting work. Do not
  reset branches, merge main, push, archive repositories, publish a website, or
  release a package without the appropriate user authorization. Existing
  CLAUDE.md explicitly holds pushes until cleared.
- Short single-line commits; no Co-Authored-By. After fingerprinted changes:
  committed clean source -> full correctness receipt -> receipt commit ->
  authorized push -> green CI. Never hand-edit signed receipts.
- Do not silently weaken numerical gates or overwrite old results. Attribute
  model, arithmetic, precision, integration, or reference changes first.

## Verified checkout and upstream state

Read-only GitHub checks were made during this audit; recheck before execution.

| Component | Current consumer state | Published target / disposition |
| --- | --- | --- |
| MPCGPU main | `0efde8c` | Remote main unchanged at audit time |
| MPCGPU development | Local `3cdf45f`; remote `modernize-grid-glass` and `modernizing-tests` both `c3c2df2` | One local documentation commit ahead; branches are mirrors, not competing implementations |
| GBD-PCG | In-tree under `GBD-PCG/` | Keep integrated; old standalone GitHub repo is **not actually archived**, despite local wording saying retired |
| GRiD | `e31f7bd` | Published `modernizing-tests` is `65fd051`; main is older `0a6c18e`. Sibling working checkout is newer `96a9117`, not the published branch head at inspection |
| GLASS | `7c495bd` | Published main `8ce68a2` |
| pytest-gpu-proof | Bootstrap/CI use unbounded `>=0.1`; receipt generated with 0.1.0/schema 1 | Upstream main `7d32f49`, project version 0.4.0; use matched 0.4.0 producer/verifier and schema-3 policy |
| QDLDL | `df48100` | Leave pinned unless a separately justified change is needed |

Use published, immutable dependency SHAs. For GRiD, `65fd051` is the immediate
reviewed integration candidate; if the GRiD agent publishes a newer release-ready commit
before integration, review its changes and evidence and record that exact target.
Do not follow a moving working-tree HEAD or assume GRiD main is newest.

Last remote MPCGPU CI: `30776707171` and `30776707105`, successful on
2026-08-03 at `c3c2df2`. That is historical evidence, not current readiness.
The committed receipt ended 2026-08-03 and exceeds its own 30-day age policy.

## Overall assessment

The architecture is appropriate and worth retaining: generated GRiD dynamics,
GLASS block primitives, and a cooperative grid-wide PCG implementation in
GBD-PCG. GLASS's single-block PCG is **not** a drop-in replacement for that
cooperative decomposition. MPCGPU remains a specialized iiwa CUDA/C++ solver;
it does not currently provide a supported Python/Julia binding layer. Its
pyproject configures tests rather than packaging a solver library.

The development branch is substantially modernized but **not merge-ready**.
The largest problems are unsafe public edges, non-hermetic validation,
unreliable rebuilds, inconsistent examples/docs, and repeated host setup work.
Compile-speed and runtime improvements remain hypotheses until measured.

## Findings and evidence

Line references refer to the audited source, not future edited positions.

### F1. Public GBD-PCG host wrapper is unsafe (merge blocker)

`GBD-PCG/include/interface.cuh:33-88` allocates `d_Pinv` without initializing it,
then invokes the solver and returns constant `1` instead of the actual iteration
count. A false `empty_pinv` merely prints a warning and continues. The kernel
always reads the preconditioner (`GBD-PCG/include/pcg.cuh:140` vicinity); it has
no empty-preconditioner argument, despite an extra entry in the launch argument
array. The CSR overload exits the process as an unimplemented stub.

The device wrapper (`interface.cuh:110-133`) specializes compile-time dimensions
but accepts unchecked runtime dimensions for launch/scratch sizing; it ignores
the advertised block/grid fields and does not invoke its occupancy guard.
`checkPcgOccupancy` itself uses device 0 instead of the current CUDA device.
MPCGPU's paper example checks occupancy, but not every public launch path does.

Fix the documented host path with an identity preconditioner and actual result,
or explicitly remove/deprecate unsupported entry points. Validate dimensions,
current-device cooperative capacity, and configuration before launch. Prefer
structured result/status and ownership-safe cleanup over process termination.
Do not claim arbitrary runtime dimensions or asynchronous execution.

### F2. Small-dimension memory safety is not covered (merge blocker)

Static scratch-layout calculation in `GBD-PCG/include/pcg.cuh:35-40,112-131`:
the buffers through the full three-slot `s_p` require
`6*s*s + 12*s + 2*max(s,N)` elements, while allocation uses
`max(6*s*s + 10*s + 2*max(s,N), 9*s*s)`.
For the shipped `s=2,N=3` demo, that is **54 required vs 50 allocated**.
The larger-dimension maximum masks this in the existing 6x8 and 14x32 gates.
This is a static out-of-bounds finding; no sanitizer reproduction was run here.

Additionally, `GBD-PCG/include/utils.cuh:21-26` reads the next neighbor for block
zero even when there is only one knot. Support N=1 correctly or reject it before
launch; document the supported minimum. Add layout assertions and bounded
memcheck cases for the smallest supported dimensions, plus non-power-of-two N.

### F3. Build outputs can be stale (merge blocker)

`Makefile:16-20` and `GBD-PCG/examples/Makefile:12-23` executable targets lack
source/header prerequisites and flag/configuration keys. Read-only reproduction:
`make -n examples ARCH=sm_86` reported **nothing to be done** against existing
executables. Changing architecture, horizon, precision, or dependencies must
not silently reuse an old binary. The nominal double demo target even compiles
`pcg_solve.cu` rather than `pcg_solve_dp.cu`.

Use dependency files plus configuration-specific build directories/stamps, one
shared flag definition, and explicit named demo/correctness/timing profiles.
No wholesale build-system rewrite is necessary merely to fix this.

### F4. Regeneration and tests depend on a mutable sibling (merge blocker)

`tools/regen_grid.py:42-45,73-87` chooses the sibling-project URDF if available,
otherwise a mesh-less local copy, then emits all algorithms plus collision and
contact code only to match that project’s generated bytes. `test/test_gates.py:71-79` requires that
sibling's generated header. Current headers already differ (`cmp` exit 1), as
expected with their different GRiD pins. This gate cannot currently pass and
does not demonstrate an MPCGPU numerical regression.

Own the model inputs and codegen recipe. Replace moving-sibling equality with
regenerate-and-compare against MPCGPU's committed output plus independent
dynamics/derivative/cost oracles. Keep cross-project model-hash/parity checks as
explicit integration checks, not a mandatory sibling checkout.

The generated header has 39,629 lines. Unused device functions may be eliminated
from the binary, but the assertion in the generator that carrying them "costs
nothing" does not establish parsing/compilation cost. First retain the needed
surface during migration; then test a minimal GRiD algorithm profile without
unused collision/contact/second-order families. If collision remains, own its
mesh assets, declare voxel dependencies, and reject fallback geometry. Keep model assets and generation self-contained.

### F5. Receipt and tests provide weaker guarantees than their labels (blocker)

- `test/run_gpu_proof.sh:40-47` and CI install an unbounded plugin version;
  policy lacks schema-3/signature/scope restrictions. Pin producer + verifier +
  policy together; fingerprint dependency gitlinks, recipe, config, and harness.
- Only **8 pytest items** are collected. `test_gbdpcg_gates` records a whole
  shell runner as one pass even when its dumped-system gate prints SKIP.
- `GBD-PCG/test/run_gates.sh:48-86` depends on ambient `/tmp/mpc_*.bin`, or writes
  a shared `/tmp/gates_bdmv_S.bin`. Test inputs and even coverage change with
  unrelated prior work. Use deterministic owned fixtures and per-run temp dirs.
- Bootstrap omits NumPy although the synthetic-strip gate requires it.
- Shell verdict parsers often check printed numbers without checking the
  executable's exit status (`tools/run_gates.sh`, GBD matvec/dumped gates).
- No CPU-only structural lane exists. Missing receipts cause CI to skip its
  enforcement. Final merge criteria must require a present, current receipt
  and a full expected collection, not merely a green workflow.

Expose individual GBD gates to pytest, require finite outputs, verify exit status,
and make every skip visible and justified. The 1e-1 dumped-system residual bar
is only a smoke threshold, not proof of high-accuracy convergence. Pair frozen
real systems with true-residual and independent reference checks.

### F6. Examples and input contracts need repair (merge blocker)

The top-level README launches legacy paper sweeps. The PCG example traverses
start/goal pairs although only 0_0, 0_1, and 0_2 have modern EE reference files;
missing files are printed/skipped rather than presented as a deliberate demo
selection. Public defaults also differ from the well-tested "fair" profile.
GBD legacy demos are documented as broken yet remain normal build targets.

`tools/validate_track.cu:32-47` checks only the EE trajectory length, not matching
state/control rows, widths, finiteness, or enough state data before host/device
copies. Add malformed/missing/ragged/nonfinite/short input tests before CUDA
allocation. Make the public quickstart one short, finite, deterministic example
per backend, not a paper-parameter sweep. Remove or clearly quarantine known
invalid historical data without destroying its provenance.

`include/mpcsim.cuh:342-347` records L2 position error but `:493-495` returns an
L1 final error. The validation result labels these together. Standardize metric
definitions and version changed protocols rather than mixing old/new rows.

### F7. Runtime overhead and duplication deserve a bounded second stage

Both SQP implementations allocate/free device scratch, create/destroy eight
streams and a cuBLAS handle on every solve (`include/pcg/sqp.cuh:82-154,464-492`;
`include/qdldl/sqp.cuh:92-200,408-443`). QDLDL also repeats symbolic setup and
allocates host workspaces; etree/factor return values are not checked.
PCG's file is 499 lines; QDLDL's is 448, with duplicated merit/line-search,
regularization, update, and bookkeeping control flow.

Introduce a small reusable workspace/context with clear lifetimes and backend
resources, test repeated solve/reset/destruction and errors, then extract truly
identical orchestration. Preserve mathematical differences and operation order.
Do not build a generic solver framework or replace the cooperative algorithm.
Check float/double/QDLDL ABI consistency; advertise only supported combinations.

`GBD-PCG/include/utils.cuh:131-139` performs both GLASS copy and a manual copy to
the same destination. Manual `gato_memcpy` also remains in QDLDL setup, contrary
to blanket "no hand-rolled in-block BLAS" claims. Consolidate these small helpers
with parity checks. Audit debug-only branches and stale comments, not just LOC.

Performance benefit is **unmeasured**. Internal SQP timers include setup but stop
before cleanup; therefore they are not complete public-call latency. Future
measurements must distinguish setup, steady-state solve, and end-to-end call.

### F8. Timing harness can accept stale output (must fix before timing)

`tools/time_persolve.sh:43-47` ignores the child exit status and then reads a fixed
previous-results path. It lacks isolated output directories, run provenance,
reference hashes, a shared lock, explicit quiet-window authorization, and safe
stop/resume rules. It also compiles inside the timing launcher. Its p90 comment
says nearest-rank but implementation uses rounded `(n-1)*0.9` indexing.
`tools/run_3way_iiwa.sh` hardcodes user directories, can proceed after failed
compilation, writes tracked references, and launches sibling-repo jobs.

Do not run these unchanged as the new evidence collector. Build beforehand,
fail closed on failed/nonfinite/incomplete results, freeze inputs, isolate every
repeat, and keep other-project work outside the MPCGPU-only default launcher.

### F9. Documentation, distribution boundary, and website

`docs/modernization.md:43-46` says tracking is not a valid gate and modernization
is correctness-complete; current test runners and later docs contradict this.
Create a short docs index/status page and label old analyses historical.
Document exact defaults vs benchmark profiles, precision, cooperative GPU
requirements, supported problem/model, tolerance semantics, allocation/stream
ownership, and dependency versions. Audit code comments against the current API.

No supported high-level-language wrapper was found; do not advertise one or
automatically add one. First make the C++ API safe. If bindings are wanted later,
wrap that single workspace API rather than duplicating solver state in Python.

At audit time no top-level LICENSE/COPYING/NOTICE was tracked in MPCGPU.
Resolved by the paper author: MIT covers MPCGPU code, documentation, website
and paper figures. LICENSE and NOTICE now record that decision.
The old GBD-PCG repo is not archived remotely. After the integrated main lands,
an authorized pointer-only README/archival decision may clarify its status.

## Ordered execution plan

### Wave 0 — baseline and reproducible safety gates (no timing)

- [ ] Recheck branches/status and pin published dependency targets. Preserve
  `3cdf45f` and current historical evidence.
- [ ] Add deterministic host tests for parsers, configuration/rebuild identity,
  gate failure propagation, scratch accounting, and receipt scope.
- [ ] Capture current numerical fixtures where existing paths are valid;
  record known failing contracts separately. Do not certify the current whole
  suite merely by suppressing its sibling-header mismatch or hidden GBD skip.
- [ ] Fix build dependency/configuration handling and deterministic gate inputs.
- [ ] Upgrade receipt producer/verifier/policy to 0.4.0/schema 3; add host CI.

Exit: trustworthy builds and test inventory; no ambient `/tmp`/sibling inputs;
negative test results propagate to failures; a clean install has declared deps.

### Wave 1 — pin migration and API/memory correctness (no timing)

- [ ] Switch owned .gitmodules URLs to HTTPS and verify anonymous recursive
  clone, with no inherited credential/SSH rewrite requirement.
- [ ] Integrate the selected published GRiD + GLASS pins; regenerate locally.
  Use explicit `glass::block::` where existing arithmetic/synchronization must
  remain fixed. Bare `glass::` now has measured dispatch for some cells; do not
  assume every existing call changes, or silently accept a new reduction order.
- [ ] Fix F1/F2 public wrapper, small-dimension scratch, N=1 handling, dimension
  guards, current-device occupancy, statuses, and QDLDL error/precision checks.
- [ ] Replace invalid GBD demos with validated SPD float/double examples (or
  narrow the advertised precision contract with explicit rejection).
- [ ] Gate FD/ID/minv/FK/derivatives and terminal-cost regression against owned
  fixtures plus independent oracles, then both fixed-pacing MPC backends.
- [ ] Add GBD known-solution SPD cases, nontrivial preconditioning, zero RHS,
  exact warm start, cap/breakdown behavior, invalid config, odd N, small blocks,
  and residual-replacement/true-exit modes. Distinguish eta from true residual.
- [ ] Run bounded memcheck, initcheck where applicable, and race/sync checks on
  representative small cases. Preserve uncompleted checks as incomplete.

Exit: numerical drift explained; supported public paths finite/correct; unsafe
inputs rejected; no hidden skips; one complete clean-source receipt.

### Wave 2 — usability, bloat, and runtime preparation (no timing)

- [ ] Repair quickstart/examples and trajectory validation; unify L2 reporting.
- [ ] Consolidate duplicated copy helpers and build flags, then introduce/test
  reusable solver workspace; extract shared orchestration in a separate,
  reviewable change. Test repeated use, separate instances, cleanup, and errors.
- [ ] Trial a minimal generated header, preserving adapter/math parity. Record
  artifact sizes and codegen inputs now; defer compile-speed comparisons.
- [ ] If a workspace or compile-profile change becomes a broad redesign,
  separate it explicitly from the merge candidate rather than weaken gates.
  Never remove needed features solely to shrink line counts.
- [ ] Build/run the actual documented quickstart in a fresh anonymous clone
  with self-contained inputs, no existing executables, and a fresh tooling venv.
- [ ] Refresh docs/README, API reference, development guide, citation metadata,
  status/limitations, and historical-result labels. Resolve license question.
- [ ] Prepare the website locally as described below, without deployment.
- [ ] Final source commit, full receipt, authorized development-branch push,
  green CPU and receipt CI; prepare immutable timing binaries/manifests.

Exit: documented examples work; supported API and implementation agree;
candidate ready for a timing window, not yet performance-certified.

### Wave 3 — explicit quiet window only

- [ ] Add a tested launcher with dry-run, shared `/tmp/a2rlab-timing.lock`,
  explicit permission environment flag, finite/exit/sample-count gates, unique
  outputs, per-repeat provenance, STOP boundaries, and fresh-output resume.
- [ ] Measure candidate PCG/QDLDL at N=32,64,128,256,512 (subject to current-device
  occupancy); at least three process repeats with fixed references/warm starts,
  interleaved order where appropriate. Record tracking quality and iteration/
  exit distributions alongside latency. Save all raw samples and hashes.
- [ ] Compare against a preserved pre-change development baseline on the SAME
  machine/problem/protocol. Separate pin/API fixes from workspace changes if
  attribution needs it; main's older model is not an apples-to-apples baseline.
- [ ] Optional separate compile leg: clean object-tree and incremental/no-op
  builds, wall time, peak memory and binary size; label cache conditions.
- [ ] Review regressions before changing defaults or publishing numbers.

Provisional coordinator budget: **30–45 minutes for MPCGPU runtime checks**, plus
**15–30 minutes if compile timing is requested**. These are reservations, not
measured duration estimates; recalibrate after preparation and release early.
GBD-PCG correctness is already covered; a separate broad microbenchmark sweep
is optional, not hidden in this budget. Other projects have separate reservations.
Sequence workloads; do not overlap them.

Coordinator handoff must give all six requested fields:

| Field | Required handoff content |
| --- | --- |
| Launcher | Proposed `tools/run_timing_handoff.sh`; **not implemented yet**, no runnable command promised |
| Working directory | `/home/plancher/Desktop/MPCGPU` (runner also resolves repo root) |
| Runtime | Provisional 30–45 min runtime; optional 15–30 min compile leg |
| Prerequisites | Explicit assigned slot, lock, matching source/receipt/CI/binaries, CUDA device checks, frozen references; no hidden installs/builds |
| Outputs | Unique `tmp/timing/<run-id>/` with logs, raw samples, manifests, metrics and verdicts; preserve partials |
| Stop/resume | STOP between independent legs/repeats; urgent terminate exact owned process tree, invalidate affected repeat; rerun unfinished work in a fresh directory, no SIGSTOP timing resume |

### Wave 4 — final main integration and release decision

- [ ] Review the complete main diff, including subtree history, stale files,
  model changes, API compatibility and upstream attribution.
- [ ] Integrate latest main only when authorized; validate the exact candidate
  again (including fresh-clone commands and receipt ancestry/fingerprint).
- [ ] Require no unexplained correctness/performance regression, no hidden
  skips, current signed receipt, green host CI, truthful docs/links, and working
  examples. Unsupported research configurations must be labeled explicitly.
- [ ] Merge/publish only on authorization. Switch docs links to main when the
  corresponding API actually exists there. No tag/package release implied.
- [ ] Afterward, optionally update the old GBD-PCG repository to point at the
  integrated implementation; archival is a separate authorized action.

## MPCGPU website brief

Build a lightweight static project page in the MPCGPU repository, suitable for
GitHub Pages, with no application framework or runtime dependency needed.
This is a draft recommendation, not deployment authorization or a final URL.

Suggested page order:

1. Title, one-sentence purpose, paper/code/documentation/quickstart links.
2. Pick-and-place visual: paper **Figure 3**, with a slot for the user's
   higher-resolution artwork or animation and a static accessible poster.
3. How it works: paper **Figure 2** (Schur assembly -> cooperative GBD-PCG ->
   parallel line search), with an optional compact Figure 1 overview.
4. Published results: selected Figures 4/5, with original hardware, workload,
   metric and attribution clearly stated. Do not restyle these as new timings.
5. Current software: GRiD/GLASS architecture, integrated GBD-PCG, current
   correctness/reproducibility status, supported iiwa/CUDA use, limitations.
6. Quickstart, citation, authors, acknowledgments, GRiD/GLASS links.

The linked paper (arXiv:2309.08079v3) was downloaded and pages 1,4,5 rendered and
visually inspected. Figures 1/2/3 are suitable initial explanatory assets;
prefer vector exports for diagrams and original-resolution images/video when
available. Use our figures under the project MIT license, retaining figure
numbers and experiment context. Extract assets
without redrawing experimental values. Validate desktop/mobile rendering,
contrast, alt text, asset sizes, links and reduced-motion behavior before launch.

Paper evidence is distinct from a new benchmark: Section V uses an RTX 4090,
i9-12900K, CUDA 12.1, and 100 ten-second five-goal trials. Current development
notes use a newer machine, corrected model, figure-eight references and changed
cost/regularization/warm-start choices. They are not a paper reproduction.
Figure 4 reports **linear-system** solve times, not whole MPC-call latency.
Fresh modernized-runtime measurements cannot alone refresh the whole paper.
Keep published plots as historical evidence; label any new plot by protocol,
commit and hardware. Avoid claiming the August N-sweep revalidates all paper
results. The higher-resolution figures/animation are optional follow-up inputs,
not blockers for preparing the page.

Source: https://arxiv.org/pdf/2309.08079

## Work actually performed for this audit

- Read local source, tests, build scripts, docs, receipt and existing handoff;
  compared upstream APIs/policies and published remote heads/CI metadata.
- `git diff --check`: clean at arrival. Pytest collection: 8 items (no tests
  executed). Make dry-run exposed stale architecture rebuild behavior.
- Current sibling-header comparison differs; recorded as an invalid cross-check,
  not a numerical failure. Static safety/contract findings above were not
  dynamically reproduced in this pass.
- Rendered/read the paper for the website brief. No website created/deployed.
- No CUDA build, GPU correctness run, timing, pin change, source fix, commit,
  push, merge, release, or sibling-repository mutation performed. Only this
  audit/plan document was added to MPCGPU.

Next action: discuss/approve the sequence, then implement Waves 0–2 while the
shared-box correctness/build allocation is available. Stop at timing-ready if
no exclusive window is assigned.
