# Modernization implementation status — September 28, 2026

Scope: MPCGPU plus in-tree GBD-PCG only. Other projects and upstream dependency
code were not changed. No timing, clock/governor changes, main merge, release or deployment.

## Implemented

- Published pins: GRiD 65fd051198e4ccc6f609238bca4cfa780eb2b55a,
  GLASS 8ce68a29bceb30c7764c9391d517a182d061697d; QDLDL unchanged df48100.
  HTTPS URLs; exact dependency manifest; pytest-gpu-proof 0.4.0.
- Self-contained URDF/codegen recipe with mesh-independent dynamics. Minimal GRiD
  algorithms, shared GLASS instead of embedded copies. Header 39,629 -> 13,410
  lines. Compile-time benefit is not measured. Owned calls pin glass::block::.
- Safe identity-preconditioned host GBD wrapper; actual iteration result;
  device result/limit status; dimension/configuration/current-device occupancy
  guards; unsupported CSR/host Pinv rejected. Fixed small-dimension scratch,
  single-knot halo, zero-tolerance exact-zero convergence and header ODR issues.
- Deterministic SPD float/double examples, explicit input files, no ambient
  dumped-system skip. Fresh Schur capture uses an owned temporary directory.
- Content/configuration-aware sequential builds and explicit correctness/timing
  profiles. Strict finite rectangular reference parsing before allocation;
  consistent L2 mean/max/final tracking metric.
- Per-instance reusable SQP workspace owns allocations, streams and cuBLAS;
  QDLDL symbolic structure cached. Numerical failures checked. Independent
  workspace/dimension/backend checks and fresh/reused raw-state hash parity.
- Removed obsolete commented dynamics implementations, duplicate copies, unsafe
  legacy demos and sibling-dependent runner. Historical analyses retained with
  warning banners; shared-backend orchestration extraction deferred explicitly.
- Receipt policy: clean source, schema >=3, restricted signer, no carried
  evidence, exact node manifest, actual dependency gitlinks and website bound.
  CPU CI exercises host contracts; missing/partial/stale receipts fail.
- Refreshed root/GBD READMEs, API/development/docs index, standing agent guide,
  MIT license for code, documentation, website and our paper figures; dependency notices.
- Static website draft with our paper Figures 2/3/4; separate published
  evidence/current-software status; desktop/mobile preview checked. No framework,
  remote fonts or scripts. Higher-resolution originals/video remain optional.
- Timing harness with prepare-only builds, dry-run, explicit quiet-window flag,
  advisory lock, frozen identities, receipt preflight, unique outputs, bounded
  children, fail-closed metrics and STOP/fresh-output resume. Not executed.

## Evidence and remaining gates

Intermediate evidence (not the final signed receipt):

- Original four correctness gates passed before and after dependency migration.
- Expanded 39 GPU tests passed; workspace-expanded 62-test suite passed.
- Independent Pinocchio 3.8 oracle plus two residual-mode variants: 3 passed.
  Oracle checks FD, FD derivatives, Minv via du, inverse-dynamics bias, named
  EE position and position Jacobian on five deterministic states.
- Fresh real-Schur capture/replay gate passed; no /tmp fixture dependency.
- Small GBD sanitizer checks: 2x3 memcheck host and racecheck device, plus 1x1
  device memcheck, zero reported errors/hazards. Broader exhaustive sanitizer
  coverage is not claimed.
- Full inventory: 76 tests (51 GPU, 25 host/harness), all passed. One upstream
  hppfcl-to-coal deprecation warning, no skips. Core source commit 0e2550f,
  final Make override regression fix f61d1ec; signed schema-3 receipt c5cff26
  (attesting f61d1ec) verified against public signing keys,
  exact source/dependency fingerprint, ancestry and complete node manifest.
- Isolated candidate clone at /tmp/mpcgpu-clean-Emt4eh/repo: recursive public
  HTTPS submodules fetched with global/system Git configuration and credential
  helper disabled; fresh venv installation succeeded; 25 host checks passed.
  Both documented MPC demos and float/double GBD examples built and passed.
  The clone used self-contained model inputs and freshly built executables.
  PCG mean/max/final L2: 0.028717/0.063881/0.008735; QDLDL:
  0.029387/0.064911/0.010898, both 1202 offsets. These are not timing results.
- Current workspace memcheck and model-oracle initcheck: zero errors.
  Source build warnings remain for unused correctness-profile timer variables
  and a feature-gated goal pointer; no warnings were suppressed to pass gates.
- Additional timer-disabled correctness passes completed both backends at
  N=32/128/256/512 (N=64 was already in the signed suite). Every point returned
  1202 offsets and finite tracking. Supplementary log:
  /tmp/mpcgpu-horizon-correctness-20260928.log.

| Horizon | PCG mean L2 EE error (m) | QDLDL mean L2 EE error (m) |
| --- | ---: | ---: |
| 32 | 0.027370 | 0.027527 |
| 64 | 0.028717 | 0.029387 |
| 128 | 0.030148 | 0.030208 |
| 256 | 0.030290 | 0.030329 |
| 512 | 0.030330 | 0.030322 |

These are figure-eight correctness observations, not timing or paper-reproduction
results. Pinocchio is test-only; it does not link into either native solver.

Local evidence logs are /tmp/mpcgpu-*-20260928.log. The signed gpu-proof.json,
once refreshed, is the portable clean-source record; temporary logs are not.

## Deliberate limits, not hidden completion claims

- No general robot/contact/locomotion/batch API and no Python/Julia solver
  binding. Float MPC/QDLDL; float/double GBD. The public native API is not yet a
  stable asynchronous library ABI; internal CUDA helpers can still terminate.
- GBD requires finite SPD inputs. It reports iteration cap versus configured
  convergence, not a structured indefinite-system breakdown or certified true
  residual. The docs explicitly distinguish eta from Euclidean residual.
- Shared PCG/QDLDL orchestration remains partially duplicated. The workspace
  addresses repeated resource setup without introducing a generic framework.
  Further extraction is a separately reviewable improvement, not a safety fix.
- Compile-speed measurements and old-development-baseline runtime reproduction
  are separate pending legs. Prepared fresh/reuse variants isolate workspace
  impact on the current candidate; they do not reconstruct old main.
- Main merge, remote CI acceptance, website publication and archival of the old
  GBD-PCG remote require their respective remaining validation/authorization.

## Next checkpoint

**Runtime preparation complete; receipt refresh pending after website edits.**
The 76-test clean-source receipt and fresh-clone checks above record the preceding
implementation checkpoint. Subsequent website/licensing/copy edits change the
fingerprint; the old receipt does not attest them. Only host checks run while
the box is busy. Refresh and verify the full receipt before launching timing.
All 20 immutable timing binaries are prepared under
tmp/timing-prepared/candidate-20260928; dry-run validated the 60-repeat plan.
The runtime batch was not executed. See [coordinator handoff](HANDOFF_codex_2026-09-28.md).
Reserve 30–45 minutes for runtime collection; optional compile timing requires
a separate prepared leg and 15–30 minutes. Only after analysis should new speed
claims or website result panels be added. Push/remote CI, merge and publication
are still pending their validation/authorization. See the
[ICRA replication plan](icra-replication.md) for the paper-example milestone after
timing. No GPU or timing work is launched for the copy update.
