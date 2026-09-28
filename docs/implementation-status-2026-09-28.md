# Modernization implementation status — September 28, 2026

Scope: MPCGPU plus in-tree GBD-PCG only. GATO and upstream dependency code were
not changed. No timing, clock/governor changes, main merge, release or deployment.

## Implemented

- Published pins: GRiD 65fd051198e4ccc6f609238bca4cfa780eb2b55a,
  GLASS 8ce68a29bceb30c7764c9391d517a182d061697d; QDLDL unchanged df48100.
  HTTPS URLs; exact dependency manifest; pytest-gpu-proof 0.4.0.
- Owned URDF/codegen recipe, no GATO or mesh dependency. Explicit minimal GRiD
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
  MIT owned-code license and third-party/paper NOTICE.
- Static website draft with attributed paper Figures 2/3/4; separate published
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
  hppfcl-to-coal deprecation warning, no skips. Clean-source receipt and
  isolated-clone quickstart verification are being finalized.

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

Finish clean-source receipt/fresh-clone checks, prepare immutable timing binaries
and dry-run the handoff. Then stop at timing-ready. Reserve 30–45 minutes for
runtime collection, optional separately prepared compile timing 15–30 minutes.
Only after analysis should new speed claims or website result panels be added.
