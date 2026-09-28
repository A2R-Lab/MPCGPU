# MPCGPU / GBD-PCG handoff — September 28, 2026

## Boundary

GBD-PCG remains in-tree and shares top-level GLASS. Model inputs, code generation
and validation are self-contained. MPCGPU code, documentation, website and paper
figures are MIT licensed. Dependency notices remain intact.
Shared GPU correctness and capped sequential builds were previously allowed;
**no timing or clock/governor changes were performed**.

Read CLAUDE.md and docs/implementation-status-2026-09-28.md. Preserve all
unrelated agents' work. Pushes require explicit clearance; main merge, website
deployment and old-GBD remote archival remain separate decisions.

## Verified candidate

- Branch: modernize-grid-glass. Core source 0e2550f, final build fix f61d1ec;
  previous verified receipt c5cff26 (attests f61d1ec). Website/copy edits after
  that checkpoint require a receipt refresh before timing or push.
- GRiD 65fd051, GLASS 8ce68a2, QDLDL df48100; proof tooling 0.4.0.
- Full clean-source suite: 76 passed, zero skipped; signed schema-3 receipt
  verified. The Pinocchio dependency emits one upstream deprecation warning.
- Anonymous recursive dependency fetch, fresh venv, 25 host checks, both MPC
  demos and float/double GBD demos passed in /tmp/mpcgpu-clean-Emt4eh/repo.
- Workspace fresh/reuse bit parity, independent dynamics/derivatives oracle,
  deterministic real-Schur replay, small-dimension API and sanitizer gates pass.
- Website draft: website/index.html; preview instructions in website/README.md.
  Desktop/mobile reviewed; local preview server stopped. Not deployed.
- All 20 timing binaries prepared; dry-run validated all 60 planned repeats.
  Additional timer-disabled correctness passes succeeded for both backends at
  N=32/128/256/512. N=64 is covered by the full signed suite.
- Test host: RTX 5090, Core Ultra 9 285K, CUDA 13.2.86, driver 615.71.09.
  Do not compare its raw latency directly with the paper's 4090/12900K setup.
- At handoff: no owned build, correctness or timing worker remains. No pushes,
  main changes, deployment or timing occurred; remote CI is not yet refreshed.

The two remote development names historically point to the same implementation;
they are not different solver designs. Do not force-push either. Recheck remote
heads and authorization before publishing this local candidate and receipt.

## Runtime timing batch for the coordinator

| Required field | Value |
| --- | --- |
| Launcher | `MPCGPU_QUIET_WINDOW=1 .venv/bin/python tools/timing.py run tmp/timing-prepared/candidate-20260928/plan.json tmp/timing/mpcgpu-next-window` |
| Working directory | `/home/plancher/Desktop/MPCGPU` |
| Estimated reservation | 30–45 minutes, release early; provisional, not measured. |
| Prerequisites | Plan.json exists and dry-run succeeds; clean tree; current verified receipt; exact matching binaries/inputs; explicit exclusive slot; no competing CPU/GPU work; advisory /tmp/a2rlab-timing.lock available. |
| Outputs | Unique run directory with provenance, raw samples/stdout and one success verdict per independent repeat. Prepared binary/command manifests remain under tmp/timing-prepared/candidate-20260928. |
| Stop/resume | Touch <run>/STOP for the next boundary; urgent Ctrl-C then confirm the exact child exited. Use a new output directory and --resume <prior-run>; no SIGSTOP/SIGCONT measurement resume. |

Before executing, use:

```bash
.venv/bin/python tools/timing.py run tmp/timing-prepared/candidate-20260928/plan.json \
  tmp/timing/preflight-only --dry-run
```

This does not acquire a timing slot or launch a GPU workload. Record CPU model,
current governors, other workers and toolchain at the assigned window too;
the launcher records NVIDIA state and commit/receipt hashes, and preparation
stores compiler commands. A lock alone cannot establish a quiet machine.
Choose a different unique output directory if mpcgpu-next-window already exists.

Batch: horizons 32/64/128/256/512 × PCG/QDLDL × reused/fresh workspace × three
process repeats, alternating workload order. Fresh is a same-source workspace
baseline, not old main. Internal SQP samples are not complete public-call
latency; full-process wall time includes simulation and output. Quality must
be reviewed per horizon. No paper figure is refreshed by this batch alone.

Optional compile-speed measurements need a separately prepared protocol/launcher
and another 15–30-minute reservation. They are **not** included in this launcher.
Old-development-baseline reproduction is also separate from fresh/reuse A/B.

## Copy update and next steps

The website uses direct product/research statements, a permanently visible BibTeX
block, and MIT licensing for the entire MPCGPU project, including our paper
figures. Public copy presents MPCGPU as the original ICRA 2024 work.
Only host checks run for these edits; the busy GPU is left untouched.
All 25 host/harness checks pass, including always-visible citation and website
copy assertions. Timing dry-run still validates all 60 repeats. The browser
preview tool rejected local-file navigation, so this revision has static
HTML/CSS checks rather than a new rendered visual sign-off.

Website and host-test changes are fingerprinted: **the previous receipt no longer
attests the current tree**. When GPU correctness access is available, commit the
changes, run the full receipt workflow, verify and commit the receipt, then use
the timing preflight. Do not bypass this check or run timing on the shared box.
The prepared runtime code and binaries are unchanged; dry-run checks their hashes.
After timing, follow [ICRA example replication](icra-replication.md).

## Arrival checks

```bash
git status --short
git log -4 --oneline
.venv/bin/gpu-proof verify --receipt gpu-proof.json --repo . --policy test/gpu-proof-policy.yaml --expected-skips test/expected_skips.txt --require-gpu
.venv/bin/python tools/timing.py run tmp/timing-prepared/candidate-20260928/plan.json tmp/timing/check-only --dry-run
```

Local evidence: /tmp/mpcgpu-{final-signed-receipt,clean-clone,clean-install,clean-host,
clean-quickstart,workspace-memcheck,model-initcheck,timing-prepare}-20260928.log.
Additional sweep: /tmp/mpcgpu-horizon-correctness-20260928.log.
The isolated clone is disposable test output, not another maintained checkout.
No main/release/deployment action is implied by this handoff.
