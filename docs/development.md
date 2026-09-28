# Development and validation

Initialize recursive HTTPS submodules and install requirements-dev.txt into an
owned venv. Exact reviewed gitlinks are recorded in tools/dependencies.json.
Never import models, Python environments or generated files from a sibling repo.

## Build and generation

`tools/build.py` serializes compiler invocations and hashes the compiler,
configuration, source, relevant headers, dependency commits and QDLDL library.
It checks binary hashes before cache reuse. Use `make examples` or the builder;
do not reuse hand-compiled executables as evidence. QDLDL is configured float,
32-bit indices by `make build_qdldl`.

`make regen` uses tools/iiwa14.urdf and the pinned GRiD. It emits only dynamics,
derivatives and EE kinematics needed here, and references top-level GLASS instead
of embedding a second implementation. `make check-codegen` verifies bytes in
an isolated temporary directory. Do not edit generated grid.cuh by hand.

On the shared lab box:

```bash
systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir \
  .venv/bin/python -m pytest test/ -q
```

One nvcc process at a time; no timing or governor changes. CUDA sanitizer runs
must also be bounded correctness-only runs.

## Signed receipt

The suite has individual pytest nodes, no ambient /tmp fixtures and no expected
skips. A checked-in node manifest prevents a subset receipt from passing CI.
Producer/verifier are both pytest-gpu-proof 0.4.0, schema >=3, clean-source,
restricted signer, no carried evidence, exact fingerprint scope.

1. Run/review correctness; commit source and dependency gitlinks.
2. Run `test/run_gpu_proof.sh` in a capped scope on the GPU host. It refuses
   selection arguments and dirty trees; it does not install packages.
3. Verify with `.venv/bin/gpu-proof verify --receipt gpu-proof.json --repo . --policy
   test/gpu-proof-policy.yaml --expected-skips test/expected_skips.txt --require-gpu`.
4. Commit the generated receipt. Push only when authorized; check host and
   receipt CI. Do not alter signatures or claim dirty local passes are receipts.

When adding/removing tests, review and regenerate test/expected_tests.txt from
pytest collection. Receipt attestation is a signed keyholder statement, not
cryptographic proof that a GPU executed arbitrary test code.

## Merge gate

Main integration is separate: exact candidate review, fresh-clone quickstart,
complete receipt/green CI, no unexplained correctness/performance regression,
accurate docs and final authorization. Website publishing and archival of the
old GBD-PCG remote are separate actions. No current timing claim is justified
until the exclusive-window collection and analysis are complete.
