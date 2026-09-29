# One shared builder owns compiler flags and content/configuration invalidation.
.NOTPARALLEL:
PYTHON ?= .venv/bin/python
NVCC ?= nvcc
ARCH ?= sm_120
KNOT_POINTS ?= 64
PROFILE ?= correctness
EXTRA_FLAGS ?=
BUILD = $(PYTHON) tools/build.py --nvcc "$(NVCC)" --arch "$(ARCH)" --knots $(KNOT_POINTS) --profile $(PROFILE) --extra="$(EXTRA_FLAGS)"

.PHONY: examples icra test_fd_parity gen_ref submodules regen check-codegen build_qdldl test FORCE clean
examples: examples/pcg.exe examples/qdldl.exe
examples/pcg.exe: FORCE
	$(BUILD) pcg --output $@
examples/qdldl.exe: FORCE
	$(BUILD) qdldl --output $@
# ICRA 2024 pick-and-place circuit: make icra BACKEND=pcg|qdldl KNOT_POINTS=32..512
BACKEND ?= pcg
ICRA_EXE = build/$(PROFILE)/icra-$(BACKEND)-14x$(KNOT_POINTS)-float.exe
ICRA_OUT = tmp/icra/$(BACKEND)-N$(KNOT_POINTS)
icra:
	@test "$(PROFILE)" = correctness || { echo "make icra runs correctness builds only; timing uses tools/timing.py"; exit 1; }
	$(BUILD) icra-$(BACKEND) --output $(ICRA_EXE)
	LD_LIBRARY_PATH=qdldl/build/out $(ICRA_EXE) --out $(ICRA_OUT)
	$(PYTHON) tools/icra_report.py $(ICRA_OUT)
test_fd_parity:
	$(BUILD) fd --output examples/test_fd_parity.exe
gen_ref:
	$(BUILD) reference --output tools/gen_reference.exe
submodules:
	git submodule update --init --recursive
regen:
	$(PYTHON) tools/regen_grid.py
check-codegen:
	$(PYTHON) tools/regen_grid.py --check
build_qdldl:
	cmake -S qdldl -B qdldl/build -DQDLDL_FLOAT=true -DQDLDL_LONG=false
	cmake --build qdldl/build --parallel 1
test:
	$(PYTHON) -m pytest -q
clean:
	$(PYTHON) -c 'from pathlib import Path; [p.unlink() for d in ("examples", "tools", "GBD-PCG/examples") for p in Path(d).glob("*.exe")]; [p.unlink() for d in ("examples", "tools", "GBD-PCG/examples") for p in Path(d).glob("*.build.json")]'
