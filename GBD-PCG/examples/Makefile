.NOTPARALLEL:
PYTHON ?= ../../.venv/bin/python
ARCH ?= sm_120
STATE_SIZE ?= 14
KNOT_POINTS ?= 32
BUILD = $(PYTHON) ../../tools/build.py --arch $(ARCH) --states $(STATE_SIZE) --knots $(KNOT_POINTS)
.PHONY: examples test FORCE
examples: pcg.exe pcg_dp.exe test_spd.exe
pcg.exe: FORCE
	$(BUILD) gbd-api --dtype float --output $@
pcg_dp.exe: FORCE
	$(BUILD) gbd-api --dtype double --output $@
test_spd.exe: FORCE
	$(BUILD) gbd-spd --output $@
test: examples
	./pcg.exe host
	./pcg_dp.exe host
	./test_spd.exe
