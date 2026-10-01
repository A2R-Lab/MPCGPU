#!/usr/bin/env bash
# Prepare the PCG speedup attribution (docs/speedup-attribution.md). Builds only; measures nothing.
# Sources come from this repository's own history and GLASS submodule:
#   paper   GBD-PCG 75d6214 + GLASS 90a7a21   (the ICRA 2024 kernel)
#   glass   GBD-PCG bc60729 + GLASS 066d32d   (the same kernel moved onto GLASS; only change)
#   current GBD-PCG/include + GLASS            (today's kernel)
# plus the paper-era MPC code (main = 0efde8c) and one captured ICRA Schur system per horizon.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/tmp/attribution
ARCH=${ARCH:-sm_120}
HORIZONS=${HORIZONS:-"32 64 128 256 512"}
NVCC=(systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 --same-dir nvcc)
cd "$ROOT"
mkdir -p "$OUT/src" "$OUT/bin" "$OUT/dumps"

extract() {  # name gbd_commit glass_commit
    local dir=$OUT/src/$1
    rm -rf "$dir" && mkdir -p "$dir/gbd" "$dir/GLASS"
    git archive "$2" | tar -x -C "$dir/gbd"
    git -C GLASS archive "$3" | tar -x -C "$dir/GLASS"
}
extract paper 75d6214 90a7a21
extract glass bc60729 066d32d

# Kernel benchmark binaries.
for n in $HORIZONS; do
    for version in paper glass current; do
        if [[ $version == current ]]; then inc=(-IGBD-PCG/include -IGLASS); rel=1
        else inc=(-I"$OUT/src/$version/gbd/include" -I"$OUT/src/$version/GLASS"); rel=0; fi
        "${NVCC[@]}" -std=c++17 -O3 -arch="$ARCH" "${inc[@]}" -DSTATE_SIZE=14 -DKNOT_POINTS="$n" \
            -DHAS_REL_TOL=$rel tools/attribution/pcg_kernel_bench.cu -o "$OUT/bin/bench-$version-$n.exe"
    done
done

# One captured Schur system per horizon from the maintained ICRA task (correctness build, solve 3000,
# well into the circuit after the 2000 warm-up solves).
for n in $HORIZONS; do
    exe=$OUT/bin/dump-icra-pcg-$n.exe
    .venv/bin/python tools/build.py icra-pcg --knots "$n" --output "$exe" \
        --extra="-DDUMP_KKT -DDUMP_KKT_AT_SOLVE=3000" > /dev/null
    mkdir -p "$OUT/dumps/N$n"
    MPCGPU_DUMP_DIR=$OUT/dumps/N$n LD_LIBRARY_PATH=qdldl/build/out "$exe" --out "$OUT/dumps/N$n/run" > "$OUT/dumps/N$n/run.log"
    test -s "$OUT/dumps/N$n/mpc_S.bin"
done

# Paper-era MPC code (main) with its own dependencies, timing configuration exactly as published:
# TIME_LINSYS=1, 2000 us wall-clock SQP budget, 500 Hz, its five-tolerance PCG sweep.
PAPER=$OUT/paper-code
rm -rf "$PAPER" && mkdir -p "$PAPER"
git archive 0efde8c | tar -x -C "$PAPER"
rm -rf "$PAPER/GBD-PCG" "$PAPER/GLASS"
cp -r "$OUT/src/paper/gbd" "$PAPER/GBD-PCG"
cp -r "$OUT/src/paper/GLASS" "$PAPER/GLASS"
for n in $HORIZONS; do
    for backend in pcg qdldl; do
        flags=(); [[ $backend == qdldl ]] && flags=(-DLINSYS_SOLVE=0)
        (cd "$PAPER" && "${NVCC[@]}" -arch="$ARCH" --compiler-options -Wall -O3 -Iinclude -Iinclude/common \
            -IGLASS -IGBD-PCG/include -I"$ROOT/qdldl/include" -L"$ROOT/qdldl/build/out" -lqdldl -lcublas \
            -DKNOT_POINTS="$n" -DSAVE_DATA=1 "${flags[@]}" examples/track_iiwa_$backend.cu -o "$OUT/bin/paper-$backend-$n.exe")
    done
done
sha256sum "$OUT"/bin/*.exe > "$OUT/bin/SHA256SUMS"
echo "prepared $(ls "$OUT"/bin/*.exe | wc -l) binaries and $(ls -d "$OUT"/dumps/N* | wc -l) captured systems in $OUT"
