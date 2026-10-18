#!/usr/bin/env bash
# test_usm.sh [WORKDIR] [THREADS] [TRIALS] [REPS] [E]: build test_usm.cpp and run it on a fresh random fixture (CPU only,
# no GPU, no weights). Linux: g++ with the host extension's flags (-march=znver3). macOS (Apple silicon): clang x86_64
# under Rosetta (correctness only; the speed line is meaningless there).
# E (default 12) experts x 2 copies x 9.47 MB of RAM; use E >= 48 for speed numbers that come from DRAM, not L3.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$(mktemp -d)}; THREADS=${2:-4}; TRIALS=${3:-24}; REPS=${4:-30}; E=${5:-12}
mkdir -p "$W"
python3 "$HERE/test_usm_fixture.py" "$W/fx" "$E" 1
if [ "$(uname)" = Darwin ]; then
  clang++ -arch x86_64 -O3 -mavx2 -mfma -mf16c -std=c++17 -I"$HERE" "$HERE/test_usm.cpp" -o "$W/test_usm" -lpthread
  RUN=(arch -x86_64 "$W/test_usm")
else
  g++ -O3 -mavx2 -mfma -mf16c -march=znver3 -std=c++17 -I"$HERE" "$HERE/test_usm.cpp" -o "$W/test_usm" -lpthread
  RUN=("$W/test_usm")
fi
"${RUN[@]}" "$W/fx" "$THREADS" "$TRIALS" "$REPS"
