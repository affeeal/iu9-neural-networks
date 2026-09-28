#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_jobs=${BUILD_JOBS:-2}
if [[ ! $build_jobs =~ ^[1-9][0-9]*$ ]]; then
  echo "BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
temporary_root=""
if [[ -z ${BUILD_ROOT:-} ]]; then
  temporary_root=$(mktemp -d -t iu9-neural-check.XXXXXXXX)
  build_root=$temporary_root
else
  mkdir -p -- "$BUILD_ROOT"
  build_root=$(cd -- "$BUILD_ROOT" && pwd)
fi
cleanup() {
  if [[ -n $temporary_root ]]; then
    rm -rf -- "$temporary_root"
  fi
}
trap cleanup EXIT

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1

shellcheck "$repo_root/scripts/check.sh"
cmake -S "$repo_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
  -DBUILD_TESTING=ON -DNN_ENABLE_PLOTS=OFF -DNN_SANITIZE="${SANITIZE:-OFF}"
cmake --build "$build_root" --parallel "$build_jobs"
ctest --test-dir "$build_root" --output-on-failure
python3 -m unittest discover -s "$repo_root/tests" -p 'test_*.py' -v
python3 "$repo_root/hw5/sample/hw5.py" --help
python3 -m json.tool "$repo_root/hw5/report/original-experiments.ipynb" > /dev/null
