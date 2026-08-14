#!/usr/bin/env bash
set -euo pipefail

python_bin=${ASTRAI_TRAIN_PYTHON:-python3}
wheelhouse=${ASTRAI_KERNEL_WHEELHOUSE:-}
index_args=()
if [[ -n "$wheelhouse" && -d "$wheelhouse" ]]; then
    index_args=(--find-links "$wheelhouse")
fi

if "$python_bin" -m pip --version >/dev/null 2>&1; then
    pip_cmd=("$python_bin" -m pip)
elif command -v uv >/dev/null 2>&1; then
    pip_cmd=(uv pip --python "$python_bin")
elif command -v pip3 >/dev/null 2>&1 && pip3 --help | grep -q -- '--python'; then
    # Some lean training environments are created without pip but can still
    # be targeted safely by a recent system pip.
    pip_cmd=(pip3 --python "$python_bin")
else
    echo "No pip/uv installer can target $python_bin" >&2
    exit 1
fi

"${pip_cmd[@]}" install "${index_args[@]}" --upgrade 'setuptools<82' wheel ninja packaging
"${pip_cmd[@]}" install "${index_args[@]}" --upgrade 'liger-kernel==0.8.1'
# Build TE's PyTorch extension against the environment's exact Torch/CUDA
# instead of letting an isolated build environment resolve a second Torch.
"${pip_cmd[@]}" install "${index_args[@]}" --upgrade 'transformer-engine==2.16.0'
"${pip_cmd[@]}" install "${index_args[@]}" --upgrade --no-build-isolation --no-deps \
    'transformer-engine-torch==2.16.0'

"$python_bin" - <<'PY'
import importlib.metadata

import liger_kernel
import transformer_engine.pytorch as te

print("liger-kernel", importlib.metadata.version("liger-kernel"))
print("transformer-engine", importlib.metadata.version("transformer-engine"))
print("TE DotProductAttention", te.DotProductAttention)
PY
