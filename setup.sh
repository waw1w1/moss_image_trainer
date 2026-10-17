#!/usr/bin/env bash
# Install the locked training environment and verify it; never start training.
set -euo pipefail

moss_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$moss_project_dir"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'This environment requires Linux x86_64.' >&2
    exit 1
fi
for moss_compiler in cc c++; do
    if ! command -v "$moss_compiler" >/dev/null 2>&1; then
        printf '%s\n' 'Missing C/C++ compiler. Ubuntu/Debian: sudo apt-get install -y build-essential' >&2
        exit 1
    fi
done
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    printf '%s\n' 'No accessible NVIDIA driver/GPU. Install the host driver or enable GPU access in the container first.' >&2
    exit 1
fi

moss_uv=""
if [[ -x "$moss_project_dir/.runtime/uv/uv" ]]; then
    moss_uv="$moss_project_dir/.runtime/uv/uv"
elif command -v uv >/dev/null 2>&1 && [[ "$(command -v uv)" != "$moss_project_dir/.venv/"* ]]; then
    moss_uv="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    moss_uv="$HOME/.local/bin/uv"
fi
if [[ -z "$moss_uv" ]]; then
    mkdir -p -- "$moss_project_dir/.runtime/uv"
    moss_installer="$(mktemp "$moss_project_dir/.runtime/uv/install.XXXXXX")"
    trap 'rm -f -- "$moss_installer"' EXIT
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 https://astral.sh/uv/0.12.4/install.sh --output "$moss_installer"
    elif command -v wget >/dev/null 2>&1; then
        wget -q https://astral.sh/uv/0.12.4/install.sh -O "$moss_installer"
    else
        printf '%s\n' 'Install curl or wget to bootstrap uv.' >&2
        exit 1
    fi
    UV_UNMANAGED_INSTALL="$moss_project_dir/.runtime/uv" sh "$moss_installer"
    moss_uv="$moss_project_dir/.runtime/uv/uv"
fi

# Ignore packages injected by an activated server environment; use this project only.
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$moss_project_dir/.venv"
"$moss_uv" sync --locked --managed-python --python 3.12
"$moss_uv" pip check --python "$moss_project_dir/.venv/bin/python"
"$moss_project_dir/.venv/bin/moss-trainer" doctor
printf '%s\n' 'Environment ready. Configure model/data paths, cache features, then explicitly run train when ready.'
