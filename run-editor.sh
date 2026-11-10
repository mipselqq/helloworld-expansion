#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$project_dir/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    printf '%s\n' 'Project environment not found. Follow the setup instructions in README.md.' >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_dir"
exec "$python" -m artemis.gui --config "$project_dir/config.toml" "$@"
