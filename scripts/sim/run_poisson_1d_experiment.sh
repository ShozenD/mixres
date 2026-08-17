#!/usr/bin/env bash
# Orchestrates the 1-D Poisson pclm experiment end-to-end:
#   generate data (Python) -> fit ungroup::pclm() + compute MSE (R) -> plot (Python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

CONFIG="$REPO_ROOT/config/poisson_1d.yaml"
SEED=""
BIN_WIDTH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --bin-width) BIN_WIDTH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Resolve effective seed/bin-width once so every stage uses identical values,
# falling back to the config file's defaults when not overridden on the CLI.
if [[ -z "$SEED" ]]; then
  SEED="$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['seed'])")"
fi
if [[ -z "$BIN_WIDTH" ]]; then
  BIN_WIDTH="$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['bin_width'])")"
fi

echo "Running poisson_1d experiment: config=$CONFIG seed=$SEED bin_width=$BIN_WIDTH"

cd "$REPO_ROOT"

"$PYTHON" scripts/sim/generate_poisson_1d.py --config "$CONFIG" --seed "$SEED" --bin-width "$BIN_WIDTH"
Rscript scripts/sim/poisson_1d_ungroup.R "$SEED" "$BIN_WIDTH"
"$PYTHON" scripts/sim/plot_poisson_1d_ungroup.py --seed "$SEED" --bin-width "$BIN_WIDTH"
