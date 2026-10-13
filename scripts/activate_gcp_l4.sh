# Source from any directory: source /path/to/AutoCheckout-CL/scripts/activate_gcp_l4.sh
_autocheckout_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_autocheckout_root/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export USE_TF=0
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$_autocheckout_root/data/cache/torch_extensions_l4"
export MAX_JOBS=4
unset _autocheckout_root
