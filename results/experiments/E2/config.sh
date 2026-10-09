# E2 (MD-DETR-like): private pool only, pseudo-labels by a fixed threshold 0.65 on the top-5 queries;
# on the FSA base like E1, E3, E4.
source "$REPO/configs/exp/common.sh"
EXP=E2
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${FSA_ARGS[@]}" --use_shared 0 --pseudo threshold --pseudo_thresh_high 0.65
    --pseudo_topk 5)
