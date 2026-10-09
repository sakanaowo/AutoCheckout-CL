# E3: PDP with all fixes F1-F13 on the FSA base (I1), no other improvement (I2-I5) and no augmentation.
source "$REPO/configs/exp/common.sh"
EXP=E3
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${FSA_ARGS[@]}")
