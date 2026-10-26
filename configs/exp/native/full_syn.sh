# Shared by EXP-FULL-SYN.sh and EXP-FULL-SYN-FSA.sh: everything of EXP-B2 except the training images, which come
# from the synthetic release (validation and test images stay real, under RAW_ROOT).
source "$REPO/configs/exp/native/EXP-B2.sh"
ARGS+=(--train_img_dir "${SYNTH_ROOT:?SYNTH_ROOT must point to the synthetic release}")
