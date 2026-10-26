# FULL-SYN PDP (docs/project/full-syn-train-plan-2026-10-22.md, section 3): EXP-B2 trained on synthetic v2 and
# scored on real RPC. Set by autocheckout/full_syn.py: TASK_DIR (symlink folder: synthetic train_task_<t>.json,
# real val/test files), SYNTH_ROOT (images of the synthetic release) and FSA_HF (hf_model of the FSA run).
EXP=${EXP:-EXP-FULL-SYN}
source "$REPO/configs/exp/native/full_syn.sh"
# Task 1 starts from FSA instead of the COCO detector; F14 and I3 on (B2 has both off).
ARGS+=(--repo_name "${FSA_HF:?FSA_HF must point to the hf_model of the FSA run}"
    --pseudo_dedup_iou 0.5 --pseudo_gt_iou 0.5)
