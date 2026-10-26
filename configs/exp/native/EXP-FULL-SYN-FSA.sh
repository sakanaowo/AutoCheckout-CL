# FULL-SYN FSA (plan section 3, D21): full fine-tuning of the EXP-B2 model on synthetic task 1, without prompts or
# pseudo-labels (FINETUNE_ARGS of configs/exp/common.sh). The epoch with the best val_task_1 AP50 is exported as
# task_1/hf_model, the start of PDP task 1 (EXP-FULL-SYN.sh).
EXP=${EXP:-EXP-FULL-SYN-FSA}
source "$REPO/configs/exp/native/full_syn.sh"
ARGS+=(--use_prompts 0 --local_query 0 --pseudo none --freeze "" --optim_groups detr --lr_backbone_names backbone
    --save_hf 1 --select_best_epoch 1)
