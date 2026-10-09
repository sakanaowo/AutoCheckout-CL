EXP=${EXP:-EXP-B2}
BACKBONE=convnextv2_base.fcmae_ft_in22k_in1k
source "$REPO/configs/exp/native/common.sh"
BACKBONE_WEIGHTS=${BACKBONE_WEIGHTS:-$REPO/runs/pretrained/hf-cache/models--timm--convnextv2_base.fcmae_ft_in22k_in1k/snapshots/9e250ed8f88b436472d2b24af26f82a8aa8c719d/model.safetensors}
ARGS+=(--backbone_pretrained_file "$BACKBONE_WEIGHTS")
