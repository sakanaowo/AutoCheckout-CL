# Native real-only release, matching notebooks 02/03. Legacy experiments live one level above.
RAW_ROOT=${RAW_ROOT:-${AUTOCHECKOUT_DATA_ROOT:-$REPO/data/archive}}
RELEASE_ROOT=${RELEASE_ROOT:-${AUTOCHECKOUT_RELEASE_ROOT:-$REPO/data/processed/rpc_100-4x25_seed0_native_v1}}
TASK_CFG=${TASK_CFG:-$RELEASE_ROOT/task_config.json}
TASK_DIR=${TASK_DIR:-$RELEASE_ROOT/tasks/real}
DETECTOR_DIR=${DETECTOR_DIR:-$REPO/runs/pretrained/hf-cache/models--SenseTime--deformable-detr/snapshots/83ecd26945199939cb82806f988debdb71e6f43e}
RESOLUTION=${RESOLUTION:-640}
N_TASKS=${N_TASKS:-1}
# Smoke stays training-only; enable S6 evaluation with EVALUATE=1 for pilot/full runs.
EVAL_MODE=calibrated
SKIP_EVAL=1
if [[ ${EVALUATE:-0} == 1 ]]; then
    SKIP_EVAL=
fi
COUNT_NMS_GRID=${COUNT_NMS_GRID:-0.45}
ARGS=(
    --task_config "$TASK_CFG" --n_classes 225 --task_ann_dir "$TASK_DIR"
    --train_img_dir "$RAW_ROOT" --test_img_dir "$RAW_ROOT"
    --repo_name "$DETECTOR_DIR" --backbone "$BACKBONE"
    --image_size "$RESOLUTION" --max_image_size "$RESOLUTION"
    --accelerator gpu --n_gpus 1 --require_kernel "${REQUIRE_KERNEL:-0}"
    --batch_size "${BATCH_SIZE:-1}" --eff_batch_size "${EFFECTIVE_BATCH:-2}"
    --num_workers "${NUM_WORKERS:-0}" --tf32 0 --seed 0 --augment 0
    --epochs "${EPOCHS:-2}" --eval_epochs 1 --shuffle "${SHUFFLE:-1}" --print_freq 1
    --lr 0.0001 --lr_old 0.00001 --optim_groups prompt
    --use_prompts 1 --local_query 1 --num_prompts 100 --prompt_len 10
    --lambda_query 0.1 --ddl_lambda 0.15 --query_loss_grad 1
    --freeze backbone,encoder,decoder --new_params class_embed,prompts
    --freeze_shared_after_task1 0 --pseudo_dedup_iou 0
    --stop_after_steps "${STOP_AFTER_STEPS:-0}" --verify_resume "${VERIFY_RESUME:-0}"
)
