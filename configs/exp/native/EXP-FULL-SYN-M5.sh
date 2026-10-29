# FULL-SYN M5 (docs/project/full-syn-r1-analysis-2026-10-29.md, section 7): EXP-FULL-SYN where, from task 2 on, the
# last ConvNeXt stage (stages_3) and the encoder train at lr_old, held to the teacher's encoder output (feature
# distillation). Everything else, task 1 included, is EXP-FULL-SYN.
EXP=${EXP:-EXP-FULL-SYN-M5}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
ARGS+=(--tune_after_task1 stages_3,encoder --feature_kd 1)
