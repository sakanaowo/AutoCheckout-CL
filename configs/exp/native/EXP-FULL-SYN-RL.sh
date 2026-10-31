# Relearn (docs/project/task6-relearn-plan-2026-10-30.md): EXP-FULL-SYN on the task-5 model of round 1 expanded to
# the class slots of configs/relearn/task_config_relearn.json, where tasks 6 and 7 relearn 50 SKUs under alias
# classes. Set by autocheckout/relearn.py like EXP-FULL-SYN; TASK_CFG is the relearn config.
EXP=${EXP:-EXP-FULL-SYN-RL}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
ARGS+=(--n_classes 275)  # configs/relearn/task_config_relearn.json
