#!/bin/bash
# Train all 9 security heads sequentially.
# Usage: bash experiments/security/run_all.sh [num_epochs]
EPOCHS=${1:-3}
for SHARD in $(seq 0 8); do
    echo "=== Training shard $SHARD / 8 (epochs=$EPOCHS) ==="
    pixi run python -m experiments.security.training --shard-id "$SHARD" --num-epochs "$EPOCHS"
done
