#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mixedsde

for SEED in {1..500}
do
  echo "Repetition $SEED"
  python -u main_simus.py \
    --seed $SEED \
    --model m3 \
    --N 500 \
    > res/log_${SEED}.out 2> res/log_${SEED}.err &
  
  # Optional: limits the number of simultaneous jobs (e.g., 5 at a time)
  if (( $(jobs | wc -l) >= 5 )); then
    wait -n
  fi
done

wait
echo "Finished."
