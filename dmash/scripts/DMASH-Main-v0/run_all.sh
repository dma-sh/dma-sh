#!/bin/bash -l

static_id=0
n_statics=6

for i in $(seq $n_statics); do
  static=$((static_id + i - 1))
  sbatch run.slurm -a $static
done


