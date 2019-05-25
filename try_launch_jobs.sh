#! /bin/bash

for dir in slurm_jobs/*; do
  f=`basename $dir`
  fname="${f%.*}"
  sbatch $dir
done
