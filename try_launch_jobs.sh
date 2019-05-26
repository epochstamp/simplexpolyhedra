#! /bin/bash

for dir in slurm_jobs/*; do
  f=`basename $dir`
  fname="${f%.*}"
  jobname=`cat "$dir" | sed -n 's/.*#SBATCH --job-name=//p'`
  grep_result=`squeue -u samait 2> /dev/null | grep jobname`
  job_not_launched=0
  if [ -z "$grep_result" ]; then
    echo "Job not launched yet"
  else
    echo "Job launched"
  fi
  #sbatch $dir
done
