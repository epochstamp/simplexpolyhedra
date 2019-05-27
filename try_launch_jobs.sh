#! /bin/bash

for dir in slurm_jobs/*; do
  f=`basename $dir`
  fname="${f%.*}"
  jobname=`cat "$dir" | sed -n 's/.*#SBATCH --job-name=//p'`
  grep_result=`squeue -u samait -n "$jobname" | grep "shfq"`
  job_not_launched=0
  if [ -z "$grep_result" ]; then
    echo "Job not launched yet"
    sbatch $dir
  else
    echo "Job launched"
  fi
done
