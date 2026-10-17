#!/bin/bash
# after the bench run completes: download its small outputs (not the 4 GB working copies)
export KAGGLE_API_TOKEN=KGAT_6c9eae58978c97e187d2e6452774319d
W=/tmp/claude-0/-home-user-shl-grammar-scoring/b6319532-f3ee-57bf-b6c7-f5215a48473a/scratchpad/kaggle_www.py
K=${1:-aryanawasthiteykik/casmi26-bench}; OUT=${2:-/home/user/benchdata/run2}
mkdir -p $OUT; cd /tmp
for j in 1 2 3 4 5 6; do
  timeout 900 python3 $W kernels output $K -p $OUT -o --file-pattern '^(bench_.*\.(json|csv)|.*submission.*\.csv|e004_candidate_inputs\.json|eng_lists\.json|pc_lists\.json|run_manifest\.json)$' 2>&1 | tail -2
  [ -f $OUT/bench_answers.csv ] && [ -f $OUT/submission.csv ] && break
  sleep 90
done
ls -la $OUT
