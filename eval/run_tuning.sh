#!/usr/bin/env bash
# Run a tuning spec to the end, resuming after a crash (up to 5 restarts, 60 s apart).
#   eval/run_tuning.sh tune/staged_singstar_full.yaml          (log: tune_run.log)
# Finished groups are skipped on resume; a group in progress continues its stored trials.
spec=${1:?usage: eval/run_tuning.sh SPEC}
cd "$(dirname "$0")/.."
for attempt in 0 1 2 3 4 5; do
  [ "$attempt" -gt 0 ] && { echo "=== restart $attempt after a crash ($(date))"; sleep 60; }
  python eval/tune.py "$spec" --resume && exit 0
done
echo "=== gave up after 5 restarts: see the traceback above"; exit 1
