#!/usr/bin/env bash
# Populate assets/adapters/ with symlinks to every thesis adapter iteration:
#   assets/adapters/pto-iter<N>  -> ../PTO/<run>/iteration_<N>/adapter
#   assets/adapters/grpo-iter<N> -> ../GRPO/<run>/iteration_<N>/adapter
# serve.sh registers each as model "mi-coach-<name>". Best iterations per the
# thesis evaluation: PTO iter 10, GRPO iter 8.

set -euo pipefail
cd "$(dirname "$0")/.."

PTO_RUN="PTO/PTO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_M8_PTgreedy"
GRPO_RUN="GRPO/GRPO_Iterative_Q1Q2_Llama32-1B_LA0_MCL12_G8"

mkdir -p assets/adapters
for method in pto grpo; do
  run_var="${method^^}_RUN"
  run="${!run_var}"
  for it_dir in "assets/$run"/iteration_*; do
    [[ -f "$it_dir/adapter/adapter_config.json" ]] || continue
    n="${it_dir##*iteration_}"
    link="assets/adapters/$method-iter$n"
    rm -rf "$link"
    ln -s "../$run/iteration_$n/adapter" "$link"
  done
done
echo "Linked $(ls assets/adapters | wc -l) adapters:"
ls assets/adapters | sort -V | tr '\n' ' '; echo
