#!/bin/bash
# Ablation: run 4 configs on 5 target objects. ~25 min each, ~2h total.
set -e
cd "$(dirname "$0")"
source /home/bowenj/anaconda3/etc/profile.d/conda.sh
conda activate frogger

for cfg in A B C D; do
    case $cfg in
        A) NO_MULTI=1; NO_BASE=1; label="A_baseline";;
        B) NO_MULTI=0; NO_BASE=1; label="B_multi";;
        C) NO_MULTI=1; NO_BASE=0; label="C_base";;
        D) NO_MULTI=0; NO_BASE=0; label="D_both";;
    esac
    echo "=========================================="
    echo "  Config $cfg ($label): NO_MULTI=$NO_MULTI NO_BASE=$NO_BASE"
    echo "=========================================="
    out="output/ablation_$label"
    mkdir -p "$out"
    FROGGER_NO_MULTI=$NO_MULTI FROGGER_NO_BASE=$NO_BASE \
        python -c "
import os, sys; sys.path.insert(0, '.')
import run_target_objects_pgd as m
m.OUT_DIR = '$out'
os.makedirs(m.OUT_DIR, exist_ok=True)
m.main()
" 2>&1 | tee "/tmp/ablation_${label}.log" | grep -E "^(  ===|  Best:|  Variant|  Actuation|  Optimization cand|  [1-9][0-9]*/|  G[0-9]|  assignment|Object|[a-z_]+ *[0-9]+|============================================================)" || true
done
echo "All configs done."
