#!/bin/bash
set -e
echo "=== STAIR Experiment Setup ==="
pip install -q numpy scipy scikit-learn datasets
mkdir -p /workspace/results
echo "Setup complete. Running experiment..."
cd /workspace
python main.py /workspace/results
echo "=== Experiment Complete ==="
cat /workspace/results/experiment_summary.json
