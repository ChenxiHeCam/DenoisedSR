#!/usr/bin/env bash
# Fetch the large model weights that don't fit in this Git repo.
#
# Two weights ship directly in models/:
#   - gat_best.pt              (1.6 MB)  GAT variable-support predictor (primary)
#   - gat_operator.pt          (0.7 MB)  GAT operator predictor (ablation)
#   - cooc_graph.joblib        (0.4 MB)  variable co-occurrence graph (GAT edges)
#
# Two RF partners are too large for Git and must be fetched separately:
#   - support_predictor_v2_40k.joblib    (~213 MB)  RF variable-support partner
#   - operator_predictor_full.joblib     (~358 MB)  RF operator partner (ablation)
#
# UPSTREAM URL: <fill in once published to Zenodo / Hugging Face Hub>
# Example (uncomment + edit when URL is known):
#   BASE="https://zenodo.org/records/XXXXXXX/files"
#   curl -fL -o models/support_predictor_v2_40k.joblib "${BASE}/support_predictor_v2_40k.joblib"
#   curl -fL -o models/operator_predictor_full.joblib  "${BASE}/operator_predictor_full.joblib"
#
# Until the public URL is set, you have two options:
#   (a) Run with GAT-only support prediction. The GAT model alone reaches
#       recall >= 0.995 on AI-Feynman (see ablation panel, Fig 2 right).
#       Replace `s = RF(col) + sigmoid(GAT(col))` with `s = sigmoid(GAT(col))`
#       in the eval scripts; the ensemble threshold tau=0.10 still applies.
#   (b) Train the RF locally from frontend/train/train_support_predictor_v2.py
#       (training data is not included in this release; see frontend/README.md
#       'Training-pipeline scripts' for the data requirements).

set -e
echo "fetch_weights.sh: upstream URL not yet configured."
echo "See $(dirname "$0")/$(basename "$0") for instructions."
exit 1
