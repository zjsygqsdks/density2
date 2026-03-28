#!/usr/bin/env bash
# run.sh — Top-level entry point for DustNeRF on a headless server.
#
# Usage:
#   bash run.sh train           # train the model
#   bash run.sh train --resume  # resume training
#   bash run.sh export          # export density grid after training
#   bash run.sh all             # train then export
#
# Environment variables (optional overrides):
#   CONFIG   path to config/info.json   (default: config/info.json)
#   DATA     path to data directory     (default: data)
#   OUT      path to output directory   (default: outputs)

set -euo pipefail

CONFIG="${CONFIG:-config/info.json}"
DATA="${DATA:-data}"
OUT="${OUT:-outputs}"

MODE="${1:-all}"
shift || true
EXTRA_ARGS="$@"

CKPT="${OUT}/checkpoints/ckpt_latest.pt"

echo "========================================"
echo "  DustNeRF — Spatial Dust Reconstruction"
echo "========================================"
echo "  CONFIG : ${CONFIG}"
echo "  DATA   : ${DATA}"
echo "  OUT    : ${OUT}"
echo "  MODE   : ${MODE}"
echo "========================================"

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
    echo
    echo "[run.sh] Starting training …"
    python -m src.train \
        --config "${CONFIG}" \
        --data   "${DATA}" \
        --out    "${OUT}" \
        ${EXTRA_ARGS}
fi

if [[ "${MODE}" == "export" || "${MODE}" == "all" ]]; then
    if [[ ! -f "${CKPT}" ]]; then
        echo "[run.sh] ERROR: checkpoint not found at ${CKPT}"
        echo "[run.sh] Run 'bash run.sh train' first."
        exit 1
    fi
    echo
    echo "[run.sh] Exporting density grid …"
    python -m src.export \
        --config "${CONFIG}" \
        --ckpt   "${CKPT}" \
        --out    "${OUT}" \
        ${EXTRA_ARGS}

    echo
    echo "[run.sh] ============================================"
    echo "[run.sh] Export complete."
    echo "[run.sh] Download this archive to your local machine:"
    echo "[run.sh]   ${OUT}/dust_export.tar.gz"
    echo "[run.sh]"
    echo "[run.sh] Then visualise locally:"
    echo "[run.sh]   tar -xzf dust_export.tar.gz"
    echo "[run.sh]   python visualize_local.py --export export/"
    echo "[run.sh] ============================================"
fi
