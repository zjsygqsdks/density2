#!/usr/bin/env bash
# =============================================================================
# sync_results.sh — Poll the remote server for training/export completion,
#                   download results, extract them, and optionally run analysis.
#
# Usage
# -----
#   bash sync_results.sh [OPTIONS]
#
# Options
#   --wait         Block until training finishes, then download (default: download now)
#   --poll N       Polling interval in seconds when --wait is used (default: 60)
#   --no-analyze   Skip running src/analyze.py after download
#   --no-improve   Skip running auto_improve.py after analysis
#   --local-dir D  Local directory to store downloaded results (default: ./results)
#
# Required: same server.env / env vars as deploy.sh
# =============================================================================

set -euo pipefail

# ── Load server.env ───────────────────────────────────────────────────────────
if [[ -f server.env ]]; then
    # shellcheck disable=SC1091
    source server.env
fi

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_DIR="${REMOTE_DIR:-}"

if [[ -z "${REMOTE_HOST}" || -z "${REMOTE_USER}" || -z "${REMOTE_DIR}" ]]; then
    echo "[sync] ERROR: REMOTE_HOST, REMOTE_USER, REMOTE_DIR not set. See deploy.sh for details."
    exit 1
fi

SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_rsa}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_DIR}/.venv}"
CONDA_ENV="${CONDA_ENV:-}"
NOHUP_LOG="${NOHUP_LOG:-${REMOTE_DIR}/train.log}"

# ── Parse flags ───────────────────────────────────────────────────────────────
DO_WAIT=0
POLL_INTERVAL=60
DO_ANALYZE=1
DO_IMPROVE=1
LOCAL_DIR="./results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)        DO_WAIT=1 ;;
        --poll)        shift; POLL_INTERVAL="$1" ;;
        --no-analyze)  DO_ANALYZE=0 ;;
        --no-improve)  DO_IMPROVE=0 ;;
        --local-dir)   shift; LOCAL_DIR="$1" ;;
        *) echo "[sync] Unknown option: $1"; exit 1 ;;
    esac
    shift
done

SSH_OPTS="-p ${REMOTE_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
if [[ -f "${SSH_KEY}" ]]; then
    SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY}"
fi

ssh_run() {
    ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

if [[ -n "${CONDA_ENV}" ]]; then
    ACTIVATE_CMD="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
else
    ACTIVATE_CMD="source ${REMOTE_VENV}/bin/activate"
fi

REMOTE_ARCHIVE="${REMOTE_DIR}/outputs/dust_export.tar.gz"
REMOTE_METRICS="${REMOTE_DIR}/outputs/train_metrics.jsonl"
REMOTE_CKPT="${REMOTE_DIR}/outputs/checkpoints/ckpt_latest.pt"

echo "============================================================"
echo "  DustNeRF Sync"
echo "============================================================"
echo "  Source  : ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "  Local   : ${LOCAL_DIR}"
echo "  Wait    : ${DO_WAIT}"
echo "============================================================"

# ── Optional: wait for training to finish ─────────────────────────────────────
if [[ "${DO_WAIT}" -eq 1 ]]; then
    echo
    echo "[sync] Polling for training completion every ${POLL_INTERVAL}s …"
    echo "       (Ctrl-C to abort polling and download whatever exists)"
    while true; do
        IS_RUNNING=$(ssh_run "pgrep -f 'src.train' && echo yes || echo no" 2>/dev/null || echo "no")
        HAS_ARCHIVE=$(ssh_run "test -f '${REMOTE_ARCHIVE}' && echo yes || echo no" 2>/dev/null || echo "no")

        if [[ "${HAS_ARCHIVE}" == "yes" ]]; then
            echo "[sync] Export archive found on server. Proceeding with download."
            break
        fi

        if [[ "${IS_RUNNING}" == "no" ]]; then
            echo "[sync] Training process ended. Triggering export …"
            ssh_run "bash -lc '
                cd ${REMOTE_DIR}
                ${ACTIVATE_CMD}
                bash run.sh export >> ${NOHUP_LOG}.export 2>&1
            '" || true
            break
        fi

        # Print last progress line from log
        LAST_LINE=$(ssh_run "tail -1 ${NOHUP_LOG} 2>/dev/null" || echo "")
        echo "  $(date '+%H:%M:%S') still training …  ${LAST_LINE}"
        sleep "${POLL_INTERVAL}"
    done
fi

# ── If no archive yet, trigger export now ────────────────────────────────────
HAS_ARCHIVE=$(ssh_run "test -f '${REMOTE_ARCHIVE}' && echo yes || echo no" 2>/dev/null || echo "no")
if [[ "${HAS_ARCHIVE}" != "yes" ]]; then
    HAS_CKPT=$(ssh_run "test -f '${REMOTE_CKPT}' && echo yes || echo no" 2>/dev/null || echo "no")
    if [[ "${HAS_CKPT}" == "yes" ]]; then
        echo
        echo "[sync] No export archive found; triggering export on server …"
        ssh_run "bash -lc '
            set -euo pipefail
            cd ${REMOTE_DIR}
            ${ACTIVATE_CMD}
            bash run.sh export
        '"
    else
        echo "[sync] ERROR: No checkpoint and no archive found on server."
        echo "       Run 'bash deploy.sh train' first."
        exit 1
    fi
fi

# ── Create local results directory ────────────────────────────────────────────
mkdir -p "${LOCAL_DIR}"

# ── Download archive ──────────────────────────────────────────────────────────
ARCHIVE_BASENAME="dust_export.tar.gz"
LOCAL_ARCHIVE="${LOCAL_DIR}/${ARCHIVE_BASENAME}"

echo
echo "[sync] Downloading ${REMOTE_ARCHIVE} …"
scp ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARCHIVE}" \
    "${LOCAL_ARCHIVE}"
echo "[sync] Archive saved → ${LOCAL_ARCHIVE}"

# ── Download training metrics ─────────────────────────────────────────────────
LOCAL_METRICS="${LOCAL_DIR}/train_metrics.jsonl"
echo "[sync] Downloading training metrics …"
scp ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_METRICS}" \
    "${LOCAL_METRICS}" 2>/dev/null \
    && echo "[sync] Metrics saved → ${LOCAL_METRICS}" \
    || echo "[sync] WARNING: train_metrics.jsonl not found (optional)."

# ── Download train.log (plain text) ──────────────────────────────────────────
LOCAL_LOG="${LOCAL_DIR}/train.log"
scp ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}:${NOHUP_LOG}" \
    "${LOCAL_LOG}" 2>/dev/null \
    && echo "[sync] train.log saved → ${LOCAL_LOG}" \
    || echo "[sync] WARNING: train.log not found."

# ── Extract archive ───────────────────────────────────────────────────────────
EXPORT_DIR="${LOCAL_DIR}/export"
echo
echo "[sync] Extracting archive …"
tar -xzf "${LOCAL_ARCHIVE}" -C "${LOCAL_DIR}/"
echo "[sync] Extracted → ${EXPORT_DIR}"

# ── Run analysis ──────────────────────────────────────────────────────────────
if [[ "${DO_ANALYZE}" -eq 1 ]]; then
    echo
    echo "[sync] Running analysis …"
    python src/analyze.py \
        --export    "${EXPORT_DIR}" \
        --metrics   "${LOCAL_METRICS}" \
        --out       "${LOCAL_DIR}/analysis" \
        || echo "[sync] WARNING: analysis script failed (check Python deps)."
fi

# ── Run auto-improvement ──────────────────────────────────────────────────────
if [[ "${DO_ANALYZE}" -eq 1 && "${DO_IMPROVE}" -eq 1 ]]; then
    REPORT="${LOCAL_DIR}/analysis/analysis_report.json"
    if [[ -f "${REPORT}" ]]; then
        echo
        echo "[sync] Running auto-improvement …"
        python auto_improve.py \
            --report "${REPORT}" \
            --config config/info.json \
            || echo "[sync] WARNING: auto_improve.py failed."
    fi
fi

echo
echo "============================================================"
echo "  Sync complete."
echo "  Results   : ${LOCAL_DIR}/"
echo "  3-D view  : open ${LOCAL_DIR}/export/dust_cloud.html"
echo "  Slices    : ${LOCAL_DIR}/export/dust_density_slices.png"
if [[ "${DO_ANALYZE}" -eq 1 ]]; then
    echo "  Analysis  : ${LOCAL_DIR}/analysis/"
    echo "  Report    : ${LOCAL_DIR}/analysis/analysis_report.json"
fi
echo ""
echo "  To visualise interactively:"
echo "    python visualize_local.py --export ${LOCAL_DIR}/export/"
echo "============================================================"
