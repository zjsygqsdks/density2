#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Upload DustNeRF project to a remote training server and start
#              training / export in a persistent background session.
#
# Usage
# -----
#   bash deploy.sh [MODE]
#
# MODE (default: all)
#   train    — upload and start training only
#   export   — trigger export on server (checkpoint must exist)
#   all      — upload, train, then export
#   status   — show running jobs on server
#   stop     — kill training process on server
#
# Required environment variables (or set in server.env):
#   REMOTE_HOST   — server hostname or IP           e.g. 192.168.1.100
#   REMOTE_PORT   — SSH port                        e.g. 22
#   REMOTE_USER   — SSH username                    e.g. ubuntu
#   REMOTE_DIR    — absolute path on server         e.g. /home/ubuntu/density2
#
# Optional:
#   SSH_KEY       — path to private key             (default: ~/.ssh/id_rsa)
#   DATA_DIR      — local data directory to upload  (default: data/)
#   REMOTE_VENV   — path to Python venv on server   (default: ${REMOTE_DIR}/.venv)
#   CONDA_ENV     — conda environment name (used instead of venv if set)
#   CONFIG        — config file to push             (default: config/info.json)
#   NOHUP_LOG     — remote log file path            (default: ${REMOTE_DIR}/train.log)
# =============================================================================

set -euo pipefail

# ── Load server.env if it exists ─────────────────────────────────────────────
if [[ -f server.env ]]; then
    # shellcheck disable=SC1091
    source server.env
fi

# ── Required vars ─────────────────────────────────────────────────────────────
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_DIR="${REMOTE_DIR:-}"

if [[ -z "${REMOTE_HOST}" || -z "${REMOTE_USER}" || -z "${REMOTE_DIR}" ]]; then
    echo "[deploy] ERROR: REMOTE_HOST, REMOTE_USER, and REMOTE_DIR must be set."
    echo "         Create a server.env file or export them before running."
    echo ""
    echo "  Example server.env:"
    echo "    REMOTE_HOST=192.168.1.100"
    echo "    REMOTE_PORT=22"
    echo "    REMOTE_USER=ubuntu"
    echo "    REMOTE_DIR=/home/ubuntu/density2"
    exit 1
fi

# ── Optional vars ────────────────────────────────────────────────────────────
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_rsa}"
DATA_DIR="${DATA_DIR:-data}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_DIR}/.venv}"
CONDA_ENV="${CONDA_ENV:-}"
CONFIG="${CONFIG:-config/info.json}"
NOHUP_LOG="${NOHUP_LOG:-${REMOTE_DIR}/train.log}"
MODE="${1:-all}"

# ── SSH / SCP helpers ────────────────────────────────────────────────────────
SSH_OPTS="-p ${REMOTE_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
if [[ -f "${SSH_KEY}" ]]; then
    SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY}"
fi

ssh_run() {
    # Run a command on the remote server
    ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

rsync_push() {
    # Sync local files to remote, skipping large/temp dirs
    rsync -avz --progress \
        -e "ssh ${SSH_OPTS}" \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='outputs/' \
        --exclude='*.tar.gz' \
        "$@"
}

# ── Python activation snippet ────────────────────────────────────────────────
if [[ -n "${CONDA_ENV}" ]]; then
    ACTIVATE_CMD="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
else
    ACTIVATE_CMD="source ${REMOTE_VENV}/bin/activate"
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  DustNeRF Deploy"
echo "============================================================"
echo "  Target  : ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "  Remote  : ${REMOTE_DIR}"
echo "  Mode    : ${MODE}"
echo "============================================================"

# ── STATUS ───────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "status" ]]; then
    echo
    echo "[deploy] Checking running jobs on server …"
    ssh_run "ps aux | grep -E 'src.train|src.export|run.sh' | grep -v grep || echo '(no training process found)'"
    echo
    echo "[deploy] Latest checkpoint:"
    ssh_run "ls -lht ${REMOTE_DIR}/outputs/checkpoints/ 2>/dev/null | head -5 || echo '(no checkpoints)'"
    echo
    echo "[deploy] Last 20 lines of train.log:"
    ssh_run "tail -20 ${NOHUP_LOG} 2>/dev/null || echo '(log not found)'"
    exit 0
fi

# ── STOP ─────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "stop" ]]; then
    echo
    echo "[deploy] Stopping training process on server …"
    ssh_run "pkill -f 'src.train' && echo 'stopped src.train' || echo 'no src.train process found'"
    exit 0
fi

# ── UPLOAD CODE ──────────────────────────────────────────────────────────────
echo
echo "[deploy] Uploading project code …"
ssh_run "mkdir -p ${REMOTE_DIR}"
rsync_push \
    --exclude="data/" \
    ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# ── UPLOAD DATA (if directory exists) ────────────────────────────────────────
if [[ -d "${DATA_DIR}" ]]; then
    echo
    echo "[deploy] Uploading data directory (${DATA_DIR}) …"
    rsync_push \
        "${DATA_DIR}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/"
else
    echo "[deploy] WARNING: local '${DATA_DIR}' not found — skipping data upload."
    echo "         Place your video files in ${REMOTE_DIR}/data/ on the server manually."
fi

# ── INSTALL DEPENDENCIES ─────────────────────────────────────────────────────
echo
echo "[deploy] Installing Python dependencies on server …"
ssh_run "bash -lc '
    set -euo pipefail
    cd ${REMOTE_DIR}
    if [[ -z \"${CONDA_ENV}\" ]]; then
        python3 -m venv ${REMOTE_VENV} --system-site-packages 2>/dev/null || true
        source ${REMOTE_VENV}/bin/activate
    else
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
    fi
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo \"[deploy] Dependencies installed.\"
'"

# ── LAUNCH TRAINING / EXPORT ──────────────────────────────────────────────────
if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
    echo
    echo "[deploy] Launching training on server (background, log → ${NOHUP_LOG}) …"
    ssh_run "bash -lc '
        set -euo pipefail
        cd ${REMOTE_DIR}
        ${ACTIVATE_CMD}
        mkdir -p outputs/checkpoints outputs/logs
        nohup bash run.sh train > ${NOHUP_LOG} 2>&1 &
        echo \$! > ${REMOTE_DIR}/.train_pid
        echo \"[deploy] Training started, PID=\$(cat ${REMOTE_DIR}/.train_pid)\"
        echo \"         Monitor: ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} tail -f ${NOHUP_LOG}\"
    '"
fi

if [[ "${MODE}" == "export" ]]; then
    echo
    echo "[deploy] Triggering export on server …"
    CKPT="${REMOTE_DIR}/outputs/checkpoints/ckpt_latest.pt"
    ssh_run "bash -lc '
        set -euo pipefail
        cd ${REMOTE_DIR}
        ${ACTIVATE_CMD}
        if [[ ! -f \"${CKPT}\" ]]; then
            echo \"[deploy] ERROR: ${CKPT} not found. Train first.\"; exit 1
        fi
        nohup bash run.sh export > ${NOHUP_LOG}.export 2>&1 &
        echo \"[deploy] Export started, log → ${NOHUP_LOG}.export\"
    '"
fi

echo
echo "============================================================"
echo "  Deployment complete."
echo "  Monitor training:"
echo "    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} \\"
echo "        tail -f ${NOHUP_LOG}"
echo ""
echo "  When training finishes, download results:"
echo "    bash sync_results.sh"
echo "============================================================"
