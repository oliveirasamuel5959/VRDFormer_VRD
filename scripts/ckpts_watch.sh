#!/usr/bin/env bash
set -Eeuo pipefail

# Remote RunPod server
REMOTE_HOST="root@69.30.85.138"
REMOTE_DIR="/root/VRDFormer_VRD/data/ckpts/vidorsmall_stage1"

# Local project directory
LOCAL_DIR="/home/samuel/projects/VRDFormer_VRD/data/ckpts/20261024"

# SSH configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_PORT=22145

# Check every 30 seconds
INTERVAL=30

SSH_OPTS=(
    -p "$SSH_PORT"
    -i "$SSH_KEY"
    -o BatchMode=yes
    -o ConnectTimeout=15
    -o ServerAliveInterval=15
)

SCP_OPTS=(
    -P "$SSH_PORT"
    -i "$SSH_KEY"
    -o BatchMode=yes
    -o ConnectTimeout=15
)

mkdir -p "$LOCAL_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# Get a fingerprint of every remote file.
# Includes relative path, file size and modification timestamp.
get_remote_state() {
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "find '$REMOTE_DIR' -type f -printf '%P\t%s\t%T@\n' | sort"
}

# Copy the remote checkpoint directory.
sync_checkpoints() {
    log "Synchronizing checkpoints..."

    scp -r "${SCP_OPTS[@]}" \
        "$REMOTE_HOST:$REMOTE_DIR" \
        "$LOCAL_DIR"

    log "Checkpoint synchronization completed."
}

log "Monitoring $REMOTE_HOST:$REMOTE_DIR"
log "Local destination: $LOCAL_DIR"
log "Polling interval: ${INTERVAL}s"

previous_state=""

while true; do
    if current_state="$(get_remote_state)"; then
        if [[ "$current_state" != "$previous_state" ]]; then
            log "New or modified remote files detected."

            if sync_checkpoints; then
                previous_state="$current_state"
            else
                log "WARNING: Synchronization failed; will retry."
            fi
        fi
    else
        log "WARNING: Could not query remote files; will retry."
    fi

    sleep "$INTERVAL"
done
