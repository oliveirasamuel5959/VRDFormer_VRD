#!/usr/bin/env bash
# Diagnose a hung VRDFormer DDP run — run this *while the job is still hung*.
#
#   sh scripts/runpod/diagnose_hang.sh [path/to/train.log]
#
# Read-only with respect to the training job: it inspects processes, kernel messages,
# /dev/shm, GPU state and the log. It changes no training state. The one side effect is
# installing `py-spy` (a local Python stack sampler) into the active environment if it
# is missing, so real stacks can be captured at all; remove it with `pip uninstall py-spy`.
#
# The decisive output is section 2 (where each rank is blocked) plus section 1 (whether
# every rank is still alive). See "How to read this" at the end of the report.
set -uo pipefail

LOG="${1:-}"
PATTERN="${PATTERN:-main.py}"

hr() { printf '\n────────────────────────────────────────────────────────────\n'; }
say() { printf '%s\n' "$*"; }

printf 'VRDFormer DDP hang diagnostics — %s\n' "$(date -Is)"
say "pattern: $PATTERN${LOG:+ | log: $LOG}"

# ---------------------------------------------------------------------------
# 1. Which training processes are alive?
# ---------------------------------------------------------------------------
hr; say "1. PROCESSES"

# Collect PIDs, keeping only real Python processes: `pgrep -f main.py` also matches
# shells/editors whose command line merely contains the pattern, which would inflate
# the count and break the worker-vs-launcher classification below.
all_pids=(); workers=(); launchers=()
for pid in $(pgrep -f "$PATTERN" 2>/dev/null); do
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    case "$comm" in
        python*|python) ;;
        *) continue ;;
    esac
    all_pids+=("$pid")
    # torchrun sets RANK/LOCAL_RANK in a worker's environment; its own launcher process
    # has neither. That discriminates them regardless of torch version.
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q '^RANK='; then
        workers+=("$pid")
    else
        launchers+=("$pid")
    fi
done

n_workers=${#workers[@]}
n_launchers=${#launchers[@]}
say "match '$PATTERN': ${#all_pids[@]} python process(es) — $n_workers worker(s), $n_launchers launcher/other"

want="${WORLD_SIZE:-${NPROC_PER_NODE:-}}"
if [ -n "$want" ]; then
    if [ "$n_workers" -ne "$want" ]; then
        say ""
        say "*** WORKERS ($n_workers) != WORLD_SIZE ($want) — a rank is GONE. ***"
        say "*** Surviving ranks are blocked waiting on it: family (A), a rank-local fault. ***"
        say "*** Its own error is the root cause — see sections 5 and 7. ***"
    else
        say "worker count matches WORLD_SIZE=$want."
    fi
else
    say "(set WORLD_SIZE=<nproc_per_node> to have this compare the worker count automatically)"
fi

if [ "${#all_pids[@]}" -gt 0 ]; then
    ps -o pid,ppid,stat,etime,%cpu,rss,cmd -p "$(IFS=,; echo "${all_pids[*]}")" 2>/dev/null || true
    say ""
    say "STAT notes: D = uninterruptible sleep (usually a blocked syscall/driver), S = sleeping,"
    say "           Z = zombie (its parent never reaped it)."
else
    say "no matching process — the job is not running (it may have already exited)."
fi

# ---------------------------------------------------------------------------
# 2. Where is each process blocked?  <- the decisive section
# ---------------------------------------------------------------------------
hr; say "2. STACKS (py-spy)"
if ! command -v py-spy >/dev/null 2>&1; then
    say "py-spy not found; installing it (pip install py-spy) so stacks can be read..."
    pip install --quiet py-spy >/dev/null 2>&1 || true
fi

if command -v py-spy >/dev/null 2>&1; then
    if [ "${#all_pids[@]}" -eq 0 ]; then
        say "no processes to sample."
    fi
    for pid in "${all_pids[@]}"; do
        label="launcher/other"
        for w in "${workers[@]:-}"; do
            [ "$w" = "$pid" ] && label="worker"
        done
        say ""
        say "--- pid $pid ($label) ---"
        timeout 30 py-spy dump --pid "$pid" 2>&1 | sed 's/^/    /' \
            || say "    (py-spy dump failed for $pid — may need ptrace; rerun under sudo)"
    done
    say ""
    say "Read: a stack parked in ncclAllReduce / ProcessGroupNCCL / wait => blocked on a peer."
    say "      A stack in decord / cv2 / _worker_loop / DataLoader => local data-path stall."
else
    say "py-spy unavailable and could not be installed; stack dumps skipped."
    say "Fallbacks: 'cat /proc/<pid>/stack' (kernel stack), 'gdb -p <pid>' (native)."
fi

# ---------------------------------------------------------------------------
# 3. NCCL / distributed env actually in force for a worker
# ---------------------------------------------------------------------------
hr; say "3. NCCL & DISTRIBUTED ENV OF A WORKER"
probe_pid="${workers[0]:-}"
if [ -n "$probe_pid" ]; then
    say "from /proc/$probe_pid/environ:"
    tr '\0' '\n' < "/proc/$probe_pid/environ" 2>/dev/null \
        | grep -E '^(NCCL|TORCH_NCCL|TORCH_DISTRIBUTED|OMP|RANK|LOCAL_RANK|WORLD_SIZE|MASTER)' \
        | sort | sed 's/^/    /' || say "    (unreadable)"
    say ""
    say "Note: NCCL_P2P_DISABLE=1 present means intra-node traffic falls back to shared"
    say "memory — which makes section 6 (/dev/shm) load-bearing."
else
    say "no worker process to inspect."
fi

# ---------------------------------------------------------------------------
# 4. GPU state
# ---------------------------------------------------------------------------
hr; say "4. GPUS"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv 2>/dev/null || say "nvidia-smi query failed"
    say ""
    say "If GPUs sit at ~0% while the job is hung, the ranks are blocked, not computing."
else
    say "nvidia-smi unavailable"
fi

# ---------------------------------------------------------------------------
# 5. Kernel-level culprits: OOM killer, NVIDIA Xid, interconnect errors
# ---------------------------------------------------------------------------
hr; say "5. KERNEL MESSAGES (OOM killer / NVIDIA Xid / link errors)"
if command -v dmesg >/dev/null 2>&1; then
    if dmesg -T >/dev/null 2>&1; then DMESG="dmesg -T"; else DMESG="dmesg"; fi
    hits=$($DMESG 2>/dev/null | grep -Ei "oom|killed process|out of memory|xid|nvrm|nvidia|nvlink|pcie bus error" | tail -n 40 || true)
    if [ -n "$hits" ]; then
        say "$hits"
        say ""
        say "A 'Killed process ... python' line here means the OOM killer took a rank."
        say "An 'Xid' line means a GPU fault."
    else
        say "none of: oom / killed process / out of memory / xid / nvrm / nvidia / nvlink in the kernel log"
    fi
else
    say "dmesg unavailable (try: sudo sh scripts/runpod/diagnose_hang.sh ${LOG})"
fi

# ---------------------------------------------------------------------------
# 6. Shared memory and host RAM  (the classic NCCL silent-hang cause)
# ---------------------------------------------------------------------------
hr; say "6. /dev/shm AND HOST MEMORY"
df -h /dev/shm 2>/dev/null || say "/dev/shm not found"
shm_mb=$(df -m /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [ -n "${shm_mb:-}" ] && [ "$shm_mb" -lt 1024 ] 2>/dev/null; then
    say ""
    say "*** WARNING: /dev/shm is only ${shm_mb} MB. ***"
    say "NCCL uses shared memory for intra-node transport. With NCCL_P2P_DISABLE=1 it is the"
    say "*only* intra-node path, and a too-small /dev/shm is a well-known silent-hang cause."
    say "Fix: raise the container's shared memory (docker/pod '--shm-size', 8g suggested) — or"
    say "drop NCCL_P2P_DISABLE=1 entirely if P2P works on this host."
fi
say ""
free -h 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. The log: last lines + watchdog/NCCL markers
# ---------------------------------------------------------------------------
hr; say "7. TRAINING LOG"
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    say "last 30 lines of $LOG:"
    tail -n 30 "$LOG" | sed 's/^/    /'
    hr; say "watchdog / timeout / NCCL / abort markers in $LOG"
    grep -nEi "watchdog|timeout|timed out|nccl|unhandled|traceback|non-finite|aborting all ranks" "$LOG" \
        | tail -n 40 | sed 's/^/    /' || say "    none found"
    say ""
    say "A 'Watchdog caught collective operation timeout' line names the op AND the ranks that"
    say "did not join — that directly identifies which rank stopped participating."
else
    say "no log file given. Usage: sh scripts/runpod/diagnose_hang.sh path/to/train.log"
fi

# ---------------------------------------------------------------------------
hr; cat <<'EOF'
HOW TO READ THIS
----------------
1. Count the workers (section 1).
   fewer workers than GPUs  -> family (A): a rank died. Its own error is the root cause;
                               find it in section 5 (OOM/Xid) or in that rank's own stderr.
   all workers alive        -> family (B): everyone is blocked. Sections 2 and 6 apply.

2. Read the stacks (section 2).
   all ranks in ncclAllReduce / ProcessGroupNCCL::wait
        -> a collective is waiting for a peer that never arrives. If every rank is parked
           on the SAME collective, one rank is slow/stuck upstream (data path). If they are
           on DIFFERENT collectives, the ranks have diverged in control flow.
   ranks in decord / cv2.resize / DataLoader _worker_loop
        -> data-path stall. Check clip lengths and decord against section 1's RSS numbers;
           a worker that hangs is not always caught by DataLoader's worker timeout.
   launcher in _poll_processes / join, workers busy
        -> the job is healthy, just slow.

3. Then check, in order: /dev/shm size and NCCL link errors (6, 5); whether the loss went
   non-finite ("aborting all ranks" in 7 — with the engine fix this now names the rank).

NEXT RUN — make the hang self-diagnosing so you get all ranks' stacks automatically:
   TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
   TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
   TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
   TORCH_DISTRIBUTED_DEBUG=DETAIL \
   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
   torchrun --standalone --nproc_per_node=<N> main.py ...
EOF
