# Training on a Server with tmux

A practical runbook for keeping VRDFormer stage-1 training alive across SSH disconnects and checking its output later.

**Maintainer:** VRDFormer project maintainers | **Last verified against:** repository workflow on 2026-11-17

> Start training inside tmux from the beginning. tmux keeps its session running after you detach and disconnect SSH; it cannot move an already-running foreground training process into a tmux session.

## 1. Check that tmux is available

From the server shell:

```bash
tmux -V
```

If the command is not found and the server uses Ubuntu or Debian, install tmux:

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

## 2. Connect to the server and create a session

SSH into the server, go to the repository, then create a named tmux session:

```bash
cd ~/VRDFormer_VRD
tmux new -s vrdformer-stage1
```

Replace `~/VRDFormer_VRD` with the actual checkout path if different. You are now inside tmux; commands entered here continue running when the session is detached.

If the repository uses a virtual environment, activate it inside this session before training, for example:

```bash
source .venv/bin/activate
```

Use the environment setup appropriate for your server if it is not `.venv`.

## 3. Check the training configuration, then start training

The stage-1 wrapper is `scripts/stage1/train_vidorsmall.sh`; it launches `main.py` using `configs/vidorsmall_stage1.json`. Before starting, verify that the config has the intended dataset paths, `output_dir`, `batch_size`, `epochs`, and `lr_drop`. The config values override matching command-line defaults.

The checked-in wrapper currently launches two processes. If you need a different GPU count, check or edit its `torchrun --nproc_per_node` value before running it.

To save a persistent copy of all terminal output while still seeing it live, run this **inside tmux**, from the repository root:

```bash
mkdir -p logs
LOG="logs/stage1-vidorsmall-$(date +%Y%m%d-%H%M%S).log"
set -o pipefail
export PYTHONUNBUFFERED=1
sh scripts/stage1/train_vidorsmall.sh 2>&1 | tee "$LOG"
```

`set -o pipefail` makes the shell report a failure from the training command even though its output is piped through `tee`. Keep the tmux pane open while the command runs. Do not press Ctrl+C in this pane unless you intend to interrupt training.

Training checkpoints are written under the configured `output_dir`. Check the config and server disk space before a long run.

## 4. Detach from tmux without stopping training

While attached to the training session, press these keys in sequence:

1. **Ctrl+B** (tmux prefix)
2. Release both keys, then press **D**

You should return to the normal server shell. The training keeps running inside tmux. You can now disconnect SSH safely.

Do not type `exit` in the training shell or close the tmux session while the command is running; that can terminate the process. Detach with **Ctrl+B**, then **D** instead.

## 5. List tmux sessions

After reconnecting to the server—or from another server shell—list sessions with:

```bash
tmux ls
```

A session named `vrdformer-stage1` means that tmux still has that session. If tmux says there is no server running, there are no active tmux sessions for your user.

## 6. Reattach and inspect the live training output

Reattach to the session:

```bash
tmux attach -t vrdformer-stage1
```

You will return to the tmux terminal. If training is still running, its output continues to appear. If it has finished, the command has returned to the shell prompt; review the saved log as described below.

If tmux reports that the session is already attached elsewhere, you can attach here and detach the other client with:

```bash
tmux attach -d -t vrdformer-stage1
```

This disconnects the other tmux client, but leaves the session and its processes running.

## 7. Read or follow the saved log

You can inspect the latest stage-1 log without attaching to tmux:

```bash
ls -lt logs/stage1-vidorsmall-*.log
```

Follow the newest log live from another shell:

```bash
LOG=$(ls -t logs/stage1-vidorsmall-*.log | head -n 1)
tail -n 50 -f "$LOG"
```

`tail -f` shows new output as it is written. Press **Ctrl+C** to stop following the file; this stops `tail`, not the training process. The log files are relative to the repository root, so run these commands from there or use the full log path.

## 8. Finish or intentionally stop a run

When training finishes, the process returns to the shell prompt inside tmux. The tmux session itself remains until you exit that shell or close the session. Reattach and check the last lines of the log and the configured checkpoint directory before cleaning anything up.

To intentionally interrupt training, reattach and press **Ctrl+C** in the pane running the training command. This requests an interrupt; it is not a graceful checkpoint mechanism. Do not use `tmux kill-session` as a way to detach—it terminates processes in that session.

## Quick reference

| Task | Command or keys |
| --- | --- |
| Create a session | `tmux new -s vrdformer-stage1` |
| List sessions | `tmux ls` |
| Detach and keep training alive | **Ctrl+B**, then **D** |
| Reattach | `tmux attach -t vrdformer-stage1` |
| Follow latest saved log | `tail -n 50 -f "$LOG"` after setting `LOG` as above |
| Stop following a log | **Ctrl+C** in the shell running `tail` |
| Stop training intentionally | Reattach, then **Ctrl+C** in the training pane |
