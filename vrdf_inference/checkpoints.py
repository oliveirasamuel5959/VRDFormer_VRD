"""Checkpoint loading, class lists, device selection, and building one stage's model."""

import argparse
from pathlib import Path

import torch

from models import build_model
from vrdf_inference.console import die, log


def load_class_names(obj_path, action_path):
    def read(path):
        # VidOR's class lists are CRLF-terminated.
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    obj_names = read(obj_path)
    action_names = read(action_path)
    log(
        f"classes: {len(obj_names)} objects from {obj_path}, {len(action_names)} verbs from {action_path}"
    )
    return obj_names, action_names


def load_checkpoint(path):
    if not Path(path).exists():
        die(f"checkpoint not found: {path}")
    # weights_only=False: checkpoints pickle an argparse.Namespace (see util/checkpoints.py).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        die(f"{path} has no 'model' key (found {sorted(ckpt.keys())})")
    saved_args = ckpt.get("args")
    saved_args = (
        vars(saved_args) if isinstance(saved_args, argparse.Namespace) else saved_args
    )
    log(f"loaded {path} (epoch {ckpt.get('epoch')}, {len(ckpt['model'])} tensors)")
    return ckpt, saved_args


def resolve_device(requested):
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            die(
                f"--device {requested} was requested but CUDA is unavailable on this machine. "
                f"Run on the RunPod pod, or pass --device cpu (slow but functional)."
            )
        return torch.device(requested)
    return torch.device(requested)


def load_model(stage, args, ckpt, device):
    """Build one stage and load its weights strictly.

    ``ckpt`` is the already-loaded checkpoint dict, so the (large) file is read once per stage.
    """
    args = argparse.Namespace(**vars(args))
    args.device = str(device)
    # build_model dispatches on args.tracking for stage 1; the tracking wrapper needs the
    # matcher and returns None without targets, so inference always builds the base class.
    args.tracking = False
    model, _, _ = build_model(args)
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        # Saved from a DDP-wrapped module; unwrap so the load is not defeated by the prefix.
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"stage {stage}: {type(model).__name__} built and loaded ({n_params / 1e6:.2f}M params)"
    )
    return model
