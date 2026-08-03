"""Cheap device-placement repro for the stage-1 track-query path.

Exercises the exact code that crashed -- HungarianMatcher.forward,
TrackingBase.add_track_queries_to_targets and SetCriterionTrack.forward -- with
synthetic CUDA tensors, forcing both the false-negative and the false-positive
branch to fire on every target. No dataset, no checkpoint, no backbone: runs in
a couple of seconds.

    python tools/repro_track_device.py

Add --real to additionally build the real model + one real batch from a config
and run one forward/backward (needs the dataset + DETR pretrain weights):

    python tools/repro_track_device.py --real --dataset_config configs/vidor_kaggle_stage1.json

To confirm the harness actually catches the bug, run it against the old code:

    git stash && python tools/repro_track_device.py ; git stash pop
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.matcher import HungarianMatcher
from models.vrdformer_track import SetCriterionTrack, TrackingBase


NUM_QUERIES = 200
NUM_OBJ_CLASSES = 79   # vidor + focal_loss -> 80 - 1
NUM_VERB_CLASSES = 50
BATCH_SIZE = 2
NUM_PAIRS = 6          # ground-truth subject-object pairs per frame


class TrackerShim(TrackingBase):
    """Bare TrackingBase: add_track_queries_to_targets only needs the matcher,
    the two probabilities and num_queries."""

    def __init__(self, matcher, fp_prob, fn_prob):
        nn.Module.__init__(self)  # TrackingBase.__init__ does not call super()
        TrackingBase.__init__(self,
                              track_query_false_positive_prob=fp_prob,
                              track_query_false_negative_prob=fn_prob,
                              matcher=matcher,
                              backprop_prev_frame=False)
        self.num_queries = NUM_QUERIES


def rand_boxes(n, device):
    cxcy = torch.rand(n, 2, device=device) * 0.6 + 0.2
    wh = torch.rand(n, 2, device=device) * 0.2 + 0.05
    return torch.cat([cxcy, wh], dim=1)


def make_outputs(device, num_queries=NUM_QUERIES, requires_grad=False):
    def t(*shape):
        x = torch.randn(*shape, device=device)
        return x.requires_grad_() if requires_grad else x
    return {
        'pred_sub_logits': t(BATCH_SIZE, num_queries, NUM_OBJ_CLASSES + 1),
        'pred_obj_logits': t(BATCH_SIZE, num_queries, NUM_OBJ_CLASSES + 1),
        'pred_verb_logits': t(BATCH_SIZE, num_queries, NUM_VERB_CLASSES),
        'pred_sub_boxes': torch.rand(BATCH_SIZE, num_queries, 4, device=device),
        'pred_obj_boxes': torch.rand(BATCH_SIZE, num_queries, 4, device=device),
        'hs_embed': t(BATCH_SIZE, num_queries, 256),
    }


def make_target(device, shift_track_ids=0):
    """Mirrors datasets/dataset.py ConvertCocoPolysToMask + target_to_cuda."""
    sub_tids = torch.arange(NUM_PAIRS, device=device) + shift_track_ids
    obj_tids = torch.arange(NUM_PAIRS, device=device) + 100 + shift_track_ids
    verb = torch.zeros(NUM_PAIRS, NUM_VERB_CLASSES, device=device)
    verb[torch.arange(NUM_PAIRS), torch.randint(0, NUM_VERB_CLASSES, (NUM_PAIRS,))] = 1.0
    return {
        'sub_boxes': rand_boxes(NUM_PAIRS, device),
        'obj_boxes': rand_boxes(NUM_PAIRS, device),
        'sub_labels': torch.randint(0, NUM_OBJ_CLASSES, (NUM_PAIRS,), device=device),
        'obj_labels': torch.randint(0, NUM_OBJ_CLASSES, (NUM_PAIRS,), device=device),
        'verb_labels': verb,
        'sub_track_ids': sub_tids,
        'obj_track_ids': obj_tids,
    }


def assert_all_cuda(obj, name, device):
    if isinstance(obj, torch.Tensor):
        assert obj.device == device, f"{name} is on {obj.device}, expected {device}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_all_cuda(v, f"{name}.{k}", device)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_all_cuda(v, f"{name}[{i}]", device)


def run_synthetic(device):
    matcher = HungarianMatcher(cost_sub_class=0.5, cost_obj_class=0.5,
                               cost_verb_class=1, cost_bbox=5, cost_giou=2,
                               focal_loss=True, focal_alpha=0.25).to(device)

    # both probabilities at 1.0 so the FN subsampling and the FP injection
    # branches are guaranteed to run on every target.
    tracker = TrackerShim(matcher, fp_prob=1.0, fn_prob=1.0).to(device)

    for trial in range(20):
        torch.manual_seed(trial)

        prev_out = make_outputs(device)
        # prev_target is a *nested* dict inside target -- the thing target_to_cuda
        # used to skip. Build it already on the device, as the fixed helper does.
        prev_targets = [make_target(device, shift_track_ids=0) for _ in range(BATCH_SIZE)]
        targets = [make_target(device, shift_track_ids=0) for _ in range(BATCH_SIZE)]
        for target, prev_target in zip(targets, prev_targets):
            target['prev_target'] = prev_target

        prev_indices = matcher(prev_out, prev_targets)
        tracker.add_track_queries_to_targets(targets, prev_indices, prev_out,
                                             add_false_pos=True)

        for i, target in enumerate(targets):
            for key in ('track_query_hs_embeds', 'track_query_sub_boxes',
                        'track_query_obj_boxes', 'track_query_match_ids',
                        'track_queries_mask', 'track_queries_fal_pos_mask'):
                assert key in target, f"missing {key}"
                assert_all_cuda(target[key], f"targets[{i}].{key}", device)

        n_track = [len(t['track_query_hs_embeds']) for t in targets]
        n_fp = [int(t['track_queries_fal_pos_mask'][:n].sum())
                for t, n in zip(targets, n_track)]
        assert len(set(n_track)) == 1, (
            f"track query counts differ across the batch ({n_track}); "
            "torch.stack in Transformer.forward would fail")
        assert sum(n_fp) > 0, "false-positive branch never fired -- repro is not covering it"

        # the decoder input is [track queries | static queries]
        num_queries_total = n_track[0] + NUM_QUERIES
        outputs = make_outputs(device, num_queries=num_queries_total, requires_grad=True)
        outputs['aux_outputs'] = []

        criterion = SetCriterionTrack(
            NUM_OBJ_CLASSES, NUM_VERB_CLASSES,
            matcher=matcher,
            weight_dict={'loss_ce': 1, 'loss_ce_verb': 1, 'loss_bbox': 5, 'loss_giou': 2},
            eos_coef=0.1,
            losses=["labels", "verb_labels", "cardinality", "boxes"],
            focal_loss=True, focal_alpha=0.25, focal_gamma=2.0,
            track_query_false_positive_eos_weight=True,
        ).to(device)

        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k]
                     for k in loss_dict if k in criterion.weight_dict)
        losses.backward()

        if trial == 0:
            print(f"  track queries/target: {n_track[0]}  "
                  f"(false positives: {n_fp})  loss: {losses.item():.4f}")

    print("  20/20 trials passed (matcher -> track queries -> criterion -> backward)")


def run_real(argv):
    import json
    from argparse import Namespace

    import util.dist as dist
    from datasets import dataloader_initializer
    from engine import is_loss_invalid
    from main import get_args_parser
    from models import model_initializer
    from util.checkpoints import param_initializer
    from util.misc import target_to_cuda
    from util.optim import optim_initializer

    args = argparse.ArgumentParser(parents=[get_args_parser()]).parse_args(argv)
    args.debug = True          # skip raw annotations + zero-shot triplet diff
    args.num_workers = 0
    dist.init_distributed_mode(args)
    with open(args.dataset_config) as f:
        vars(args).update(json.load(f))
    args.debug = True
    args.num_workers = 0

    device = torch.device(args.device)
    model, model_without_ddp, criterion, _ = model_initializer(args, device)
    optimizer = optim_initializer(args, model_without_ddp)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    param_initializer(args, model_without_ddp, optimizer, lr_scheduler)
    data_loader_train, _, _ = dataloader_initializer(args)

    model.train()
    criterion.train()
    samples, targets = next(iter(data_loader_train))
    samples = samples.to(device)
    targets = [target_to_cuda(t) for t in targets]

    outputs, targets, *_ = model(samples, targets)
    loss_dict = criterion(outputs, targets)
    losses = sum(loss_dict[k] * criterion.weight_dict[k]
                 for k in loss_dict if k in criterion.weight_dict)
    is_loss_invalid(losses)
    losses.backward()
    print(f"  one real forward/backward OK -- loss {losses.item():.4f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs CUDA: a device mismatch cannot be reproduced on CPU only")
    device = torch.device('cuda')

    argv = sys.argv[1:]
    real = '--real' in argv
    if real:
        argv.remove('--real')

    print("[synthetic] matcher / add_track_queries_to_targets / criterion")
    run_synthetic(device)

    if real:
        print("[real] one batch from the config")
        run_real(argv)

    print("OK")
