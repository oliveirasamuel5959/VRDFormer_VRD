import torch
import math
import datetime
import numpy as np
from collections import defaultdict
from types import SimpleNamespace
import util.misc as utils
from util.misc import NestedTensor, target_to_cuda
from util.dist import reduce_dict, get_rank, print_all_ranks, is_main_process, get_world_size
from util.box_ops import debug_and_vis, box_cxcywh_to_xyxy, box_iou
from util.compute import voc_ap
from util.evaluate import evaluate


def is_loss_invalid(loss, synced_value=None):
    """Abort the whole run if the loss is not finite.

    Args:
        loss: this rank's local loss scalar, before the accumulate_steps division.
        synced_value: the same loss after ``reduce_dict``, i.e. averaged over all ranks.

    A non-finite loss on any one rank survives ``reduce_dict``'s all_reduce (NaN/inf
    propagate through a sum), so passing that reduced value lets *every* rank detect it
    and raise together. The previous version raised/``sys.exit(1)``-ed on the offending
    rank alone; because ``setup_for_distributed`` replaces ``print`` with a no-op on
    ranks != 0, that rank vanished silently and the others stayed blocked forever in the
    next collective. Raising on all ranks lets the launcher tear the job down loudly.
    """
    local = loss.item()
    bad = not math.isfinite(local)
    if synced_value is not None and not math.isfinite(synced_value):
        bad = True
    if not bad:
        return
    # print_all_ranks escapes setup_for_distributed's suppression of output on ranks != 0.
    print_all_ranks(f"[rank {get_rank()}] non-finite loss at {datetime.datetime.now()}: "
                    f"local={local!r} global={synced_value!r} -- aborting all ranks")
    raise RuntimeError(
        f"non-finite loss on rank {get_rank()}: local={local}, global={synced_value}")


def train_stage2(model, criterion, data_loader, optimizer, device, epoch, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=None,
        debug=args.debug)

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('sub_class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('verb_class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

    optimizer.zero_grad()

    # Automatic mixed precision: bf16 on Ampere+ GPUs (no loss scaling needed),
    # fp16 with GradScaler elsewhere.
    amp_enabled = getattr(args, 'amp', False)
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype != torch.bfloat16)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch, batch_size=args.batch_size*args.accumulate_steps)):
        # if args.debug:
        #     debug_and_vis(args.datasets, samples, targets, i)

        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        samples = samples.to(device)  # b,t,3,h,w
        # targets: list over clips; each clip -> list of per-frame target dicts
        targets = [[target_to_cuda(t) for t in clip_targets] for clip_targets in targets]
        B = len(targets)

        # given annos of boxes, predicate class of {s,p,o}
        # we don't use the rec-query and initialize static-query by given boxes

        memory = None
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            for fid in range(args.seq_len):
                cur_frame = samples.select_frame(fid)  # b,3,H,W
                frame_targets = [clip_targets[fid] for clip_targets in targets]
                memory = model(cur_frame,
                               frame_targets,
                               memory,
                               eos=(fid+1)==args.seq_len
                            )

            # sum per-clip losses -> preserves bs1 gradient scale (no LR change needed)
            loss_dicts = [criterion(m) for m in memory]
            loss_dict = {k: sum(d[k] for d in loss_dicts) for k in loss_dicts[0]}

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # Checked *after* reduce_dict: a non-finite loss on one rank survives the
        # all_reduce, so all ranks observe it and abort together instead of one rank
        # exiting while the rest block forever in the next collective.
        is_loss_invalid(losses, synced_value=loss_value)

        losses = losses / args.accumulate_steps
        if scaler.is_enabled():
            scaler.scale(losses).backward()
        else:
            losses.backward()
        if (i+1) % args.accumulate_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(sub_class_error=loss_dict_reduced['sub_class_error'])
        metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(verb_class_error=loss_dict_reduced['verb_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"],
                             lr_backbone=optimizer.param_groups[1]["lr"])

    if (i+1) % args.accumulate_steps != 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_stage1(model, criterion, data_loader, optimizer, device, epoch, args):
    #vis_iter_metrics = None
    #if visualizers:
    #    vis_iter_metrics = visualizers['iter_metrics']
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=None,
        debug=args.debug)

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('sub_class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('verb_class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

    optimizer.zero_grad()
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch, batch_size=args.batch_size*args.accumulate_steps)):
        # if args.debug:
        #     debug_and_vis(args.datasets, samples, targets, i) 
        samples = samples.to(device)  # bs,3,h,w
        targets = [target_to_cuda(t) for t in targets]
        
        #import pdb;pdb.set_trace()
        # samples [2,3,H,W]
        # targets: [xxx, xxx]
        # 'boxes', 'labels', 'image_id', 'track_ids', 'area', 'iscrowd', 
        # 'orig_size', 'size', 'labels_ignore', 'area_ignore', 
        # 'iscrowd_ignore', 'boxes_ignore', 'track_ids_ignore', 
        # 'prev_image', 'prev_target'
        
        # in order to be able to modify targets inside the forward call we need
        # to pass it through as torch.nn.parallel.DistributedDataParallel only
        # passes copies
        outputs, targets, *_ = model(samples, targets)
        #import pdb;pdb.set_trace()
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # Same group-wide abort as train_stage2: one rank exiting alone leaves the rest
        # blocked in the next collective, with its message hidden by
        # setup_for_distributed on ranks != 0.
        is_loss_invalid(losses, synced_value=loss_value)

        losses = losses / args.accumulate_steps
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            
        if (i+1) % args.accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(sub_class_error=loss_dict_reduced['sub_class_error'])
        metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(verb_class_error=loss_dict_reduced['verb_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"],
                             lr_backbone=optimizer.param_groups[1]["lr"])
        '''
        if visualizers and (i == 0 or not i % args.vis_and_log_interval):
            _, results = make_results(
                outputs, targets, postprocessors, args.tracking, return_only_orig=False)

            vis_results(
                visualizers['example_results'],
                samples.unmasked_tensor(0),
                results[0],
                targets[0],
                args.tracking)
        '''
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

def eval_stage2(model, val_loader, device, epoch, args):
    model.eval()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=None,
        debug=args.debug)
    
    with open("data/%s/action.txt"%args.dataset, "r") as f:
        action_list = [l.strip() for l in f.readlines()]
    action_dict = dict(zip(range(len(action_list)), action_list))
    val_dataset = val_loader.dataset
    groundtruth = dict()
    prediction = dict()
    amp_enabled = getattr(args, 'amp', False)
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    for i, (samples, targets) in enumerate(metric_logger.log_every(val_loader, epoch)):
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        samples = samples.to(device)
        targets = [target_to_cuda(t) for t in targets[0]]
        video_id = targets[0]['video_id']

        groundtruth[video_id] = targets[0]['groundtruth']
        frame_ids = [int(target['frame_id']) for target in targets]

        memory = None
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            for fid in range(len(frame_ids)):
                cur_frame = samples.select_frame(fid)  # 1,3,H,W,
                memory = model(cur_frame,
                               [targets[fid]],
                               memory,
                               eos=(fid+1)==len(frame_ids),
                               is_eval=True
                            )
            model_without_ddp = model.module if hasattr(model, 'module') else model
            preds, scores = model_without_ddp.relation_classifier(memory[0], gt=targets[0]['groundtruth'], is_eval=True)
        prediction[video_id] = []
        for j, pred in enumerate(preds):
            prediction[video_id].append({'triplet': (groundtruth[video_id][j]['triplet'][0], action_dict[pred], groundtruth[video_id][j]['triplet'][2]),
                                         'score': scores[j]
                                        })
        print("[info] Video %d"%i)
    scores = evaluate(groundtruth, prediction, val_dataset)
    print(scores)
    return scores


def _xyxy_ious(boxes_a, boxes_b):
    """Pairwise IoU as an (N, M) numpy array; zeros when either side is empty."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    iou, _ = box_iou(torch.as_tensor(boxes_a, dtype=torch.float32),
                     torch.as_tensor(boxes_b, dtype=torch.float32))
    return torch.nan_to_num(iou).numpy()


def _greedy_ap(num_gt, scores, hits):
    """AP over one class's ranked detections, each GT matchable once (VOC convention).

    ``hits`` holds one entry per detection: the index of the matched ground truth, or None.
    """
    order = np.argsort(-np.asarray(scores), kind="stable")
    matched = set()
    tp = np.zeros(len(order), dtype=np.float32)
    fp = np.zeros(len(order), dtype=np.float32)
    for rank, det_i in enumerate(order):
        gt_i = hits[det_i]
        if gt_i is not None and gt_i not in matched:
            matched.add(gt_i)
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0
    if num_gt == 0:
        return None
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / num_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return float(voc_ap(rec, prec))


@torch.no_grad()
def eval_stage1(model, val_loader, device, epoch, args, iou_thresh=0.5, max_videos=None):
    """Stage-1 detection metrics on the val split.

    Reports three numbers, each pooled over every scored frame:

    * ``box_ap`` -- AP@0.5 per object class and role, macro-averaged.
    * ``pair_ap`` -- AP@0.5 over (sub_cls, obj_cls) groups, a detection correct only when
      *both* boxes clear the threshold.
    * ``dups_per_gt`` -- detections matching each ground-truth object @0.5, averaged. 1.0 is
      ideal; higher means the model emits several boxes for one object, which is the failure
      this metric exists to track.

    Detection decoding goes through ``vrdf_inference.stage1.decode_frame`` so the numbers
    describe exactly the detections inference produces, not a second implementation of them.
    """
    from models.vrdformer import VRDFormer
    from vrdf_inference.stage1 import decode_frame

    model.eval()
    model_without_ddp = model.module if hasattr(model, "module") else model
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval, delimiter="  ", vis=None, debug=args.debug)

    # decode_frame reads its thresholds off args; training args do not carry them, and the
    # inference defaults are the calibrated ones (see docs/RUNPOD_INFERENCE.md).
    decode_args = SimpleNamespace(
        conf_sub=getattr(args, "conf_sub", 0.15),
        conf_obj=getattr(args, "conf_obj", 0.15),
        dedup_iou=getattr(args, "dedup_iou", 0.9),
        max_dets_per_frame=getattr(args, "max_dets_per_frame", 50),
    )

    # per (role, class) ranked detection records, each (score, matched_gt_or_None). Match
    # keys are global (frame-scoped offset) so an index from one frame can never be mistaken
    # for the same object in another.
    box_records = defaultdict(list)
    gt_counts = defaultdict(int)
    pair_records = defaultdict(list)
    pair_gt_counts = defaultdict(int)
    dups_per_gt = []
    gt_key_base = 0

    num_videos = len(val_loader)
    if max_videos is not None:
        num_videos = min(num_videos, max_videos)
    for i, (samples, targets) in enumerate(metric_logger.log_every(val_loader, epoch)):
        if i >= num_videos:
            break
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        samples = samples.to(device)
        frame_targets = targets[0]

        prev_features = None
        for fid, target in enumerate(frame_targets):
            cur_frame = samples.select_frame(fid)
            # Call the base VRDFormer.forward, not the tracking wrapper: the wrapper needs
            # per-frame targets (to build track queries) and has no `stage` kwarg. This is
            # the same path single-video inference takes (vrdf_inference/stage1.py:158).
            out, _, prev_features, _, _ = VRDFormer.forward(
                model_without_ddp, cur_frame, targets=None, prev_features=prev_features)
            dets, _ = decode_frame(out, (samples.tensors.shape[-2], samples.tensors.shape[-1]),
                                   (samples.tensors.shape[-2], samples.tensors.shape[-1]),
                                   decode_args)

            gt_sub = box_cxcywh_to_xyxy(target["sub_boxes"]).numpy()
            gt_obj = box_cxcywh_to_xyxy(target["obj_boxes"]).numpy()
            gt_sub_cls = target["sub_labels"].numpy()
            gt_obj_cls = target["obj_labels"].numpy()
            det_sub = np.stack([d["box_n_sub"] for d in dets]) if dets else np.zeros((0, 4))
            det_obj = np.stack([d["box_n_obj"] for d in dets]) if dets else np.zeros((0, 4))

            for role, det_boxes, gt_boxes, gt_cls, det_cls, det_score in (
                ("sub", det_sub, gt_sub, gt_sub_cls,
                 np.array([d["sub_cls"] for d in dets], dtype=int),
                 np.array([d["sub_score"] for d in dets])),
                ("obj", det_obj, gt_obj, gt_obj_cls,
                 np.array([d["obj_cls"] for d in dets], dtype=int),
                 np.array([d["obj_score"] for d in dets])),
            ):
                ious = _xyxy_ious(det_boxes, gt_boxes)
                for c in np.unique(gt_cls):
                    gt_counts[(role, int(c))] += int((gt_cls == c).sum())
                for d_i in range(len(det_boxes)):
                    same_class = [g for g in range(len(gt_boxes)) if gt_cls[g] == det_cls[d_i]]
                    best = max(same_class, key=lambda g: ious[d_i, g]) if same_class else None
                    hit = (gt_key_base + best if best is not None
                           and ious[d_i, best] >= iou_thresh else None)
                    box_records[(role, int(det_cls[d_i]))].append((float(det_score[d_i]), hit))

            # pair metrics: group by (sub_cls, obj_cls)
            for c in set(zip(gt_sub_cls.tolist(), gt_obj_cls.tolist())):
                pair_gt_counts[c] += int(((gt_sub_cls == c[0]) & (gt_obj_cls == c[1])).sum())
            iou_sub = _xyxy_ious(det_sub, gt_sub)
            iou_obj = _xyxy_ious(det_obj, gt_obj)
            if len(dets):
                for d_i, d in enumerate(dets):
                    grp = (d["sub_cls"], d["obj_cls"])
                    cand = [g for g in range(len(gt_sub))
                            if (gt_sub_cls[g], gt_obj_cls[g]) == grp]
                    hit = None
                    if cand:
                        best = max(cand, key=lambda g: min(iou_sub[d_i, g], iou_obj[d_i, g]))
                        if min(iou_sub[d_i, best], iou_obj[d_i, best]) >= iou_thresh:
                            hit = gt_key_base + best
                    pair_records[grp].append((float(d["pair_score"]), hit))

            # duplicates per GT object, measured on subject boxes. Frames where the model
            # found nothing still contribute a 0 per GT object -- skipping them would
            # report the duplicates-per-detected-object instead, which is not the metric.
            gt_iou = _xyxy_ious(gt_sub, gt_sub)
            for g in range(len(gt_sub)):
                same = np.where(gt_sub_cls == gt_sub_cls[g])[0]
                # a GT object is a distinct box within its class: IoU0.9 merges relation
                # instances that share an object
                is_rep = all(g <= h or gt_iou[g, h] < 0.9 for h in same)
                if not is_rep:
                    continue
                dups_per_gt.append(int((iou_sub[:, g] >= iou_thresh).sum()))
            gt_key_base += len(gt_sub)

        if i % 10 == 0:
            print(f"[eval_stage1] video {i + 1}/{num_videos} ({targets[0][0]['video_id']})",
                  flush=True)

    def summarize(records, counts):
        aps = []
        for key, recs in records.items():
            ap = _greedy_ap(counts.get(key, 0), [s for s, _ in recs],
                            [h for _, h in recs])
            if ap is not None:
                aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    # Under DDP each rank scores only its shard (the val loader uses a DistributedSampler),
    # so pool the per-frame records before computing the metrics.
    payload = (dict(box_records), dict(gt_counts), dict(pair_records),
               dict(pair_gt_counts), dups_per_gt)
    if utils.is_dist_avail_and_initialized() and get_world_size() > 1:
        import torch.distributed as dist
        gathered = [None] * get_world_size()
        dist.all_gather_object(gathered, payload)
        box_records, gt_counts, pair_records, pair_gt_counts, dups_per_gt = (
            defaultdict(list), defaultdict(int), defaultdict(list), defaultdict(int), [])
        for g_box, g_gcount, g_pair, g_pcount, g_dups in gathered:
            for k, v in g_box.items():
                box_records[k].extend(v)
            for k, v in g_gcount.items():
                gt_counts[k] += v
            for k, v in g_pair.items():
                pair_records[k].extend(v)
            for k, v in g_pcount.items():
                pair_gt_counts[k] += v
            dups_per_gt.extend(g_dups)

    scores = {
        "stage1_box_ap": summarize(box_records, gt_counts),
        "stage1_pair_ap": summarize(pair_records, pair_gt_counts),
        "stage1_dups_per_gt": float(np.mean(dups_per_gt)) if dups_per_gt else 0.0,
        "stage1_num_gt_objects": len(dups_per_gt),
    }
    metric_logger.synchronize_between_processes()
    if is_main_process():
        print("[eval_stage1] " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()), flush=True)
    return scores
