# VRDFormer Stage 2 — vidorsmall (8x smaller VidOR subset)
# Initialises from the Stage 1 checkpoint named by "pretrain" in configs/vidorsmall_stage2.json
# (default data/ckpts/vidorsmall_stage1/checkpoint.pth — the final Stage-1 epoch). Edit that
# key if your Stage-1 run wrote to a different output_dir, or to pick a different epoch using
# the stage-1 evaluator's dups_per_gt / box_ap (docs/STAGE1.md §7.1).
#
# Stage 2 keeps the autograd graph alive across all seq_len frames, so peak memory per
# GPU scales with batch_size * seq_len; the config uses batch_size 1 to fit. We keep an
# effective batch of ~4 clips/optimizer step by scaling gradient accumulation with the
# number of processes (NPROC=1 -> 4, NPROC=2 -> 2, NPROC>=4 -> 1).
NPROC=${NPROC_PER_NODE:-4}
ACC=$(( 4 / NPROC ))
[ "$ACC" -lt 1 ] && ACC=1

python -m torch.distributed.launch \
    --master_port 47745 \
    --nproc_per_node=${NPROC} \
    main.py \
    --accumulate_steps ${ACC} \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidorsmall_stage2.json
