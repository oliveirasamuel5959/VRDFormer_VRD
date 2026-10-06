# VRDFormer Stage 1 — vidorsmall (8x smaller VidOR subset)
# Run from the repo root. Set --nproc_per_node to your GPU count, or run the
# script as-is for a single-GPU run.
python -m torch.distributed.launch \
    --master_port 47749 \
    --nproc_per_node=${NPROC_PER_NODE:-1} \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidorsmall_stage1.json
