# VRDFormer Stage 2 — vidorsmall (8x smaller VidOR subset)
# Requires the Stage 1 checkpoint at data/ckpts/vidorsmall_stage1/checkpoint0004.pth
# (path is set in configs/vidorsmall_stage2.json as "pretrain").
python -m torch.distributed.launch \
    --master_port 47745 \
    --nproc_per_node=${NPROC_PER_NODE:-1} \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidorsmall_stage2.json
