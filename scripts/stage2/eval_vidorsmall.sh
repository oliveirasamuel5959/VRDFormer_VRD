# VRDFormer Stage 2 evaluation — vidorsmall
# Requires the Stage 2 checkpoint from training (data/ckpts/vidorsmall_stage2/checkpoint.pth).
# Stage 2 only; stage 1 has no eval path (see CLAUDE.md).
python -m torch.distributed.launch \
    --master_port 47746 \
    --nproc_per_node=1 \
    main.py \
    --eval \
    --dataset_config configs/vidorsmall_stage2.json \
    --resume data/ckpts/vidorsmall_stage2/checkpoint.pth
