# VRDFormer Stage 2 — RunPod 4x RTX 4090 DDP (relative paths)
# Requires Stage 1 checkpoint at data/ckpts/vidor_stage1/checkpoint0000.pth
python -m torch.distributed.launch \
    --master_port 47745 \
    --nproc_per_node=4 \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidor_runpod_stage2.json
