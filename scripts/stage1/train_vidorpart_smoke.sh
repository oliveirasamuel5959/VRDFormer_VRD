# VRDFormer Phase 3.1 — Local Smoke Test (100 videos, 2 epochs)
# Quick verification that the pipeline works before full Colab training.
# Must be run from the repo root.
python -m torch.distributed.launch \
    --master_port 47749 \
    --nproc_per_node=1 \
    main.py \
    --accumulate_steps 1 \
    --lr_backbone 1e-5 \
    --lr 5e-5 \
    --num_queries 200 \
    --dataset_config configs/vidorpart_local_stage1.json \
    --epochs 2
