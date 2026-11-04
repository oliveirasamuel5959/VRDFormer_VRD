# VRDFormer Stage 2 evaluation — vidorsmall
# Requires the Stage 2 checkpoint from training (data/ckpts/vidorsmall_stage2/checkpoint.pth).
# --num_queries must match the trained model: the checkpoints store 200-query embeddings, and
# without this flag argparse builds a 100-query model and the load fails with a size mismatch.
python -m torch.distributed.launch \
    --master_port 47746 \
    --nproc_per_node=1 \
    main.py \
    --eval \
    --num_queries 200 \
    --dataset_config configs/vidorsmall_stage2.json \
    --resume data/ckpts/vidorsmall_stage2/checkpoint.pth
