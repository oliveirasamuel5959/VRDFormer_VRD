# VRDFormer Stage 1 evaluation — vidorsmall
# Requires the Stage 1 checkpoint from training (data/ckpts/vidorsmall_stage1/checkpoint.pth
# is the final epoch, overwritten every epoch; use checkpointNNNN.pth to score an earlier one).
#
# --num_queries must match the trained model: the checkpoints store 200-query embeddings, and
# without this flag argparse builds a 100-query model and the load fails with a size mismatch.
#
# Scores box AP@0.5, pair AP@0.5 and detections-per-GT-object on the val split. That last number
# is the duplicate-box metric described in docs/STAGE1.md §7.1 — 1.0 is ideal.
python -m torch.distributed.launch \
    --master_port 47747 \
    --nproc_per_node=1 \
    main.py \
    --eval \
    --num_queries 200 \
    --dataset_config configs/vidorsmall_stage1.json \
    --resume "${STAGE1_CKPT:-data/ckpts/vidorsmall_stage1/checkpoint.pth}"
