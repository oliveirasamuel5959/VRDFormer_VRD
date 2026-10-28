python inference.py \
    --mode stage1 \
    --output_dir output/inference/stage1/$(date +%Y%m%d_%H%M%S)/ \
    --video ./Normal_Videos_758_x264.mp4 \
    --max_frames 60 \
    --stage1_ckpt data/ckpts/20261024/vidorsmall_stage1/checkpoint0003.pth \
    --stage1_config configs/vidorsmall_stage1.json \
    --dataset vidorsmall \
    --device cpu
