python inference.py \
    --mode stage1 \
    --output_dir output/inference/stage1/$(date +%Y%m%d_%H%M%S)/ \
    --video data/vidorsmall/videos/2440057130.mp4 \
    --max_frames 20 \
    --stage1_ckpt data/ckpts/vidorsmall_stage1/checkpoint0001.pth \
    --stage1_config configs/vidorsmall_stage1.json \
    --dataset vidorsmall \
    --device cpu
