python inference.py \
    --mode stage1 \
    --output_dir output/inference/stage1/$(date +%Y%m%d_%H%M%S)/ \
    --video data/vidorsmall/videos/2440057130.mp4 \
    --max_frames 60 \
    --stage1_ckpt data/ckpts/20260324/vidorsmall_stage1/checkpoint0001.pth \
    --stage1_config configs/vidorsmall_stage1.json \
    --dataset vidorsmall \
    --device cpu
