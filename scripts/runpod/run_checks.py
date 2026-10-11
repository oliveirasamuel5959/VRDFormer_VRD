import os
import torch

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print()

checks = [
    ('data/vidorsmall/videos', 'dir'),
    ('data/vidorsmall/annotations/train', 'dir'),
    ('data/vidorsmall/annotations/val', 'dir'),
    ('data/vidorsmall/action.txt', 'file'),
    ('data/metadata/vidorsmall_annotations.pkl', 'file'),
    ('data/metadata/vidorsmall_train_frames_stage1.json', 'file'),
    ('data/metadata/vidorsmall_train_frames_stage2.json', 'file'),
    ('data/metadata/vidorsmall_val_frames.json', 'file'),
    ('data/weights/detr-r101-2c7b67e5.pth', 'file'),
    ('configs/vidorsmall_stage1.json', 'file'),
    ('configs/vidorsmall_stage2.json', 'file'),
]

all_ok = True
for path, kind in checks:
    exists = os.path.isdir(path) if kind == 'dir' else os.path.isfile(path)
    size = f' ({os.path.getsize(path) / 1e6:.1f} MB)' if exists and kind == 'file' else ''
    print(f'  {"OK" if exists else "MISSING"} {path}{size}')
    all_ok = all_ok and exists

if all_ok:
    n_train = len(os.listdir('data/vidorsmall/annotations/train'))
    n_val = len(os.listdir('data/vidorsmall/annotations/val'))
    print(f'\nAll checks passed - {n_train} train / {n_val} val videos. Ready to train!')
else:
    print('\nSome files are missing - check the Kaggle download.')
