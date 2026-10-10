# Installation

## Modern environment (Python 3.10+, RunPod / CUDA 12) — recommended

RunPod PyTorch images already provide a CUDA-enabled `torch` / `torchvision`. Do **not**
install PyTorch yourself; install only the extra runtime dependencies.

1. Clone and enter this repository:
    ```
    git clone https://github.com/zhengsipeng/VRDFormer_VRD.git
    cd VRDFormer_VRD
    ```

2. Install the dependencies (`decord`, `timm`, `scipy`, `lap`, `opencv-python-headless`,
   `pycocotools`, ...). Run from the repository root:
    ```
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    ```
    `timm` depends on `torch` / `torchvision`; because the image already provides a
    compatible CUDA build, pip leaves them in place instead of reinstalling.

    Single copy-paste block for a fresh pod:
    ```bash
    cd VRDFormer_VRD
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    ```

3. Verify (expect the image's `torch 2.x`, `cuda True` and `gpus 2` on a 2-GPU pod):
    ```
    python -c "import torch, torchvision, decord, timm, scipy, lap, pycocotools, cv2, numpy; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
    ```

4. (Optional) Install the MultiScaleDeformableAttention extension. Only the `*_deform*.json`
   configs use it; the default vidorsmall / vidor / vidvrd configs set `num_feature_levels: 1`
   and do not need it:
    ```
    cd models/ops && python setup.py build install && cd ../..
    ```

5. Train on 2 GPUs with the modern launcher:
    ```
    torchrun --nproc_per_node=2 --master_port 47749 main.py \
        --accumulate_steps 1 --lr_backbone 1e-5 --lr 5e-5 --num_queries 200 \
        --dataset_config configs/vidorsmall_stage1.json
    ```

## Legacy environment (Python 3.7, CUDA 11.1) — historical

The original codebase targeted Python 3.7 and PyTorch 1.10. This only works on an old
image; RunPod images no longer ship Python 3.7.

1. Build tools (`lap` compiles from source and needs `pkg_resources`):
    ```
    pip install -U pip setuptools wheel Cython
    ```
2. Install the frozen legacy pins (`decord`, `timm`, `scipy`, `lap`, `opencv-python`,
   `numpy==1.18.5`, ...):
    ```
    pip install -r docs/requirements-legacy.txt
    ```
3. Install PyTorch 1.10+cu111, torchvision 0.11 and the matching torchaudio from the
   PyTorch wheel index (see [previous versions](https://pytorch.org/get-started/previous-versions/#v150)):
    ```
    pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
4. Install pycocotools (TrackFormer fork with the fixed ignore flag):
    ```
    pip install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'
    ```
5. Install the MultiScaleDeformableAttention package:
    ```
    cd models/ops && python setup.py build install && cd ../..
    ```

or, install packages for Python 3.8 (deformable-detr is not allowed):
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Note: the legacy set requires **Python 3.7** — `numpy==1.18.5`, `scipy==1.4.1`,
`pandas==1.0.5` and `Pillow==7.1.2` have no wheels for Python 3.10+ and will not build
there. The modern set is kept in `requirements.txt` / `docs/requirements.txt`; the frozen
legacy pin list is kept in `docs/requirements-legacy.txt`.

# Old Version (abandon)
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI & make & make install

python src/models/ops/setup.py build --build-base=src/vrdformer/models/ops install
cp models/resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#AT_DISPATCH_FLOATING_TYPES_AND_HALF
#AT_DISPATCH_FLOATING_TYPES=
