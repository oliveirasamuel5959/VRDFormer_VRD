# Installation

## Modern RunPod Stage-1 setup (recommended)

For the validated Python 3.12 / PyTorch 2.8 + CUDA 12.8 environment, use the single-run installer and checks in [`RUNPOD_STAGE1.md`](RUNPOD_STAGE1.md). It installs [`requirements-runpod-stage1.txt`](../requirements-runpod-stage1.txt) and preserves the PyTorch build from the RunPod image. Do **not** install the root `requirements.txt` in this environment: it contains historical Python 3.7 pins.

The optional MultiScaleDeformableAttention extension is needed only for deformable configs. Standard configs such as `configs/vidorsmall_stage1.json` use `num_feature_levels: 1` and do not need that extension.

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
there. The root `requirements.txt` is retained as that historical pin list; it is not the
RunPod/Python 3.12 dependency set.

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
