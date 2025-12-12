# E-RayZer
This repository contains the official implementation for **Self-supervised 3D Reconstruction as Spatial Visual Pre-training**.

[Project Page](https://qitaozhao.github.io/E-RayZer) · arXiv (coming soon) · Local Gradio demo (this repo)

![teaser](https://raw.githubusercontent.com/QitaoZhao/QitaoZhao.github.io/main/research/E-RayZer/images/erayzer_teaser.png)

## Repository Map
- `gradio_app.py`: End-to-end demo UI backed by the inference engine.
- `app_core/engine.py`: Thin wrapper that loads configs, checkpoints, and exports renders, Gaussian point clouds, and turntable videos.
- `erayzer_core/`: Model, transformer blocks, and Gaussian renderer used at inference time.
- `config/erayzer.yaml`: Default inference configuration (image sizes, number of views, view selector, transformer depth, etc.).
- `checkpoints/`: Pretrained weights (`erayzer_multi.pt`, `erayzer_dl3dv.pt`).
- `examples/`: Five curated multi-view sets for quick validation.
- `third_party/gsplat/`: Vendored differentiable Gaussian splatting ops that must be installed in editable mode.

## Quick Start

### 1. Create the environment
```bash
conda create -n erayzer python=3.10 -y
conda activate erayzer

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
		--index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e third_party/gsplat/
```
- CUDA 12.1 GPUs with ≥24 GB memory are recommended for the default 10-view setting. The demo automatically falls back to CPU (slower) if CUDA is unavailable.
- `imageio[ffmpeg]` pulls the ffmpeg backend used for MP4 export. Install the system-level `ffmpeg` binary if `imageio` cannot find one on your platform.

### 2. Download or place checkpoints
- `checkpoints/erayzer_multi.pt`: Multi-dataset model (default for the demo).
- `checkpoints/erayzer_dl3dv.pt`: Model finetuned on DL3DV sequences for maximum fidelity on that dataset.

`gradio_app.py` automatically fetches `erayzer_multi.pt` from [Hugging Face](https://huggingface.co/qitaoz/E-RayZer/blob/main/checkpoints/erayzer_multi.pt) the first time you run it (cached under your Hugging Face cache directory). You can still drop pre-downloaded weights in `checkpoints/` and/or pass `--ckpt /path/to/weights.pt` to point at a custom file.

### 3. Launch the Gradio app
```bash
python gradio_app.py \
	--config config/erayzer.yaml \
	--device cuda:0 \
	--output-dir outputs \
	--share
```
- Upload 10-ish multi-view RGB images (the engine pads/repeats if you provide fewer). You can also click one of the bundled examples to preload sample views.
- Press **Run Inference**. The demo writes predicted camera poses, target renders, a Gaussian point cloud (`point_cloud.glb`), a sweep video (`render_video.mp4`), and a zipped archive under `outputs/<timestamp>/`.
- `--share` exposes the interface through Gradio tunnels; drop the flag to keep it local.

## Citation
If you use E-RayZer in academic or industrial research, please cite:

```bibtex
@inproceedings{zhao2026erayzer,
	title     = {E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training},
	author    = {Qitao Zhao and Hao Tan and Qianqian Wang and Sai Bi and Kai Zhang and Kalyan Sunkavalli and Shubham Tulsiani and Hanwen Jiang},
	booktitle = {arXiv},
	year      = {2026},
	note      = {arXiv link coming soon}
}
```

## Acknowledgements & License
- Model weights (and related assets) are Copyright 2025 Adobe Inc. and are distributed under the Adobe Research License.