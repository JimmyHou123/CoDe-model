# CoDe: A Self-Supervised Consistency Model Framework for MRI Denoising | ISBI 2026

## Overview

**CoDe** is a self-supervised consistency model framework for MRI denoising. It is designed to denoise MRI data without paired clean ground-truth images. The current implementation focuses on diffusion MRI and uses a two-stage pipeline.

- **Stage I:** train a noise estimation model from independent noisy measurements.
- **Stage II:** train a consistency model for fast one-step denoising.
- **Optional RMT/MPPCA:** generate an RMT-denoised reference for Stage II regularization.

---

## Architecture

<!-- Insert architecture figure here. Suggested path: assets/code_architecture.png or assets/code_architecture.pdf -->

<p align="center">
  <img src="assets/code_architecture.png" width="850">
</p>

<p align="center">
  <em>Figure 1. CoDe architecture. Stage I estimates an approximated clean image from independent noisy inputs. Stage II refines the result using consistency-model denoising and RMT regularization.</em>
</p>

---

## Results

<!-- Insert qualitative result figure here. Suggested path: assets/code_results.png or assets/code_results.pdf -->

<p align="center">
  <img src="assets/code_results.png" width="850">
</p>

<p align="center">
  <em>Figure 2. Qualitative denoising results. CoDe reduces noise while preserving anatomical details compared with baseline methods.</em>
</p>

---

## Repository Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CoDe-model-main.git
cd CoDe-model-main
```

Create the environment:

```bash
conda create -n code python=3.10 -y
conda activate code
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib opencv-python nibabel dipy tqdm blobfile mpi4py piq torchkde
```

Check the setup:

```bash
python -c "import torch, nibabel, dipy; print('Environment OK')"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Data Format

The input should be a 4D diffusion MRI NIfTI file:

```text
H × W × S × V
```

Example:

```text
dwi_data/tractoinferno_sub1006/sub-1006__dwi.nii.gz
```

---

## Basic Training and Testing Commands

The full pipeline is:

```text
RMT preprocessing → Stage I training → Stage II training → Testing
```

### 1. Run RMT / MPPCA preprocessing

```bash
bash run_rmt.sh
```

This generates the RMT reference file, usually:

```text
/path/to/save/rmt/denoised_mppca.npy
```

Skip this step if you do not use RMT regularization.

### 2. Train Stage I noise estimation model

```bash
bash run_stage1.sh
```

Expected checkpoint:

```text
/path/to/save/stage1/model150000.pt
```

### 3. Train Stage II consistency model

```bash
bash run_stage2.sh
```

Expected checkpoint:

```text
/path/to/save/stage2/target_model300000.pt
```

### 4. Test / run inference

```bash
bash run_test.sh
```

The denoised 4D NIfTI output will be saved under the directory specified by `--save_dir` in `run_test.sh`.

---

## Run Everything with One Command

Create a `run_all.sh` file:

```bash
#!/bin/bash
set -e

bash run_rmt.sh
bash run_stage1.sh
bash run_stage2.sh
bash run_test.sh
```

Run:

```bash
chmod +x run_all.sh
bash run_all.sh
```

---

## Notes

- Before running, replace all placeholder paths in the `.sh` files with real paths.
- Use `target_model300000.pt` for testing.
- If using `--loss_norm lpips_rmt`, make sure `--rmt_npy_path` points to a valid `denoised_mppca.npy` file.
- If not using RMT regularization, change `--loss_norm lpips_rmt` to `--loss_norm lpips`, `l1`, or `l2`, and remove `--rmt_npy_path`.
- To select a GPU, run commands with `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_stage2.sh
```

---

## Citation

```bibtex
@inproceedings{code2026,
  title     = {CoDe: A Self-Supervised Consistency Model Framework for MRI Denoising},
  author    = {Li, Junying and Hou, Qingyang and Pang, Kaifeng and Miao, Qi and Hung, Alex Ling Yu and Aygun, Elif and Shih, Shu-Fu and Dai, Qing and Wu, Holden H. and Sung, Kyunghyun},
  booktitle = {IEEE International Symposium on Biomedical Imaging},
  year      = {2026}
}
```

## License

This repository is released under the MIT License.
