# SD Card Printed-Defect Detection with VAE

> 繁體中文：這是一份歷史專題整理版，保存以 VAE 與 SSIM 進行 SD 卡印刷瑕疵分析的程式、少量可展示影像與既有輸出。它不是重新訓練或已驗證可重現的發布版本。

## Project overview

This repository preserves a student project for image-based SD card printed-defect analysis. The implementation learns reconstructions from defect-free images with a Variational Autoencoder (VAE), then compares an input image and its reconstruction with reconstruction error and Structural Similarity Index (SSIM). Some scripts additionally identify candidate regions from difference images and evaluate local regions.

## What is included

- Single-card and multi-region historical Python implementations.
- Small, selected image samples for repository presentation only.
- Selected historical result images and SSIM/error logs in the original source material.
- Weight-file inventory and SHA-256 checksums; the actual large weight files are retained locally and pending Git LFS publication.

## What is not included

- The full original image dataset.
- Unselected result images, archives, installer files, unrelated coursework, travel records, or competition submissions.
- Personal names, advisor information, partner-company information, and production-site details.
- A claim that the included code has been reproduced in the current environment.

## Method summary

1. Convert an SD card image to grayscale and resize it to the script-specific input size.
2. Apply image filtering used by the historical scripts.
3. Reconstruct the image using a VAE trained on defect-free images.
4. Compare original and reconstructed images with SSIM and reconstruction error.
5. In the multi-region variants, inspect difference-image regions for a second local comparison.

The primary scripts use 256 x 350 grayscale inputs and a latent dimension of 160. The exact behavior, thresholds, and preprocessing differ among historical script variants; see `docs/project-history.md` before choosing one.

## Repository layout

```text
src/
  single_card/              # historical single-card training, batch, and camera scripts
  multi_region/             # historical four-region scripts
  multi_region_camera_v2/   # later top-level multi-region variant
data/samples/               # selected presentation samples, not the full dataset
results/examples/           # selected historical output images
weights/                    # HDF5 weights, intended for Git LFS
docs/                       # provenance, scope, and reproducibility notes
```

## Environment and dependencies

The original environment lockfile was not preserved. `requirements.txt` lists packages inferred from imports, without version pins. Create a clean Python environment and install them before attempting to run the scripts.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

GPU, camera driver, TensorFlow version, and CUDA compatibility were not recorded. Camera scripts also require a compatible local camera and may use a Windows-specific capture backend.

## Data and weights

The `data/samples/` directory contains only small selected samples for illustration. It must not be treated as a training or evaluation dataset. See `data/README.md`.

Weights are HDF5 files and are large. Git LFS is required after cloning:

```bash
git lfs install
git lfs pull
```

See `weights/README.md` for file-to-script associations and SHA-256 checksums.

## How to run

This repository is a preservation release, not a verified reproducibility release. Start by reading the selected script and its expected current working directory:

```bash
python "src/single_card/training_model.py"
python "src/single_card/vae_folder.py"
```

The scripts use relative paths such as `model-image`, `test-image`, and weight file names. They may require adapting paths or using the corresponding source layout. Such changes should be made in a new branch and documented; this preservation copy keeps the historical algorithm code unchanged.

## Historical results and limitations

The original materials contain result images, SSIM/error logs, and multiple progress documents. Their reported metrics are historical records, not independently re-executed results for this repository. The preserved documents contain differing evaluation figures at different project stages, so this repository deliberately makes no accuracy, recall, speed, production-readiness, or generalization claim.

Important limitations:

- No reproducible environment, dataset license, or end-to-end test suite was preserved.
- Full datasets are excluded pending authorization.
- The scripts contain duplicated implementations and Windows-relative path assumptions.
- Weight files are retained locally and have not been loaded or executed in this organized copy; Git LFS publication remains pending.

## License

No license is supplied yet. Do not reuse, redistribute, or deploy the code, sample images, or weights beyond the repository access terms until the project owner confirms ownership and licensing conditions.

