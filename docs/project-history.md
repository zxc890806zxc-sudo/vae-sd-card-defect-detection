# Preservation and provenance notes

## Scope

This repository was assembled from a local historical project directory. Files were copied into this clean structure; no original code, data, or document was moved, renamed, or edited.

## Script groups

- `src/single_card/` contains the historical single-card training, batch-analysis, and camera scripts.
- `src/multi_region/` contains the historical four-region (D1-D4) scripts.
- `src/multi_region_camera_v2/` contains a later, more extensive multi-region camera and local-region analysis variant.

The groups are preserved as variants rather than merged or refactored. The repository does not assert that one is the final or preferred implementation.

## Excluded source material

The original directory also contained full image collections, duplicate ZIP archives, older experiments, unrelated coursework, presentation templates, personal records, travel receipts, third-party tools, and competition documents. Those files are not part of this repository.

## Reproducibility status

Only static syntax parsing of the selected Python code was performed during organization. No model training, camera capture, inference, dataset evaluation, or benchmark was executed. Environment versions, GPU/CUDA details, and dataset permissions remain unknown.

## Duplicate handling

The original `vae_folder_12.py` and `vae_folder_12 (1).py` had identical content. The organized copy contains one unchanged instance; the duplicate remains untouched in the original directory.
