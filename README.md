# D4diffCyto

Official code release for the MICCAI 2026 paper:

**Group Equivariant Diffusion for Anomaly Detection in Computational Cytology**

Authors: Swarnadip Chatterjee, Ssharvien Kumar Sivakumar, and Anirban Mukhopadhyay

This repository contains code for D4-aware diffusion-based anomaly detection on cytology single-cell patches. The method targets the setting where rotations and flips of a centered cell patch should not change its diagnostic class, but standard diffusion-based anomaly detectors may produce transformation-dependent reconstructions and anomaly scores.

## Implemented variants

The repository includes code for the following variants:

1. Vanilla AnoDDPM baseline
2. FA+EN inference-time D4 enforcement
3. D4-equivariant U-Net without attention
4. D4-equivariant U-Net with attention

The anomaly score is computed as the mean-squared reconstruction error between the input patch and its partial-diffusion reconstruction.

## Repository structure

```text
D4diffCyto/
├── GaussianDiffusion.py
├── UNet.py
├── diffusion_training.py
├── cells_detection.py
├── cells_detection_phase2.py
├── dataset.py
├── evaluation.py
├── helpers.py
├── simplex.py
├── graphs.py
├── generate_images.py
├── requirements.txt
├── LICENSE
├── test_args/
│   ├── mll/
│   │   ├── anoddpm.json
│   │   ├── fa_en.json
│   │   ├── d4_noattn.json
│   │   ├── d4_attn.json
│   │   └── fa_only_optional.json
│   └── amllmu/
│       ├── anoddpm.json
│       ├── d4_noattn.json
│       └── d4_attn.json
├── data/
├── model/
└── metrics/
```

## Installation

```bash
conda create -n d4diffcyto python=3.9
conda activate d4diffcyto
pip install -r requirements.txt
```

## Running inference

### Vanilla AnoDDPM

```bash
python cells_detection.py --config test_args/mll/anoddpm.json
python cells_detection.py --config test_args/amllmu/anoddpm.json
```

### FA+EN inference-time D4 enforcement

```bash
python cells_detection_phase2.py --config test_args/mll/fa_en.json
```

### D4-equivariant U-Net without attention

```bash
python cells_detection.py --config test_args/mll/d4_noattn.json
python cells_detection.py --config test_args/amllmu/d4_noattn.json
```

### D4-equivariant U-Net with attention

```bash
python cells_detection.py --config test_args/mll/d4_attn.json
python cells_detection.py --config test_args/amllmu/d4_attn.json
```

### Optional FA-only ablation

```bash
python cells_detection_phase2.py --config test_args/mll/fa_only_optional.json
```

## Outputs

The code writes checkpoints, anomaly scores, metrics, reconstructions, and generated visualizations to output folders such as:

```text
model/
metrics/
diffusion-samples/
diffusion-training-images/
diffusion-videos/
```

Large generated files, datasets, checkpoints, and intermediate outputs are excluded from the repository through `.gitignore`.

## Citation

```bibtex
@inproceedings{chatterjee2026d4diffcyto,
  title     = {Group Equivariant Diffusion for Anomaly Detection in Computational Cytology},
  author    = {Chatterjee, Swarnadip and Sivakumar, Ssharvien Kumar and Mukhopadhyay, Anirban},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year      = {2026}
}
```

## License

See `LICENSE`.