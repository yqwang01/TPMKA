# TPMKA

Triple-Phase Multimodal Knowledge Aggregation (TPMKA) is a PyTorch framework for bacterial-versus-fungal microbial keratitis subtype diagnosis from slit-lamp photography. The framework follows the paper "Triple-Phase Multimodal Knowledge Aggregation Framework for Microbial Keratitis Subtype Diagnosis on Slit-Lamp Photography".

TPMKA uses up to three slit-lamp illumination modalities:

- `B`: diffuse blue light with fluorescein staining
- `ScS`: sclerotic scatter
- `W`: diffuse white light

The training pipeline has three phases:

1. Cross-Modality Contrastive Learning (CMCL) pretraining to learn modality-invariant patient representations.
2. Modality-specific fine-tuning for each illumination condition.
3. Feature-level multimodal ensemble learning for patient-level prediction, with missing modalities handled by zero-initialized feature vectors.

The manuscript reports five-fold cross-validation on 1,645 patients and 17,158 images, with a PVT_v2_b1 backbone reaching 85.84% accuracy, 84.46% average F1-score, and 0.885 AUC. The clinical dataset is not included in this repository because of patient privacy restrictions and data-use agreements.

## Repository Layout

```text
.
|-- config.py              # Default experiment settings and path configuration
|-- main.py                # CMCL pretraining, fine-tuning, testing, and Grad-CAM
|-- compute_features.py    # Extract per-image features for ensemble training
|-- main_ensemble.py       # Train/test feature-level multimodal ensemble
|-- data_loader.py         # Dataset split, sampling, augmentation, and loaders
|-- main_funcs.py          # Single-model training, evaluation, and heatmaps
|-- ensemble_funcs.py      # Ensemble training and evaluation
|-- losses.py              # Contrastive and classification losses
|-- utils.py               # Metrics, checkpoint loading, and fairness utilities
`-- models/                # Backbones and ensemble model definitions
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if the command above does not select the right GPU build.

## Data Preparation

The code expects preprocessed `.npz` image files and a CSV manifest. Each image `.npz` should contain an `img` array. The default directory convention is:

```text
data/
|-- data_list.csv
`-- AQUA_Dataset/
    `-- <patient_id>/
        |-- B/
        |-- ScS/
        `-- W/
```

The CSV is expected to contain modality availability columns, the patient identifier, the class label, optional site information, and metadata variables used by `config.var_idx` and `config.sen_attr_idx`. See `data_loader.py` for the exact indexing used by the current experiments.

## Configuration

Edit `config.py` for experiment choices such as backbone, modality, metadata usage, loss weights, and sampling strategy. Paths can also be overridden without editing source code:

```bash
export AQUA_DATA_DIR=/path/to/AQUA_Dataset
export AQUA_TRAIN_CSV=/path/to/data_list.csv
export AQUA_OUTPUT_DIR=/path/to/snapshots
export AQUA_FEATURE_DIR=/path/to/features
export AQUA_PRETRAINED_PATH=/path/to/checkpoint.pth
export AQUA_NUM_WORKERS=8
export CUDA_VISIBLE_DEVICES=0
```

On Windows PowerShell:

```powershell
$env:AQUA_DATA_DIR="C:\path\to\AQUA_Dataset"
$env:AQUA_TRAIN_CSV="C:\path\to\data_list.csv"
$env:AQUA_OUTPUT_DIR="C:\path\to\snapshots"
$env:AQUA_FEATURE_DIR="C:\path\to\features"
$env:CUDA_VISIBLE_DEVICES="0"
```

## Usage

Run Phase One CMCL pretraining:

```bash
python main.py --mode pretraining
```

Run Phase Two modality-specific fine-tuning and testing:

```bash
python main.py --mode finetuning
python main.py --mode testing
```

Generate Grad-CAM heatmaps:

```bash
python main.py --mode heatmap
```

Extract features for Phase Three:

```bash
python compute_features.py
```

Train and test the multimodal ensemble:

```bash
python main_ensemble.py --mode ensemble_train
python main_ensemble.py --mode ensemble_test
```

## Citation

If you use this code, please cite the TPMKA manuscript after publication.

```bibtex
@article{wang2026tpmka,
  title = {Triple-Phase Multimodal Knowledge Aggregation Framework for Microbial Keratitis Subtype Diagnosis on Slit-Lamp Photography},
  author = {Wang, Yiqing and Woodward, Maria A. and Yang, Ziyun and others},
  year = {2026},
  note = {Manuscript under review}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
