# TPMKA

Triple-Phase Multimodal Knowledge Aggregation (TPMKA) is a PyTorch framework for patient-level bacterial-versus-fungal microbial keratitis classification using multimodal slit-lamp photography and clinical information. This repository accompanies the manuscript "Multimodal Knowledge Aggregation for Microbial Keratitis: Geographic Confounding and Shortcut Risk in Pooled Multicountry Data."

Project repository: [https://github.com/yqwang01/TPMKA](https://github.com/yqwang01/TPMKA)

TPMKA uses up to three slit-lamp illumination modalities:

- `B`: diffuse blue light with fluorescein staining
- `ScS`: sclerotic scatter
- `W`: diffuse white light

The training pipeline has three phases:

1. Cross-modality contrastive learning (CMCL) to encourage agreement between representations of images from the same patient.
2. Modality-specific fine-tuning for each illumination condition.
3. Multilayer perceptron (MLP)-based feature fusion of the fixed modality-specific representations and seven clinical variables for patient-level prediction. Missing modalities are represented by zero feature vectors.

The manuscript reports evaluation with five patient-disjoint outer folds on 1,645 patients and 17,158 images. TPMKA achieved five-fold mean overall performance of 83.06% macro-F1, 84.50% accuracy, and 89.93% area under the receiver operating characteristic curve (AUROC). Within-country AUROC was 83.92% in India and 66.96% in the United States. Because organism frequencies differed substantially between countries, the pooled results should be interpreted together with the country- and class-specific analyses reported in the manuscript.

The clinical dataset is not included in this repository because of patient privacy restrictions and data-use agreements. This software is intended for research use and is not a clinical diagnostic device.

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

The CSV is expected to contain modality-availability columns, a patient identifier, a bacterial-or-fungal class label, country information, and the metadata variables selected by `config.var_idx` and `config.sen_attr_idx`. The seven clinical variables used in the manuscript were country, contact-lens use, trauma history, organic-material exposure, water exposure, presenting corrected visual acuity, and uncorrected visual acuity. See `data_loader.py` and `config.py` for the exact indexing expected by the code.

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
  title = {Multimodal Knowledge Aggregation for Microbial Keratitis: Geographic Confounding and Shortcut Risk in Pooled Multicountry Data},
  author = {Wang, Yiqing and Woodward, Maria A. and Yang, Ziyun and others},
  year = {2026},
  note = {Manuscript under review}
}
```

## License

TPMKA is licensed under the GNU General Public License v3.0 or later - see the [LICENSE.txt](LICENSE.txt) file for details.

Copyright (C) 2026 Duke University and the Regents of the University of Michigan.

For questions, contact Yiqing Wang at [yq.wang@duke.edu](mailto:yq.wang@duke.edu).
