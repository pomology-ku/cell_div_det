# cell_div_det

## 1. Overview

This repository provides training and inference scripts for cell division detection using oriented bounding boxes (OBB).

- **Task**: Cell division detection
- **Input**: Electron microscopy images
- **Output**: OBB detections and JSON export

## Installation

Minimum dependencies (via `pip`):

```bash
# training
pip install -U ultralytics pyyaml

# inference (recommended)
pip install -U ultralytics pyyaml numpy opencv-python pillow tqdm torch torchvision
```

Optional (only if you use `--use_sahi_slice` in `inference_obb.py`):

```bash
pip install -U sahi
```

## 2. Training

Example command:

```bash
python train.py --data dat/pub/fold0/data.yaml --cfg params/param.exp096.yaml
```

- **Dataset YAML**: `dat/pub/fold0/data.yaml`
- **Config YAML**: `params/param.exp096.yaml`

## 3. Inference

Example command:

```bash
python inference_obb.py -m models/celldivdet.exp096_yolo11l-obb.pt -c 0.1 -s 1920 -d dat/unannotated/akatsuki_long_1_2/P#000 --out-json results/akatsuki_long_1_2_exp096.conf0.1.json
```

Arguments (brief):
- `-m`: model weights (`.pt`)
- `-c`: confidence threshold
- `-s`: image size
- `-d`: input image / directory
- `--out-json`: output JSON path

## 4. Pretrained weights and training data

Two pretrained weights and the training dataset archive are available in the Google Drive folder:

- Source folder: `https://drive.google.com/drive/folders/1eFi3cjVvsQUl5Zim0GyiK7YFjcdL5Sgz?usp=drive_link`
- Contains:
  - `celldivdet.dat.v1.tar.gz`
  - `models.zip`

Direct file pages:
- `celldivdet.dat.v1.tar.gz`: `https://drive.google.com/file/d/1pS-dkQe2srEOqXtaZZjojJIzrktKtvvI/view?usp=drive_link`
- `models.zip`: `https://drive.google.com/file/d/1SiRDhaWtdNvJC-GayjGUjIVH8y9sY_Lt/view?usp=drive_link`


## 5. Citation

Please cite:

A visualization framework for cell division activity and orientation in pre-anthesis ovaries of *Prunus* species.  
Ayame Shimbo, Soichiro Nishiyama, Tatsuya Katsuno, Akane Kusumi, Hisayo Yamane, Masahiro M. Kanaoka, and Ryutaro Tao.  
In preparation.
