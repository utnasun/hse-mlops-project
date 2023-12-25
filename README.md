# Applied MLOps project @ HSE MLDS 

## ML Task
House price prediction using [kaggle dataset](https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data/data).

## Usage

### Prerequisites
- Conda (or Miniconda)
- Poetry
- MLFlow server running on http://128.0.1.1:8080 (can be changed in configs/config.yaml)
### Example run
```
conda create -n hse_mlops_project python=3.11
conda activate hse_mlops_project
poetry install
pre-commit install
pre-commit run -a
python hse_mlops_project/train.py
python hse_mlops_project/infer.py
```
