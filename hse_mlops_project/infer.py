import os
import sys
from pathlib import Path

import hydra
import pandas as pd
from catboost import CatBoostRegressor
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error

module_dir_path = Path(__file__).absolute().parent.parent
sys.path.append(str(module_dir_path))

from hse_mlops_project.utils import preprocess_data


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def infer(cfg: DictConfig) -> None:
    model_path = Path(hydra.utils.get_original_cwd()) / cfg.model.save_path
    predictions_path = Path(hydra.utils.get_original_cwd()) / cfg.predictions.save_path

    model = CatBoostRegressor()
    model.load_model(model_path)

    os.makedirs(predictions_path.parent, exist_ok=True)

    test_data_path = Path(hydra.utils.get_original_cwd()) / cfg.data.test_path
    X_test, y_test = preprocess_data(test_data_path)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    predictions_df = pd.DataFrame({"Prediction": predictions, "Actual": y_test})
    predictions_df.to_csv(predictions_path, index=False)


if __name__ == "__main__":
    infer()
