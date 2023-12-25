import subprocess
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
from catboost import CatBoostRegressor
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error

module_dir_path = Path(__file__).absolute().parent.parent
sys.path.append(str(module_dir_path))

from hse_mlops_project.utils import plot_train_loss, preprocess_data


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    train_data_path = Path(hydra.utils.get_original_cwd()) / cfg.data.train_path

    subprocess.run(["dvc", "pull", "--force"])

    X_train, y_train = preprocess_data(train_data_path)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    experiment_name = cfg.mlflow.experiment_name
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model = CatBoostRegressor(
            iterations=cfg.model.iterations,
            learning_rate=cfg.model.learning_rate,
            depth=cfg.model.depth,
            l2_leaf_reg=cfg.model.l2_leaf_reg,
            allow_writing_files=False,
            silent=True,
            eval_metric=cfg.model.eval_metric,
        )

        mlflow.log_params(
            {
                "iterations": cfg.model.iterations,
                "learning_rate": cfg.model.learning_rate,
                "depth": cfg.model.depth,
                "l2_leaf_reg": cfg.model.l2_leaf_reg,
                "eval_metric": cfg.model.eval_metric,
            }
        )

        model.fit(X_train, y_train, eval_set=(X_train, y_train), plot=False)

        model_path = Path(hydra.utils.get_original_cwd()) / cfg.model.save_path
        model.save_model(model_path)

        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        mlflow.log_param("git_commit_id", commit_id)

        eval_metrics = model.get_evals_result()
        train_losses = eval_metrics["learn"][cfg.model.eval_metric]

        predictions = model.predict(X_train)
        mae = mean_absolute_error(y_train, predictions)
        mse = mean_squared_error(y_train, predictions)
        rmse = np.sqrt(mse)

        mlflow.log_metrics({"MAE": mae, "MSE": mse, "RMSE": rmse})

        graph_path = plot_train_loss(train_losses)

        mlflow.log_artifact(graph_path, artifact_path="plots")

        mlflow.sklearn.log_model(model, "catboost_model")


if __name__ == "__main__":
    train()
