import matplotlib.pyplot as plt
import pandas as pd


def preprocess_data(data_path: str) -> tuple:
    df = pd.read_csv(data_path)
    df["Neighborhood"] = (
        df["Neighborhood"].map({"Suburb": 0, "Rural": 1, "Urban": 2}).astype(int)
    )

    X_preprocessed = df.drop("Price", axis=1)
    target = df["Price"]

    return X_preprocessed, target


def plot_train_loss(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    graph_path = "data/train_loss_plot.png"
    plt.savefig(graph_path)
    plt.close()
    return graph_path
