import argparse, json, os
from model import *
from data import *
from train import *
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
    )
    parser.add_argument("--sg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument(
        "--model-type", type=str, default="cbow", choices=["cbow", "skip-gram"]
    )
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def plot_loss(path):
    with open(path, "r") as f:
        loss_dict = json.load(f)

    num_of_exp = len(loss_dict)
    plt.subplots(num_of_exp, 1, figsize=(10, 5 * num_of_exp))
    plt.tight_layout()
    for i, loss in enumerate(loss_dict):
        plt.subplot(num_of_exp, 1, i + 1)
        plt.plot(loss["loss"]["train_loss"], label="Train Loss")
        plt.plot(loss["loss"]["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        print(loss)
        plt.title(f"Training and Validation Loss for {loss['exeriment_name']}")
        plt.legend()

    plt.show()


def save_experiment_config(loss, config, path):
    import json, os

    config_dict = {
        k: getattr(config, k)
        for k in config.__dir__()
        if not k.startswith("_") and not callable(getattr(config, k))
    }
    config_dict["loss"] = loss
    config_dict["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = [config_dict]
    if os.path.exists(path):
        with open(path, "r") as f:
            existing_config = json.load(f)
        data.extend(existing_config)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    agrs = get_args()
    config = Config()
    config.batch_size = agrs.batch_size
    config.sg = agrs.sg
    config.min_count = agrs.min_count
    config.window = agrs.window_size
    config.epoch = agrs.epochs
    config.lr = agrs.lr
    config.vector_size = agrs.embedding_dim
    config.model_type = agrs.model_type
    config.sample = agrs.sample
    words = load_dataset(agrs.path)
    if len(words) == 0:
        raise ValueError("No words found in the dataset.")
    print(f"Total words: {len(words)}")

    if config.sample:
        words = words[:10000]

    dataset = word2vecDataset(words, config)
    total_data_size = len(dataset)
    train_size = int(0.8 * total_data_size)
    val_size = total_data_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    print(config.batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    for i in train_dataloader:
        print(i[0].shape)
        break
    vocab_size = dataset.vocab_size
    model = CBOWModel(config=config, vocab_size=vocab_size)
    model.to(config.device)
    loss = train(model, train_dataloader, val_dataloader, config)
    model.save(f"model_{config.exeriment_name}.pth")
    save_experiment_config(loss, config, "experiment_config.json")
    plot_loss("experiment_config.json")


if __name__ == "__main__":
    main()
