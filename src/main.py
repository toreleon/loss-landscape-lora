import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from local_dataset_utilities import (
    download_dataset,
    load_dataset_into_to_dataframe,
    partition_dataset,
    IMDBDataset
)
from peft import PeftModel
from visualizer import visualize_loss_landscape, get_random_direction

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_data_loaders(imdb_dataset, batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = IMDBDataset(imdb_dataset["train"], tokenizer)
    val_dataset = IMDBDataset(imdb_dataset["validation"], tokenizer)
    test_dataset = IMDBDataset(imdb_dataset["test"], tokenizer)

    return train_dataset, val_dataset, test_dataset

def dataset_preparation():
    files = ("test.csv", "train.csv", "val.csv")
    download = True

    for f in files:
        if not os.path.exists(os.path.join("data", f)):
            download = False

    if download is False:
        download_dataset()
        df = load_dataset_into_to_dataframe()
        partition_dataset(df)

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join("data", "train.csv"),
            "validation": os.path.join("data", "val.csv"),
            "test": os.path.join("data", "test.csv"),
        },
    )
    return imdb_dataset


if __name__ == "__main__":

    model_path = "model/opt-125m"
    adapter_path = "model/otp-125m-lora-r4"
    run_name = "otp-125m-lora-r4"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, local_files_only=True, offload_folder="offload", offload_state_dict = True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = model.to("cuda")
    imdb_dataset = dataset_preparation()
    train_loader, val_loader, test_loader = prepare_data_loaders(imdb_dataset)
    # Limit the train_loader to 10 batches
    train_loader = list(train_loader)[:500]
    alphas = np.linspace(-1.0, 1.0, 40)
    directions = [get_random_direction(model) for _ in range(2)]
    visualize_loss_landscape(
        run_name, model, train_loader, directions, alphas
    )
