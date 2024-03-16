import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, partition_dataset
from visualizer import visualize_loss_landscape, get_random_direction

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row["text"]
        label = row["label"]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }
    
def prepare_data_loaders(imdb_dataset, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = IMDBDataset(imdb_dataset["train"], tokenizer)
    val_dataset = IMDBDataset(imdb_dataset["validation"], tokenizer)
    test_dataset = IMDBDataset(imdb_dataset["test"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

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

    model_path = "opt-125m"
    if not os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-125m", num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = model.to("cuda")
    imdb_dataset = dataset_preparation()
    train_loader, val_loader, test_loader = prepare_data_loaders(imdb_dataset)\
    # Limit the train_loader to 10 batches
    train_loader = list(train_loader)
    train_loader = train_loader[:30]
    alphas = np.linspace(-1.0, 1.0, 50)
    directions = [get_random_direction(model) for _ in range(3)]
    visualize_loss_landscape(model, torch.nn.CrossEntropyLoss(), train_loader, directions, alphas)