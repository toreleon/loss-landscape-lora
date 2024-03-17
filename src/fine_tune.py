import os
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from local_dataset_utilities import (
    download_dataset,
    load_dataset_into_to_dataframe,
    partition_dataset,
)

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_proj'],
                        )

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")


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

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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


# from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, partition_dataset
model = AutoModelForSequenceClassification.from_pretrained(
    "model/opt-125m", num_labels=2, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained("model/opt-125m", local_files_only=True)
model = get_peft_model(model, peft_config)
model.to(device)
model.print_trainable_parameters()

# Freeze all layers except the last (classifier) layer
# for name, param in model.named_parameters():
#     if "score" not in name:  # Freeze layers that are not part of the classifier
#         param.requires_grad = False

# Allow all layers to be trained
for param in model.parameters():
    param.requires_grad = True

print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_steps=500,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    f1 = f1_score(labels, predictions, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    imdb_dataset = dataset_preparation()
    train_loader, val_loader, test_loader = prepare_data_loaders(imdb_dataset)
    # Limit the train_loader to 10 batches
    train_loader = list(train_loader)
    # train_loader = train_loader[:30]
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_metrics=compute_metrics,
    )

    # # Train the model
    trainer.train()
    trainer.evaluate(eval_dataset=test_loader, metric_key_prefix="test")
    # # Save the fine-tuned model
    model.save_pretrained("model/otp-125m-lora-r4")

    # Optionally, save the tokenizer as well if you've made changes or added tokens
    tokenizer.save_pretrained("model/otp-125m-lora-r4")
