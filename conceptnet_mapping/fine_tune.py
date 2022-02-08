import csv
import logging
from pathlib import Path
from typing import Dict, Union

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaConfig
from transformers import RobertaTokenizerFast

from app_config import WORKING_DIR

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MODEL_NAME = "roberta-base"

TRAIN_FILE = "/path/to/your/training_file.csv"
DEV_FILE = "/path/to/your/dev_file.csv"


class MappingDataset(Dataset):
    def __init__(self, encodings, labels, possible_labels):
        self.encodings = encodings
        self.labels = labels

        self.possible_labels = possible_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        # true_label = self.labels[idx]
        # for label in self.possible_labels:
        #     item[label] = torch.tensor(float(true_label == label))

        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids.argmax(-1)
    preds = pred.predictions.argmax(-1)

    # logger.info(pred)
    #
    # logger.info(labels)
    # logger.info(preds)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    logger.info(f"Accuracy: {acc}")

    return {
        "accuracy": acc,
    }


def make_input(row: Dict[str, str], sep_token: str):
    return f"{row['s']} {sep_token} {row['po']}"


def read_data_split(filename: Union[str, Path], sep_token: str):
    with open(filename) as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    possible_labels = [k for k in data[0].keys() if k.startswith("/r/")]
    logger.info(possible_labels)

    texts = [make_input(row, sep_token) for row in data]
    # labels = [row["target"] for row in data]
    labels = [[float(row[label]) for label in possible_labels] for row in data]

    return texts, labels, possible_labels


def main():
    logger.info(f"Load Tokenizer \"{MODEL_NAME}\"")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    logger.info(f"Read data")
    train_texts, train_labels, possible_labels = read_data_split(TRAIN_FILE, sep_token=tokenizer.sep_token)
    dev_texts, dev_labels, dev_possible_labels = read_data_split(DEV_FILE, sep_token=tokenizer.sep_token)

    assert len(possible_labels) == len(dev_possible_labels) and all(
        a == b for a, b in zip(possible_labels, dev_possible_labels))

    # possible_labels = list(set(train_labels))
    logger.info(possible_labels)

    logger.info(f"Tokenize {len(train_texts):,} training samples")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    logger.info(f"Tokenize {len(dev_texts):,} development samples")
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)

    train_dataset = MappingDataset(train_encodings, train_labels, possible_labels)
    dev_dataset = MappingDataset(dev_encodings, dev_labels, possible_labels)

    logger.info("Create TrainingArguments")
    training_args = TrainingArguments(
        output_dir=f"{WORKING_DIR}/conceptnet_mapping_model",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=128,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="/tmp/logs/",  # directory for storing logs
        disable_tqdm=True,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
    )

    logger.info(f"Load model \"{MODEL_NAME}\"")
    label2id = {label: i for i, label in enumerate(possible_labels)}
    id2label = {str(i): label for i, label in enumerate(possible_labels)}
    config = RobertaConfig.from_pretrained(MODEL_NAME, label2id=label2id, id2label=id2label,
                                           num_labels=len(possible_labels))
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    logger.info("Start training")
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )

    trainer.train()

    logger.info("Evaluate")
    trainer.evaluate()

    logger.info("Save model")
    model_path = f"{WORKING_DIR}/conceptnet_mapping_model/best"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    logger.info("Done")


if __name__ == '__main__':
    main()
