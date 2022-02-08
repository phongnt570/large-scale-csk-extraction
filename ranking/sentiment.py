import argparse
import logging

import pymongo
import torch
from pymongo import UpdateOne
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app_config import MONGO_HOST, MONGO_PORT, DB_NAME

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MONGO_CLIENT = pymongo.MongoClient(host=MONGO_HOST, port=MONGO_PORT)

ASCENT_DB = MONGO_CLIENT[DB_NAME]

ASSERTIONS_COL = ASCENT_DB[f"openie_assertions"]

MODEL_ID = f"cardiffnlp/twitter-roberta-base-sentiment"

device = "cuda"

# download label mapping
labels = ["negative", "neutral", "positive"]

# PT
logger.info(f"Load Sentiment Analysis model \"{MODEL_ID}\"")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)


def compute_sentiments(texts):
    encoded_input = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = model(**encoded_input)
    scores = torch.softmax(output.logits, dim=1).tolist()

    return [{labels[i]: s[i] for i in range(len(s))} for s in scores]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4_id", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    logger.info(f"Reading assertions from DB with _id starts with \"{args.c4_id:05d}-\"")
    assertions = list(ASSERTIONS_COL.find({
        "_id": {"$regex": f"^{args.c4_id:05d}-"}
    }))
    logger.info(f"Number of assertions: {len(assertions):,}")

    batch_size = args.batch_size
    num_batches = len(assertions) / batch_size
    if num_batches != int(num_batches):
        num_batches = int(num_batches) + 1
    num_batches = int(num_batches)

    logger.info(f"Num batches: {num_batches}, batch size: {batch_size}")

    queries = []

    for i in range(0, len(assertions), batch_size):
        batch = assertions[i:(i + batch_size)]
        batch_id = int(i / batch_size) + 1
        logger.info(f"Batch {batch_id:05d} / {num_batches:05d}")

        sentences = [a["source"]["sentence"] for a in batch]
        sentiments = compute_sentiments(sentences)

        for a, s in zip(batch, sentiments):
            queries.append(UpdateOne({"_id": a["_id"]}, {"$set": {"sentiment": s}}))

    logger.info(f"Bulk update {len(queries):,} queries")
    ASSERTIONS_COL.bulk_write(queries, ordered=False)

    logger.info("Done")


if __name__ == '__main__':
    main()
