# Based on https://huggingface.co/transformers/perplexity.html

import argparse
import logging

import pymongo
import torch
from bson import ObjectId
from pymongo import UpdateOne
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from app_config import MONGO_HOST, MONGO_PORT, DB_NAME

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MONGO_CLIENT = pymongo.MongoClient(host=MONGO_HOST, port=MONGO_PORT)

ASCENT_DB = MONGO_CLIENT[DB_NAME]

ASSERTIONS_COL = ASCENT_DB[f"openie_assertions"]
TRIPLES_COL = ASCENT_DB[f"grouped_triples"]
CLUSTERS_COL = ASCENT_DB[f"clustered_triples"]

device = "cuda"
model_id = "gpt2-large"

logger.info(f"Loading model \"{model_id}\" to device \"{device}\"")
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

logger.info(f"Loading tokenizer \"{model_id}\"")
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


def get_perplexity(sentence):
    encodings = tokenizer(sentence, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)

    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=int, required=True)
    parser.add_argument("--num_batches", type=int, required=True)
    parser.add_argument("--id_file", type=str, required=True)

    args = parser.parse_args()

    logger.info(f"Read cluster ids from \"{args.id_file}\"")
    with open(args.id_file) as f:
        ids = [ObjectId(line.strip()) for line in f]
    logger.info(f"There are {len(ids):,} ids")

    batch_size = int(len(ids) / args.num_batches) + 1
    logger.info(f"Num batches: {args.num_batches}, batch id: {args.batch_id}, batch size: {batch_size}")

    start = args.batch_id * batch_size
    end = (args.batch_id + 1) * batch_size

    logger.info(f"Batch start: {start:,}, batch end: {end:,}")

    process_ids = ids[start:end]
    logger.info(f"First 10 ids: {process_ids[:10]}")

    logger.info("Get clusters")
    clusters = list(CLUSTERS_COL.find({
        "_id": {"$in": process_ids}
    }))
    assert len(clusters) == len(process_ids)
    logger.info(f"Got {len(clusters):,} clusters")

    logger.info("Compute perplexity")
    queries = []
    for cluster in clusters:
        sentence = cluster["triple_sentence"]
        perplexity = get_perplexity(sentence).item()
        queries.append(UpdateOne({"_id": cluster["_id"]}, {"$set": {"perplexity": perplexity}}))

    logger.info(f"Bulk write {len(queries):,} queries")
    CLUSTERS_COL.bulk_write(queries, ordered=False)

    logger.info("Done")


if __name__ == '__main__':
    main()
