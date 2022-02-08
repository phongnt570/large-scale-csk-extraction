import argparse
import logging

import numpy as np
import pandas as pd
import pymongo
from bson import ObjectId
from pymongo import UpdateOne

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


def get_openie_assertions(cluster):
    triples = TRIPLES_COL.find({
        "_id": {"$in": cluster["triples"]}
    })

    assertion_ids = []
    for t in triples:
        assertion_ids.extend(t["assertions"])

    assertions = ASSERTIONS_COL.find({
        "_id": {"$in": assertion_ids}
    })

    return assertions


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

    logger.info("Combine sentiment")
    queries = []
    for cluster in clusters:
        openie_assertions = list(get_openie_assertions(cluster))
        df = pd.DataFrame([a["sentiment"] for a in openie_assertions])
        labels = ["negative", "neutral", "positive"]
        sentiment = {label: np.mean(df[label]) for label in labels}
        queries.append(UpdateOne({"_id": cluster["_id"]}, {"$set": {"sentiment": sentiment}}))

    logger.info(f"Bulk write {len(queries):,} queries")
    CLUSTERS_COL.bulk_write(queries, ordered=False)

    logger.info("Done")


if __name__ == '__main__':
    main()
