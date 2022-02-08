import argparse
import csv
import logging
import math

import pymongo
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


def get_freq_update_queries(clusters):
    log_scores = [math.log(c["count"]) for c in clusters]

    if len(log_scores) == 0:
        return []

    max_log = max(log_scores)
    min_log = min(log_scores)
    if max_log == min_log:
        new_scores = [1] * len(log_scores)
    else:
        new_scores = [(s - min_log) / (max_log - min_log) for s in log_scores]

    queries = []
    for c, s in zip(clusters, new_scores):
        queries.append(UpdateOne({"_id": c["_id"]}, {"$set": {"log_freq": s}}))

    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=int, required=True)
    parser.add_argument("--num_batches", type=int, required=True)
    parser.add_argument("--subject_file", type=str, required=True)

    args = parser.parse_args()

    logger.info(f"Read subjects from \"{args.subject_file}\"")
    with open(args.subject_file) as f:
        #     all_subjects = [line.strip() for line in f]
        reader = csv.DictReader(f)
        all_subjects = [row for row in reader]
    logger.info(f"There are {len(all_subjects):,} subjects")

    batch_size = int(len(all_subjects) / args.num_batches) + 1
    logger.info(f"Num batches: {args.num_batches}, batch id: {args.batch_id}, batch size: {batch_size}")

    start = args.batch_id * batch_size
    end = (args.batch_id + 1) * batch_size

    logger.info(f"Batch start: {start:,}, batch end: {end:,}")

    subjects = all_subjects[start:end]
    logger.info(f"{subjects}")

    logger.info("Compute scores")
    all_queries = []
    for subject in subjects:
        # logger.info(f"Subject: {subject}")
        clusters = list(CLUSTERS_COL.find({
            "subject": subject["subject"],
            "subject_type": subject["type"],
            "super_subject": subject["super_subject"]
        }))

        queries = get_freq_update_queries(clusters)
        if queries:
            # logger.info(f"  + {len(queries)} queries")
            all_queries.extend(queries)

    if len(all_queries):
        logger.info(f"Bulk write {len(all_queries):,} queries")
        CLUSTERS_COL.bulk_write(all_queries, ordered=False)
    else:
        logger.info("There is no query!!!")

    logger.info("Done")


if __name__ == '__main__':
    main()
