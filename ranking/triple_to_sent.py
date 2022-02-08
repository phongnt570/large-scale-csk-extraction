import argparse
import logging
from collections import Counter

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


def get_triple_sentence(cluster):
    assertions = list(get_openie_assertions(cluster))

    counter = Counter()
    for a in assertions:
        if not (a["predicate"].rstrip("\x00") == cluster["predicate"] and a["object"].rstrip("\x00") == cluster["object"]):
            continue

        source = a["source"]
        sentence = source["sentence"]
        tokens = source["tokens"]
        pos = source["positions"]

        subject = sentence[pos["subj_start_char"]:pos["subj_end_char"]]
        predicate = " ".join(" ".join(tokens[p["start"]:p["end"]]) for p in pos["pred_positions"])
        obj = sentence[pos["obj_start_char"]:pos["obj_end_char"]]

        counter[(subject, predicate, obj)] += 1

    sorted_cnt = counter.most_common()
    res = sorted_cnt[0]
    for s in sorted_cnt:
        pred = s[0][1].lower()
        pred_toks = pred.split()
        if not pred_toks[0].endswith("ing"):
            res = s
            break
    sent = " ".join(res[0]) + "."
    sent_toks = sent.split()
    first_tok = sent_toks[0]
    if first_tok.islower():
        sent_toks[0] = first_tok.capitalize()
    return " ".join(sent_toks).replace('\0', '')


def get_add_sentence_update_query(cluster):
    _id = cluster["_id"]

    return UpdateOne({"_id": _id}, {"$set": {"triple_sentence": get_triple_sentence(cluster)}})


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

    logger.info(f"Get sentences")
    queries = [get_add_sentence_update_query(cluster) for cluster in clusters]

    logger.info(f"Bulk write {len(queries):,} queries")
    CLUSTERS_COL.bulk_write(queries, ordered=False)

    logger.info("Done")


if __name__ == '__main__':
    main()
