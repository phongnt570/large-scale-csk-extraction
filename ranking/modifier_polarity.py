import argparse
import logging

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

FACET2SCORE = {
    "always": 1.0,
    "typically": 0.9,
    "mostly": 0.9,
    "mainly": 0.9,
    "usually": 0.8,
    "normally": 0.8,
    "regularly": 0.8,
    "frequently": 0.8,
    "commonly": 0.8,
    "often": 0.6,
    "sometimes": 0.4,
    "occasionally": 0.3,
    "hardly": 0.1,
    "rarely": 0.1,
}

QUANTIFIER2SCORE = {
    "all": 1.0,
    "every": 1.0,
    "most": 0.9,
    "many": 0.7,
    "some": 0.5,
    "few": 0.3,
    "no": 0,
    "none": 0,
}

DEFAULT_POLARITY = 0.5


def get_subject_lemmas(assertion):
    start = assertion["source"]["positions"]["subj_start"]
    end = assertion["source"]["positions"]["subj_end"]
    return assertion["source"]["lemmas"][start:end]


def get_subject_tokens(assertion):
    start = assertion["source"]["positions"]["subj_start"]
    end = assertion["source"]["positions"]["subj_end"]
    return assertion["source"]["tokens"][start:end]


def get_plural_ratio(assertions):
    cnt = 0
    all_cnt = 0
    for a in assertions:
        if not a["subject"] in [lemma.lower() for lemma in get_subject_tokens(a)]:
            cnt += 1
        all_cnt += 1
    return cnt / all_cnt


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


def get_relevant_facets(assertions):
    facets = []
    for a in assertions:
        for f in a["facets"]:
            if f["connector"] is None and f["statement"].lower() in FACET2SCORE:
                facets.append(f["statement"].lower())
    return facets


def get_average_facet_score(assertions):
    facets = get_relevant_facets(assertions)

    if len(facets) == 0:
        return -1

    return sum(FACET2SCORE[facet] for facet in facets) / len(facets)


def get_max_facet_score(assertions):
    facets = get_relevant_facets(assertions)

    if len(facets) == 0:
        return -1

    return max(FACET2SCORE[facet] for facet in facets)


def get_quantifiers(assertions):
    quantifiers = []
    for a in assertions:
        for lemma in get_subject_lemmas(a):
            if lemma.lower() in QUANTIFIER2SCORE:
                quantifiers.append(lemma.lower())
    return quantifiers


def get_average_quantifier_score(assertions):
    quantifiers = get_quantifiers(assertions)

    if len(quantifiers) == 0:
        return -1

    return sum(QUANTIFIER2SCORE[quantifier] for quantifier in quantifiers) / len(quantifiers)


def get_max_quantifier_score(assertions):
    quantifiers = get_quantifiers(assertions)

    if len(quantifiers) == 0:
        return -1

    return max(QUANTIFIER2SCORE[quantifier] for quantifier in quantifiers)


def get_modifier_polarity(assertions):
    facets = get_relevant_facets(assertions)
    quantifiers = get_quantifiers(assertions)

    scores = [FACET2SCORE[facet] for facet in facets] + [QUANTIFIER2SCORE[quantifier] for quantifier in quantifiers]

    pol = DEFAULT_POLARITY
    if len(scores) > 0:
        pol = sum(score for score in scores) / len(scores)

    return {
        "mod_pol": pol,
        "num_mod": len(scores)
    }


# def get_scores(cluster):
#     openie_assertions = list(get_openie_assertions(cluster))
#
#     return get_modifier_polarity(openie_assertions)

# return {
#     "plural_ratio": get_plural_ratio(openie_assertions),
#     "avg_fct_score": get_average_facet_score(openie_assertions),
#     "max_fct_score": get_max_facet_score(openie_assertions),
#     "avg_qut_score": get_average_quantifier_score(openie_assertions),
#     "max_qut_score": get_max_quantifier_score(openie_assertions),
# }


def get_modifier_polarity_update_query(cluster):
    _id = cluster["_id"]

    openie_assertions = list(get_openie_assertions(cluster))

    scores = get_modifier_polarity(openie_assertions)

    return UpdateOne({"_id": _id}, {"$set": scores})


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

    logger.info(f"Compute modifier polarity")
    queries = [get_modifier_polarity_update_query(cluster) for cluster in clusters]

    # logger.info(f"Compute normalized log frequency")
    # queries.extend(get_freq_update_queries(clusters))

    logger.info(f"Bulk write {len(queries):,} queries")
    CLUSTERS_COL.bulk_write(queries, ordered=False)

    logger.info("Done")


if __name__ == '__main__':
    main()
