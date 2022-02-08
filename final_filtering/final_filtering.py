import argparse
import csv
import json
import logging

import pymongo

from app_config import MONGO_HOST, MONGO_PORT, DB_NAME
from .module_conceptnet_hyponym import ConceptNetHyponymModule
from .module_ignore_object import IgnoreObjectsModule
from .module_ignore_po import IgnorePOModule
from .module_ignore_predicate import IgnorePredicatesModule
from .module_ignore_relation import IgnoreRelationModule
from .module_negation import NegationModule
from .module_object_contains_word import ObjectContainsWordModule
from .module_perplexity import PerplexityModule
from .module_regex import RegexModule
from .module_relation_ingore_obj_word import RelationIgnoreObjectWordModule

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MONGO_CLIENT = pymongo.MongoClient(host=MONGO_HOST, port=MONGO_PORT)
ASCENT_DB = MONGO_CLIENT[DB_NAME]

CLUSTERS_COL = ASCENT_DB[f"clustered_triples"]

FILTERS = [
    PerplexityModule(threshold=500),

    ConceptNetHyponymModule(filename="../data/conceptnet_hypos.jsonl"),
    IgnoreObjectsModule(),
    IgnorePOModule(),
    IgnorePredicatesModule(),
    IgnoreRelationModule(),
    NegationModule(),
    ObjectContainsWordModule(),
    RegexModule(),
    RelationIgnoreObjectWordModule(),
]


def should_keep(row) -> bool:
    for filtering in FILTERS:
        if not filtering.validate(row):
            return False
    return True


def normalize_object(obj):
    tokens = obj.split()
    if len(tokens) > 1 and tokens[0] in {"a", "an", "the", "one"}:
        tokens = tokens[1:]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--keep_r", action="store_true")
    parser.add_argument("--aspects", action="store_true")

    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    fil_str = "- " + "\n- ".join(str(f) for f in FILTERS)
    logger.info(f"Filtering modules:\n{fil_str}")

    logger.info("Read clusters from DB")
    rows = []
    cnt = 0
    for c in CLUSTERS_COL.find():
        cnt += 1

        row = {k: v for k, v in c.items()}

        if not should_keep(row):
            continue

        row["_id"] = str(row["_id"])

        row.pop("triples")

        pred = row["predicted_relation"]
        if not args.keep_r:
            if pred.startswith("/r/"):
                row["predicted_relation"] = pred.split("/")[-1]

        row["negative"] = row["sentiment"]["negative"]
        row["neutral"] = row["sentiment"]["neutral"]
        row["positive"] = row["sentiment"]["positive"]
        row.pop("sentiment")

        row["metadata"] = json.dumps({
            "triple_cluster": row["triple_cluster"],
            "facets": row["facets"],
        })
        row.pop("triple_cluster")
        row.pop("facets")

        rows.append(row)

    logger.info(f"There are in total {cnt:,} triples, of which {len(rows):,} survived")

    logger.info(f"Cutting off long-tailed clusters, topN = {args.top_n}")
    subject2rows = {}
    subject2aspects = {}
    for row in rows:
        subject = (row["subject"], row["subject_type"], row["super_subject"])
        if subject not in subject2rows:
            subject2rows[subject] = []
        subject2rows[subject].append(row)

        if args.aspects and row["subject_type"] == "aspect":
            primary_subject = row["super_subject"]
            if primary_subject not in subject2aspects:
                subject2aspects[primary_subject] = set()
            subject2aspects[primary_subject].add(row["subject"])

    final_rows = []
    for subject, assertions in subject2rows.items():
        sorted_assertions = sorted(assertions, key=lambda x: x["count"], reverse=True)

        seen_triples = set()
        retained_assertions = []
        for a in sorted_assertions:
            t = (a["predicted_relation"], normalize_object(a["predicted_object"]))
            if t not in seen_triples:
                retained_assertions.append(a)
                seen_triples.add(t)
                if len(retained_assertions) >= args.top_n:
                    break
        final_rows.extend(retained_assertions)

    logger.info(f"There are {len(final_rows):,} survived")

    logger.info(f"There are {sum(len(aspects) for aspects in subject2aspects.values()):,} aspects")

    output_file = args.output_file
    logger.info(f"Write {len(final_rows):,} triples to \"{output_file}\"")
    if not args.aspects:
        with open(output_file, "w+") as output_f:
            writer = csv.DictWriter(output_f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)
    else:
        with open(output_file, "w+") as output_f:
            writer = csv.DictWriter(output_f, fieldnames=["subject", "predicate", "object"])
            writer.writeheader()

            writer.writerows([{
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
            } for row in final_rows])

            writer.writerows([{
                "subject": subject,
                "predicate": "has",
                "object": aspect,
            } for subject, aspects in subject2aspects.items() for aspect in aspects])

    logger.info("Done")


if __name__ == '__main__':
    main()
