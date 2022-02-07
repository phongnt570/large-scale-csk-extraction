import argparse
import csv
import gzip
import logging
from pathlib import Path

from app_config import WORKING_DIR

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_idx", type=int, required=True)
    parser.add_argument("--in_dir", type=str, default=f"{WORKING_DIR}/relevant_triples")
    parser.add_argument("--out_dir", type=str, default=f"{WORKING_DIR}/grouped_triples")

    args = parser.parse_args()

    input_file = Path(args.in_dir) / f"c4-train.{args.file_idx:05d}-of-01024.csv.gz"
    logger.info(f"Reading triples from \"{input_file}\"")
    with gzip.open(input_file, "rt") as f:
        reader = csv.DictReader((line.replace('\0', '') for line in f))
        triples = [row for row in reader]
    logger.info(f"There are {len(triples):,} triples")

    logger.info("Grouping")
    triple2ids = {}
    for t in triples:
        tup = (t["subject"], t["predicate"], t["object"], t["subject_type"], t["super_subject"])
        if tup not in triple2ids:
            triple2ids[tup] = []
        triple2ids[tup].append(t["assertion_id"])
    logger.info(f"There are {len(triple2ids):,} unique triples")

    output_file = Path(args.out_dir) / f"c4-train.{args.file_idx:05d}-of-01024.csv.gz"
    logger.info(f"Writing to \"{output_file}\"")
    with gzip.open(output_file, "wt") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "assertion_ids", "subject_type",
                                               "super_subject"])
        writer.writeheader()
        for t, ids in triple2ids.items():
            writer.writerow({
                "subject": t[0],
                "predicate": t[1],
                "object": t[2],
                "assertion_ids": "|".join(ids),
                "subject_type": t[3],
                "super_subject": t[4],
            })
    logger.info("Done")


if __name__ == '__main__':
    main()
