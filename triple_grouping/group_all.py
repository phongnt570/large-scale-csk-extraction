import csv
import gzip
import logging
from multiprocessing import Pool
from pathlib import Path

from app_config import WORKING_DIR

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)


def read_grouped_file(filename):
    with gzip.open(filename, "rt") as f:
        reader = csv.DictReader(f)
        triples = []
        for row in reader:
            t = {k: v for k, v in row.items()}
            t["assertion_ids"] = t["assertion_ids"].split("|")
            triples.append(t)
    return triples


def write_results(data, min_freq: int = 1):
    batch = data["batch"]
    output_file = data["output_file"]

    with gzip.open(output_file, "wt") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "assertion_ids", "subject_type",
                                               "super_subject"])
        writer.writeheader()
        cnt = 0
        for t, ids in batch:
            if len(ids) >= min_freq:
                writer.writerow({
                    "subject": t[0],
                    "predicate": t[1],
                    "object": t[2],
                    "assertion_ids": "|".join(ids),
                    "subject_type": t[3],
                    "super_subject": t[4],
                })
                cnt += 1

    logger.info(f"There are {cnt:,} triples with frequency >= {min_freq} written to file \"{output_file}\"")


def main():
    filenames = [f"{WORKING_DIR}/grouped_triples/c4-train.{i:05d}-of-01024.csv.gz" for i in range(1024)]

    triple2ids = {}
    for filename in filenames:
        triples = read_grouped_file(filename)
        logger.info(f"Read \"{filename}\": {len(triples):,} triples")
        old_cnt = len(triple2ids)
        for t in triples:
            tup = (t["subject"], t["predicate"], t["object"], t["subject_type"], t["super_subject"])
            if tup not in triple2ids:
                triple2ids[tup] = []
            triple2ids[tup].extend(t["assertion_ids"])

        logger.info(f"Unique triples: {len(triple2ids):,} (+ {(len(triple2ids) - old_cnt):,})")

    output_dir = Path(f"{WORKING_DIR}/grouped_triples_all")
    output_dir.mkdir(exist_ok=True)
    num_batches = 64
    batch_size = int(len(triple2ids) / num_batches) + 1
    logger.info(f"Num batches: {num_batches}. Batch size: {batch_size:,}.")

    logger.info("Getting items list")
    all_data = list(triple2ids.items())

    datasets = [{
        "batch": all_data[(i * batch_size):((i + 1) * batch_size)],
        "output_file": output_dir / f"{i:03d}-of-{num_batches:03d}.csv.gz"
    } for i in range(num_batches)]

    with Pool(64) as p:
        p.map(write_results, datasets)

    logger.info("Done")


if __name__ == '__main__':
    main()
