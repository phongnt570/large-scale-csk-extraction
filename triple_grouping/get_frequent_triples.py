import csv
import gzip
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

from app_config import WORKING_DIR

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MIN_FREQ = 3
NUM_BATCHES = 64

csv.field_size_limit(sys.maxsize)


def one_file(i: int):
    input_file = f"{WORKING_DIR}/grouped_triples_all/{i:03d}-of-{NUM_BATCHES:03d}.csv.gz"
    triples = []
    cnt = 0
    with gzip.open(input_file, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(row["assertion_ids"].split("|")) >= MIN_FREQ:
                triples.append(row)
            cnt += 1
    logger.info(
        f"Read triples from \"{input_file}\". There are {len(triples):,} / {cnt:,} triples "
        f"with frequency >= {MIN_FREQ}")

    return triples

    # output_dir = Path("/GW/C4/work/grouped_triples_v3_geq_3")
    # output_dir.mkdir(exist_ok=True)
    # output_file = output_dir / f"triples-{i:03d}-of-{NUM_BATCHES:03d}.csv.gz"


def write_results(args):
    triples, output_file = args
    logger.info(f"Writing {len(triples):,} triples to \"{output_file}\"")
    with gzip.open(output_file, "wt") as f:
        writer = csv.DictWriter(f, fieldnames=["triple_id", "subject", "predicate", "object", "assertion_ids",
                                               "subject_type", "super_subject"])
        writer.writeheader()
        writer.writerows(triples)


def main():
    with Pool(64) as p:
        triple_lists = p.map(one_file, range(NUM_BATCHES))

    logger.info(f"Combine triple lists")
    all_triples = [t for tl in triple_lists for t in tl]

    logger.info(f"There are in total {len(all_triples):,} unique triples with frequency >= {MIN_FREQ}")

    output_dir = Path(f"{WORKING_DIR}/grouped_triples_geq_{MIN_FREQ}")
    output_dir.mkdir(exist_ok=True)

    batch_size = int(len(all_triples) / NUM_BATCHES) + 1
    logger.info(f"Num batches: {NUM_BATCHES}. Batch size: {batch_size:,}.")
    triples_batches = [all_triples[(i * batch_size):((i + 1) * batch_size)] for i in range(NUM_BATCHES)]
    for i, triples_batch in enumerate(triples_batches):
        for j, triple in enumerate(triples_batch):
            triple["triple_id"] = f"triple-{i:03d}-{j:07d}"
    output_file_batches = [output_dir / f"triples-{i:03d}-of-{NUM_BATCHES:03d}.csv.gz" for i in range(NUM_BATCHES)]
    with Pool(64) as p:
        p.map(write_results, zip(triples_batches, output_file_batches))

    logger.info("Done")


if __name__ == '__main__':
    main()
