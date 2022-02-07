import argparse
import csv
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Union, Tuple

from app_config import WORKING_DIR
from .assertion_reader import load_one_assertion_file, AssertionId
from .filtering_helper import is_likely_valid

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MIN_FILE_INDEX = 0
MAX_FILE_INDEX = 1023

NUM_FILES = 64


def get_assertions_of_subjects(subjects: Dict[str, Set[Tuple[str, str]]], assertion_lists: List[List[Dict[str, Any]]],
                               c4_id: int,
                               good_su_pairs: Set[Tuple[str, str]]) \
        -> List[Dict[str, Union[str, AssertionId]]]:
    res = []
    for part_id, al in enumerate(assertion_lists):
        for asst_id, a in enumerate(al):
            subj = a["subject"]
            # if subj not in subjects:
            #     continue
            # if (subj, a["source"]["document"]) not in good_su_pairs:
            #     continue

            if not is_likely_valid(a):
                continue

            if subj not in subjects:
                continue

            for subj_type, super_subject in subjects[subj]:
                if (super_subject, a["source"]["document"]) not in good_su_pairs:
                    continue

                res.append({
                    "subject": subj,
                    "predicate": a["predicate"],
                    "object": a["object"],
                    "assertion_id": AssertionId(c4_id=c4_id, part_id=part_id, asst_id=asst_id),
                    "subject_type": subj_type,
                    "super_subject": super_subject,
                })

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, required=True)
    parser.add_argument("--c4_file_index", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--openie_dir", type=str, default=f"{WORKING_DIR}/openie_output")
    parser.add_argument("--threshold", type=float, default=0.6)

    args = parser.parse_args()

    assert MIN_FILE_INDEX <= args.c4_file_index <= MAX_FILE_INDEX

    openie_dir = Path(args.openie_dir)
    directory = openie_dir / Path(f"c4-train.{args.c4_file_index:05d}-of-01024")
    filenames = [directory / f"{i:03d}-of-064.jsonl.gz" for i in range(NUM_FILES)]

    logger.info(f"Reading assertions from \"{directory}\"")
    # with Pool(NUM_FILES) as p:
    #     assertion_lists = p.map(load_one_assertion_file, filenames)
    assertion_lists = []
    for filename in filenames:
        # logger.info(f"{filename}")
        assertion_lists.append(load_one_assertion_file(filename))
    logger.info(f"{sum(len(l) for l in assertion_lists):,} assertions read")

    url_file = f"{WORKING_DIR}/urls_w_similarity/{args.c4_file_index:05d}-of-01024.csv"
    logger.info(f"Reading URLs with similarity file \"{url_file}\"")
    with open(url_file) as f:
        reader = csv.DictReader(f, fieldnames=["subject", "url", "count", "similarity"])
        good_su_pairs = {(row["subject"], row["url"]) for row in reader if float(row["similarity"]) >= args.threshold}
    logger.info(f"There are {len(good_su_pairs):,} good subject-URL pairs")

    # subjects = get_subject_list(args.subjects)

    logger.info(f"Reading subject file \"{args.subjects}\"")
    subjects = {}
    with open(args.subjects) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["subject"]
            t = (row["type"], row["super_subject"])
            if s not in subjects:
                subjects[s] = set()
            subjects[s].add(t)

    logger.info(f"Filtering assertions for {len(subjects):,} subjects")
    filtered_assertions = get_assertions_of_subjects(subjects, assertion_lists, args.c4_file_index, good_su_pairs)
    logger.info(f"Got {len(filtered_assertions):,} filtered assertions")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / Path(f"c4-train.{args.c4_file_index:05d}-of-01024.csv.gz")
    logger.info(f"Writing to \"{output_file}\"")
    with gzip.open(output_file, "wt") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "assertion_id", "subject_type",
                                               "super_subject"])
        writer.writeheader()
        writer.writerows(filtered_assertions)

    logger.info("Done")


if __name__ == '__main__':
    main()
