import argparse
import csv
import gzip
import json
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Tuple, Set

from ascent_openie import oie_from_spacy_sent

from app_config import WORKING_DIR
from .spacy_reader import read_one_spacy_file

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MIN_FILE_INDEX = 0
MAX_FILE_INDEX = 1023

NUM_FILES = 64


def run_open_ie_for_file(files: Tuple[Union[str, Path], Union[str, Path]], good_urls: Set[str] = None):
    input_file, output_file = files

    logger.info(f"File \"{input_file}\" reading")
    docs = read_one_spacy_file(input_file)

    logger.info(f"File \"{input_file}\" extracting")
    assertions = []
    for doc in docs:
        if good_urls is not None and doc.user_data["url"] not in good_urls:
            continue
        for sent in doc.sents:
            assertions.extend(oie_from_spacy_sent(sent, get_appos=True))

    logger.info(f"File \"{output_file}\" writing")
    with gzip.open(output_file, "wt") as f:
        for a in assertions:
            if a["subject"] and a["predicate"] and a["object"]:
                f.write(json.dumps(a))
                f.write("\n")

    logger.info(f"File \"{output_file}\" done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_index", type=int, required=False)
    parser.add_argument("--processors", type=int, default=64)
    parser.add_argument("--in_filename", type=str, required=False)
    parser.add_argument("--out_filename", type=str, required=False)
    parser.add_argument("--filter_urls", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6)

    args = parser.parse_args()

    if args.in_filename and args.out_filename:
        run_open_ie_for_file((args.in_filename, args.out_filename))
        return

    assert MIN_FILE_INDEX <= args.file_index <= MAX_FILE_INDEX

    input_dir = Path(f"{WORKING_DIR}/spacy_output/c4-train.{args.file_index:05d}-of-01024/")
    output_dir = Path(f"{WORKING_DIR}/openie_output/c4-train.{args.file_index:05d}-of-01024/")
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Input folder: \"{input_dir}\"")

    filenames = [input_dir / f"{i:03d}-of-{NUM_FILES:03d}.spacy" for i in range(NUM_FILES)]
    output_filenames = [output_dir / f"{i:03d}-of-{NUM_FILES:03d}.jsonl.gz" for i in range(NUM_FILES)]

    url_file = f"{WORKING_DIR}/urls_w_similarity/{args.file_index:05d}-of-01024.csv"
    good_urls = None
    if args.filter_urls:
        logger.info(f"Using filtered URLs. Reading URLs with similarity file \"{url_file}\"")
        with open(url_file) as f:
            reader = csv.DictReader(f, fieldnames=["subject", "url", "count", "similarity"])
            good_urls = {row["url"] for row in reader if float(row["similarity"]) >= args.threshold}
        logger.info(f"There are {len(good_urls):,} good URLs")

    func = partial(run_open_ie_for_file, good_urls=good_urls)
    with Pool(args.processors) as p:
        p.map(func, zip(filenames, output_filenames))


if __name__ == '__main__':
    main()
