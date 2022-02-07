import argparse
import logging
import os
import time
from pathlib import Path

import spacy
from spacy.tokens import DocBin

from .c4_reader import read_one_file

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

WORKING_DIR = "/path/to/your/working_directory".rstrip("/")

MIN_FILE_INDEX = 0
MAX_FILE_INDEX = 1023

NUM_BATCHES = 64

logger.info("Load SpaCy model...")
nlp = spacy.load("en_core_web_md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_index", type=int, required=True)
    parser.add_argument("--processors", type=int, default=128)
    parser.add_argument("--num_docs", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    assert MIN_FILE_INDEX <= args.file_index <= MAX_FILE_INDEX

    actual_file_name = f"{WORKING_DIR}/C4/c4-train.{args.file_index:05d}-of-01024.json.gz"

    documents = read_one_file(actual_file_name)
    if args.num_docs > 0:
        documents = documents[:args.num_docs]

    start_time = time.process_time()

    pipe = nlp.pipe([document["text"] for document in documents],
                    n_process=args.processors, batch_size=args.batch_size)
    docs = [doc for doc in pipe]
    for i, doc in enumerate(docs):
        doc.user_data["timestamp"] = documents[i]["timestamp"]
        doc.user_data["url"] = documents[i]["url"]

    end_time = time.process_time()

    process_time = end_time - start_time

    logger.info(f"Total processing time: {process_time:.4f} seconds.")
    logger.info(f"Average processing time per document: {(process_time / len(docs)):.4f} seconds.")

    logger.info(f"Write to disk...")
    output_batch_size = int(len(docs) / NUM_BATCHES) + 1
    if len(docs) % NUM_BATCHES == 0:
        output_batch_size -= 1
    output_folder = f"{WORKING_DIR}/spacy_output/c4-train.{args.file_index:05d}-of-01024/"
    os.makedirs(output_folder, exist_ok=True)
    for i in range(NUM_BATCHES):
        start = i * output_batch_size
        end = (i + 1) * output_batch_size
        doc_bin = DocBin(store_user_data=True, docs=docs[start:end])
        doc_bin.to_disk(Path(output_folder) / f"{i:03d}-of-{NUM_BATCHES:03d}.spacy")
    logger.info("Done!")


if __name__ == '__main__':
    main()
