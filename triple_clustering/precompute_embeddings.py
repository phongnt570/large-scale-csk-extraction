"""Based on https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/computing-embeddings/computing_embeddings_mutli_gpu.py"""

import argparse
import csv
import gzip
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict

from sentence_transformers import SentenceTransformer

from app_config import WORKING_DIR

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"

NUM_BATCHES = 64

MIN_FREQ = 3

csv.field_size_limit(sys.maxsize)


def make_sentence(triple: Dict[str, str]) -> str:
    return f"{triple['subject']} {triple['predicate']} {triple['object']}"


# Important, you need to shield your code with if __name__.
# Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ind", type=int, required=True)
    args = parser.parse_args()

    assert 0 <= args.ind < NUM_BATCHES

    input_file = f"{WORKING_DIR}/grouped_triples_geq_{MIN_FREQ}/triples-{args.ind:03d}-of-{NUM_BATCHES:03d}.csv.gz"
    logger.info(f"Read triples from \"{input_file}\"")
    triples = []
    with gzip.open(input_file, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append({
                "triple_id": row["triple_id"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "count": len(row["assertion_ids"].split("|")),
                "subject_type": row["subject_type"],
                "super_subject": row["super_subject"],
            })
    logger.info(f"There are {len(triples):,} triples")

    logger.info(f"Make sentences")
    sentences = [make_sentence(triple) for triple in triples]

    # Define the model
    logger.info(f"Load SentenceTransformer model \"{MODEL_NAME}\"")
    model = SentenceTransformer(MODEL_NAME)

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(sentences, pool)
    logger.info(f"Embeddings computed. Shape: {embeddings.shape}")

    # Write pickle
    output_dir = Path(f"{WORKING_DIR}/grouped_triples_geq_{MIN_FREQ}_embeddings/")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"embeddings-{args.ind:03d}-of-{NUM_BATCHES:03d}.pkl"
    logger.info(f"Writing results to\"{output_file}\"")
    with open(output_file, "wb") as f_out:
        pickle.dump({"triples": triples, "embeddings": embeddings}, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional: Stop the processes in the pool
    model.stop_multi_process_pool(pool)

    logger.info("Done")
