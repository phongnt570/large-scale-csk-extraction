import argparse
import csv
import logging
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Union, NamedTuple, Any, Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app_config import WORKING_DIR

logging.basicConfig(level=logging.INFO,
                    format='[%(processName)s] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S')

logger = logging.getLogger(__name__)

NUM_EMBEDDING_PARTS = 64
MAX_TRIPLES = int(5e4)

MIN_FREQ = 3


class FineGrainedSubject(NamedTuple):
    subject: str
    subject_type: str
    super_subject: str


def read_triples_and_embeddings(filename: Union[str, Path]) -> Dict[str, Any]:
    logger.info(f"Reading embeddings from \"{filename}\"")
    with open(filename, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data


def get_data_for_subject(subject: FineGrainedSubject, all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    triples = [t for d in all_data for t in d["triples"] if (
            t["subject"] == subject.subject and
            t["subject_type"] == subject.subject_type and
            t["super_subject"] == subject.super_subject)][:MAX_TRIPLES]

    embeddings = np.array(
        [e for d in all_data for t, e in zip(d["triples"], d["embeddings"]) if (
                t["subject"] == subject.subject and
                t["subject_type"] == subject.subject_type and
                t["super_subject"] == subject.super_subject)][:MAX_TRIPLES])

    return {
        "subject": subject.subject,
        "triples": triples,
        "embeddings": embeddings,
        "subject_type": subject.subject_type,
        "super_subject": subject.super_subject,
    }


def clustering(triples, embeddings, distance_threshold: float = 0.5) -> Dict[int, List[Dict[str, Any]]]:
    # Normalize the embeddings to unit length
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_triples = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_triples:
            clustered_triples[cluster_id] = []
        clustered_triples[cluster_id].append(triples[sentence_id])

    return clustered_triples


def cluster_for_one_subject(dataset: Dict[str, Any]) -> List[Dict[str, str]]:
    if len(dataset["triples"]) == 0:
        return []
    elif len(dataset["triples"]) == 1:
        rep = dataset["triples"][0]
        return [{
            "subject": rep["subject"],
            "predicate": rep["predicate"],
            "object": rep["object"],
            "count": rep["count"],
            "triple_ids": rep["triple_id"],
            "subject_type": rep["subject_type"],
            "super_subject": rep["super_subject"],
        }]

    rep_triples = []

    logger.info(f"  - Clustering subject \"{dataset['subject']}\" - {(len(dataset['triples'])):,} triples")
    clustered_triples = clustering(dataset["triples"], dataset["embeddings"])
    sorted_clustered_triples = sorted(clustered_triples.values(), key=lambda cls: sum(int(t["count"]) for t in cls),
                                      reverse=True)
    for cluster in sorted_clustered_triples:
        c = sorted(cluster, key=lambda t: int(t["count"]), reverse=True)
        rep = c[0]
        rep_triples.append({
            "subject": rep["subject"],
            "predicate": rep["predicate"],
            "object": rep["object"],
            "count": sum(int(t["count"]) for t in cluster),
            "triple_ids": "|".join([t["triple_id"] for t in cluster]),
            "subject_type": rep["subject_type"],
            "super_subject": rep["super_subject"],
        })
    logger.info(
        f"  + There are {(len(sorted_clustered_triples)):,} cluster triples for subject \"{dataset['subject']}\"")

    return rep_triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=f"{WORKING_DIR}/clustered_triples/")
    parser.add_argument("--embeddings_dir", type=str,
                        default=f"{WORKING_DIR}/grouped_triples_geq_{MIN_FREQ}_embeddings/")
    parser.add_argument("--num_processors", type=int, default=64)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)

    args = parser.parse_args()

    logger.info(f"Reading subject file \"{args.subjects}\"")
    subjects = []
    with open(args.subjects) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.append(FineGrainedSubject(
                subject=row["subject"],
                subject_type=row["type"],
                super_subject=row["super_subject"]
            ))
    logger.info(f"There are {len(subjects):,} subjects in total")

    out_num_parts = int(len(subjects) / args.step) + 1
    out_part_id = int(args.id / args.step)

    subjects = subjects[args.id:(args.id + args.step)]
    logger.info(f"Selected subjects: {subjects}")

    logger.info("Reading triples and precomputed embeddings from disk")
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_filenames = [embeddings_dir / f"embeddings-{i:03d}-of-{NUM_EMBEDDING_PARTS:03d}.pkl" for i in
                            range(NUM_EMBEDDING_PARTS)]
    with Pool(args.num_processors) as p:
        all_data = p.map(read_triples_and_embeddings, embeddings_filenames)
    # all_data = []
    # for embeddings_filename in embeddings_filenames:
    #     all_data.append(read_triples_and_embeddings(embeddings_filename))
    logger.info(f"There are {(sum(len(data['triples']) for data in all_data)):,} triples in total")

    logger.info("Getting triples to be computed")
    datasets = [get_data_for_subject(subject, all_data) for subject in subjects]
    logger.info(
        f"There are {(sum(len(dataset['triples']) for dataset in datasets)):,} triples of the {len(subjects)} "
        f"subjects to be processed")

    logger.info("Clustering started")
    with Pool(args.num_processors) as p:
        all_rep_triples = p.map(cluster_for_one_subject, datasets)

    logger.info(f"There are {(sum(len(rep_triples) for rep_triples in all_rep_triples)):,} clustered triples in total")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"triples-{out_part_id:05d}-of-{out_num_parts:05d}.csv"
    logger.info(f"Writing results to \"{output_file}\"")
    with open(output_file, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "count", "triple_ids", "subject_type",
                                               "super_subject"])
        writer.writeheader()
        writer.writerows([rep for rep_triples in all_rep_triples for rep in rep_triples])

    logger.info("Finished.")


if __name__ == '__main__':
    main()
