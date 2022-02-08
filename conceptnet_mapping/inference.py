import argparse
import logging
from typing import List

import pymongo
import torch
from bson import ObjectId
from pymongo import UpdateOne
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizerFast

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


def get_object_tags(assertion):
    start = assertion["source"]["positions"]["obj_start"]
    end = assertion["source"]["positions"]["obj_end"]
    return assertion["source"]["tags"][start:end]


def assertion_has_adjective_object(assertion):
    tags = ["JJ", "JJR", "JJS"]
    object_tags = set(get_object_tags(assertion))
    for tag in tags:
        if tag in object_tags:
            return True
    return False


def assertion_has_passive_verb_object(assertion):
    tags = ["VBN"]
    object_tags = set(get_object_tags(assertion))
    for tag in tags:
        if tag in object_tags:
            return True
    return False


def assertion_has_noun_object(assertion):
    tags = ["NN", "NNS"]
    object_tags = set(get_object_tags(assertion))
    for tag in tags:
        if tag in object_tags:
            return True
    return False


def has_lot_of_adjectives(cluster, assertions):
    assertions = [a for a in assertions if a["object"] == cluster["object"]]
    has_adj_obj = [assertion_has_adjective_object(a) for a in assertions]
    if sum(has_adj_obj) / len(has_adj_obj) < 0.5:
        return False
    return True


def has_lot_of_nouns(cluster, assertions):
    assertions = [a for a in assertions if a["object"] == cluster["object"]]
    has_noun_obj = [assertion_has_noun_object(a) for a in assertions]
    if sum(has_noun_obj) / len(has_noun_obj) < 0.5:
        return False
    return True


def has_lot_passive_verb_object(cluster, assertions):
    assertions = [a for a in assertions if a["object"] == cluster["object"]]
    has_pass_obj = [assertion_has_passive_verb_object(a) for a in assertions]
    if sum(has_pass_obj) / len(has_pass_obj) < 0.5:
        return False
    return True


def preprocess_triples(rows, sep_token):
    return [f"{row['subject']} {sep_token} {row['predicate']} {row['object']}" for row in rows]


def predict_relation(rows, model, tokenizer, device, batch_size=64) -> List[str]:
    num_batches = len(rows) / batch_size
    if int(num_batches) != num_batches:
        num_batches = int(num_batches) + 1
    num_batches = int(num_batches)

    logger.info(f"Batch size: {batch_size}. Number of batches: {num_batches:,}")

    # Predictive model
    results = []
    for i in range(0, len(rows), batch_size):
        # logger.info(f"Batch {int(i / batch_size):,} / {num_batches:,}")
        batch_of_rows = rows[i:(i + batch_size)]
        inputs = tokenizer(preprocess_triples(batch_of_rows, tokenizer.sep_token), truncation=True, padding=True,
                           return_tensors="pt").to(device)
        outputs = model(**inputs)
        results.extend(outputs[0].softmax(1).argmax(-1).tolist())

    relations = [model.config.id2label[idx] for idx in results]

    # Rules
    for i in range(len(relations)):
        relation = relations[i]
        row = rows[i]

        if relation in {"/r/IsA"}:
            assertions = list(get_openie_assertions(row))
            if has_lot_of_adjectives(row, assertions) and not has_lot_of_nouns(row, assertions):
                relations[i] = "/r/HasProperty"
            elif has_lot_passive_verb_object(row, assertions) and not has_lot_of_nouns(row, assertions):
                relations[i] = "/r/ReceivesAction"

        if relation == "/r/HasProperty" and row["predicate"] != "be":
            relations[i] = "/r/ReceivesAction"

        if row["predicate"] == "symbolize" or row["object"].startswith("a symbol of ") or row["object"].startswith(
                "symbol of ") or row["object"].startswith("the symbol of "):
            relations[i] = "/r/SymbolOf"

        if row["object"].startswith("an emblem of ") or row["object"].startswith(
                "emblem of ") or row["object"].startswith("the emblem of "):
            relations[i] = "/r/SymbolOf"

        if row["predicate"] in {"need", "require"}:
            relations[i] = "/r/HasPrerequisite"

        if row["predicate"] in {"contain", "include"}:
            relations[i] = "/r/HasA"

        if row["predicate"] in {"be related to", "relate to"}:
            relations[i] = "/r/RelatedTo"

    return relations


def should_concat_po(predicted_relation, old_pred):
    if predicted_relation == "/r/CapableOf":
        if old_pred not in {"be capable of", "can", "be able to"}:
            return True
    # elif predicted_relation in {"/r/UsedFor"}:
    #     return True
    elif predicted_relation in {"/r/HasProperty", "/r/IsA", "/r/ReceivesAction"}:
        if old_pred != "be":
            return True

    return False


def postprocess_object(row, predicted_relation) -> str:
    new_obj = row["object"]

    old_obj = row["object"]
    old_pred = row["predicate"]

    if should_concat_po(predicted_relation, old_pred):
        new_obj = old_pred + " " + old_obj

    if predicted_relation in {"/r/HasProperty", "/r/IsA", "/r/ReceivesAction"}:  # , "/r/UsedFor"}:
        if old_pred.startswith("be "):
            new_obj = old_pred[len("be "):] + " " + old_obj

    elif predicted_relation == "/r/PartOf" and old_obj.startswith("a part of "):
        new_obj = old_obj[len("a part of "):]

    elif predicted_relation == "/r/PartOf" and old_obj.startswith("part of "):
        new_obj = old_obj[len("part of "):]

    elif predicted_relation == "/r/PartOf" and old_obj.startswith("the part of "):
        new_obj = old_obj[len("the part of "):]

    elif predicted_relation == "/r/SymbolOf" and old_obj.startswith("a symbol of "):
        new_obj = old_obj[len("a symbol of "):]

    elif predicted_relation == "/r/SymbolOf" and old_obj.startswith("the symbol of "):
        new_obj = old_obj[len("the symbol of "):]

    elif predicted_relation == "/r/SymbolOf" and old_obj.startswith("symbol of "):
        new_obj = old_obj[len("symbol of "):]

    elif predicted_relation == "/r/SymbolOf" and old_obj.startswith("an emblem of "):
        new_obj = old_obj[len("an emblem of"):]

    elif predicted_relation == "/r/SymbolOf" and old_obj.startswith("emblem of "):
        new_obj = old_obj[len("emblem of"):]

    elif predicted_relation == "/r/SymbolOf" and old_obj.startswith("the emblem of "):
        new_obj = old_obj[len("the emblem of"):]

    elif predicted_relation == "/r/UsedFor":
        old_pred_toks: List[str] = old_pred.split()
        if "to" in old_pred_toks:
            idx = old_pred_toks.index("to")
            if idx < len(old_pred_toks) - 1:
                new_obj = " ".join(old_pred_toks[(idx + 1):]) + " " + old_obj

    elif predicted_relation == "/r/Desires":
        old_pred_toks: List[str] = old_pred.split()
        if "to" in old_pred_toks:
            idx = old_pred_toks.index("to")
            if idx < len(old_pred_toks) - 1:
                new_obj = " ".join(old_pred_toks[(idx + 1):]) + " " + old_obj

    new_obj = new_obj.replace("n’t", "not")
    new_obj = new_obj.replace("n't", "not")

    return new_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_id", type=int, required=True)
    parser.add_argument("--num_batches", type=int, required=True)
    parser.add_argument("--id_file", type=str, required=True)

    args = parser.parse_args()

    assert args.gpu >= 0

    model_path = args.model

    device = torch.device(f"cuda:{args.gpu}")

    # Load data from Mongo DB
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

    # load model and tokenizer
    # logger.info(f"Load tokenizer from \"{model_path}\"")
    # tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    logger.info(f"Load model from \"{model_path}\"")
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # inference
    logger.info("Predict relations")
    predicted_relations = predict_relation(clusters, model=model, tokenizer=tokenizer, device=device)

    # new object
    logger.info("Postprocess objects")
    predicted_objects = [postprocess_object(cluster, relation) for cluster, relation in
                         zip(clusters, predicted_relations)]

    # write new results
    queries = []
    for cluster, relation, obj in zip(clusters, predicted_relations, predicted_objects):
        queries.append(
            UpdateOne({"_id": cluster["_id"]}, {"$set": {"predicted_relation": relation, "predicted_object": obj}}))

    logger.info(f"Bulk write {len(queries):,} queries")
    CLUSTERS_COL.bulk_write(queries, ordered=False)

    logger.info("Done")


if __name__ == '__main__':
    main()
