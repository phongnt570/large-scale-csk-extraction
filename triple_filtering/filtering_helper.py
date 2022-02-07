from typing import Any, Dict

from .dictionaries import IGNORED_OBJECTS, IGNORED_PREDICATES, IGNORED_PO_PAIRS


def is_likely_valid(a: Dict[str, Any]) -> bool:
    subj = a["subject"]
    pred = a["predicate"]
    obj = a["object"]

    # all element must be non-empty
    if not (subj and pred and obj):
        return False

    # subject must be different from object
    if subj == obj.split()[-1]:
        return False

    # object must not be pronoun or ignored phrases
    if obj in IGNORED_OBJECTS:
        return False

    # predicate must not be in ignored list
    if pred in IGNORED_PREDICATES:
        return False

    # predicate-object must not be in ignored list
    if (pred, obj) in IGNORED_PO_PAIRS:
        return False

    # predicate/object length must be greater than 1
    if len(pred) <= 1 or len(obj) <= 1:
        return False

    # object should not be too long
    if len(obj.split()) > 5:
        return False

    # ignore apposition
    if len(a["source"]["positions"]["pred_positions"]) == 0:
        return False

    return True
