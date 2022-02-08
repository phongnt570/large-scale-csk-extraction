from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule

RELATION_2_OBJ_WORDS = {
    "/r/AtLocation": ["position", "positions", "condition", "conditions", "location", "locations", "situation",
                      "situations", "case", "cases", "same", "demand", "supply", "process", "form", "shape", "mode",
                      "talk"],

    "/r/HasA": ["time", "chance", "place", "problem", "idea", "advantage", "disadvantage", "ability", "access", "fun",
                "plan", "choice", "reason", "comment", "interest", "trouble", "way", "name", "difficulty", "tendency"],

    "/r/HasProperty": ["of"],

    "/r/SimilarTo": ["same"],

    "/r/IsA": ["any", "only", "first", "last"],

    "/r/UsedFor": ["minute", "minutes", "hour", "hours", "day", "days", "month", "months", "year", "years", "century",
                   "centuries", "use", "used"],

}


class RelationIgnoreObjectWordModule(FilteringModule):
    def __init__(self):
        self.relation_2_obj_words = RELATION_2_OBJ_WORDS

    def validate(self, cluster: Dict[str, Any]) -> bool:
        relation = cluster["predicted_relation"]
        arg2 = cluster["predicted_object"]

        arg2_toks = set(arg2.split())

        obj_words = self.relation_2_obj_words.get(relation, [])
        for w in obj_words:
            if w in arg2_toks:
                return False

        return True

    def get_name(self):
        return f"RelationIgnoreObjectWordModule - {len(self.relation_2_obj_words)} relations"
