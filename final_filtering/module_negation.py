from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule


class NegationModule(FilteringModule):
    def __init__(self):
        pass

    def validate(self, cluster: Dict[str, Any]) -> bool:
        predicate = cluster["predicate"]
        arg2 = cluster["predicted_object"]

        if "not" in arg2.split() or "not" in predicate.split():
            return False

        if "no" in arg2.split() or "no" in predicate.split():
            return False

        return True

    def get_name(self):
        return "NegationModule"
