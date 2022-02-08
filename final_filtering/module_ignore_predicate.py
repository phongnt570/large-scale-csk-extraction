from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule

IGNORED_PREDICATES = {
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "be like",
    "be defined in",
    "be defined by",
    "include",  # NOTE: very noisy statements
    "star",
    "be directed by",
}


class IgnorePredicatesModule(FilteringModule):
    def __init__(self):
        self.ignored_predicates = IGNORED_PREDICATES

    def validate(self, cluster: Dict[str, Any]) -> bool:
        predicate = cluster["predicate"]

        return predicate not in self.ignored_predicates

    def get_name(self):
        return f"IgnorePredicatesModule - {len(self.ignored_predicates)} predicates"
