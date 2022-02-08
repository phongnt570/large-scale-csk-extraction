from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule

RELATIONS = {
    "/r/Antonym",
    "/r/Synonym",
    "/r/MannerOf",
    "/r/LocatedNear",
    # "/r/HasSubevent",
    # "/r/RelatedTo",
    "/r/DistinctFrom",
}


class IgnoreRelationModule(FilteringModule):
    def __init__(self):
        self.ignored_relations = RELATIONS

    def validate(self, cluster: Dict[str, Any]) -> bool:
        relation = cluster["predicted_relation"]

        return relation not in self.ignored_relations

    def get_name(self):
        return f"IgnoreRelationModule - {len(self.ignored_relations)} relations"
