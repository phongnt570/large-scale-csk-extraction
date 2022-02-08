import json
from pathlib import Path
from typing import Dict, Any, Union

from final_filter.abc_filtering_module import FilteringModule


class ConceptNetHyponymModule(FilteringModule):
    def __init__(self, filename: Union[str, Path]):
        self.filename = filename

        with open(filename) as f:
            self.subject2hypos = {}
            for line in f:
                data = json.loads(line)
                self.subject2hypos[data["subject"]] = data["hyponyms"]

    def validate(self, cluster: Dict[str, Any]) -> bool:
        subject = cluster["subject"]
        relation = cluster["predicted_relation"]
        arg2 = cluster["predicted_object"]

        arg2_toks = arg2.split()

        if relation == "/r/IsA" and subject in self.subject2hypos:
            for h in self.subject2hypos[subject]:
                if h in arg2_toks or f"{h}s" in arg2_toks or f"{h}es" in arg2_toks:
                    return True
            return False

        return True

    def get_name(self):
        return f"ConceptNetHyponymModule - \"{self.filename}\""
