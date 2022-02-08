import re
from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule

REGEX_PATTERNS = [
    r"(day|week|month|year)s*[- ]old",
    r"of the family",
    r"founded in \d+",
    r"established in \d+",
    r"introduced in \d+",
    r"placed in \d+",
    r"open in \d+",
    r"rebuilt in \d+",
    r"built in \d+",
    r"completed in \d+",
    r"a \d+$",
    r"for instance",
    r"for example",
    r"ranked in ",
    r"ranked among ",
    r"begin at \d+",
    r"start at \d+",
    r"originate in \d+",
    r"run to \d+",
    r"introduced to ",
    r"contacted at ",
    r"closed at \d+",
    r"open at \d+",
    r"www\.",
    r"https*//:\.",
    r"^[\d\.,]+$"
]

PREDICATE_REGEX = {
    "/r/AtLocation": [r" use$", r" season$", r" risk$", r" vicinity$", r" proximity$", r" distance$", r" choices*$",
                      r"^\d+$", r"^need of ", r" capacity$", r" time$", r" heat$",
                      r" size$", r"mid-\d+"],
    "/r/UsedFor": [r" purposes*$"],
    # "/r/Causes": [r" problems*$", r" issues*$"],
}


class RegexModule(FilteringModule):
    def __init__(self):
        self.patterns = REGEX_PATTERNS
        self.pred2patterns = PREDICATE_REGEX

    def validate(self, cluster: Dict[str, Any]) -> bool:
        rel = cluster["predicted_relation"]
        arg2 = cluster["predicted_object"]

        for pattern in self.patterns + self.pred2patterns.get(rel, []):
            p = re.compile(pattern)
            if p.search(arg2):
                return False

        return True

    def get_name(self):
        return f"RegexModule - " \
               f"{len(self.patterns) + sum(len(regexes) for regexes in self.pred2patterns.values())} patterns"
