from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule

BAD_WORDS = [
    "my",
    "mine",
    "your",
    "yours",
    "his",
    "her",
    "its",

    "you",
    "us",
    "me",

    "this",
    "that",
    "these",
    "those",
    "which",
    "whom",
    "what",
    "how",
    "when",
    "where",
    "why",

    "etc",
    "named",

    "something",
    "anything",
    "nothing",
    "everything",
]


class ObjectContainsWordModule(FilteringModule):
    def __init__(self):
        self.bad_words = BAD_WORDS

    def validate(self, cluster: Dict[str, Any]) -> bool:
        arg2 = cluster["predicted_object"]

        arg2_toks = set(arg2.split())

        for bad_word in self.bad_words:
            if bad_word in arg2_toks:
                return False

        return True

    def get_name(self):
        return f"ObjectContainsWordModule - {len(self.bad_words)} bad words"
