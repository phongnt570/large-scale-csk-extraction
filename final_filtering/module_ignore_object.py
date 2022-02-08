from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule

IGNORED_OBJECTS = {
    "in",
    "on",
    "at",
    "of",
    "from",
    "by",

    "mine",
    "yours",
    "ours",
    "its",
    "theirs",

    "etc",
    "for etc",
    "of course",
    "other",
    "the one",
    "same",
    "the same",
    "the fact",
    "most cases",

    "likely", "ready", "present", "alive", "dead", "needed", "presented", "represented",
    "due",
    "on top",
    "the top",
    "top",

    "this",
    "that",
    "these",
    "those",

    "what",
    "which",
    "where",
    "when",
    "who",
    "whom",
    "how",
    "why",

    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",

    "the first", "the second",
    "the third", "the fourth",
    "the fifth", "the sixth",
    "the seventh", "the eighth",
    "the ninth", "the tenth",

    "the best", "best", "better",
    "the worst", "worst", "worse",

    "all", "more",
    "less", "much", "many",

    "n/a",

    "be", "do", "have", "live", "die", "go", "make", "come", "help", "eat",
    "find", "see", "say", "know", "get", "let", "become", "allow", "ask",

    "the ability", "the right",

    "gone", "done", "born", "taken", "found", "known", "located",
    "seen", "used", "listed", "intended",

    "me", "myself", "us", "ourselves",
    "you", "yourself", "yourselves",
    "them", "themselves", "her", "herself",
    "him", "himself", "it", "itself",

    "make sure", "make history",

    "this case", "this situation", "the time", "this time",
    "today", "this year", "last year", "this month", "this week",
    "any way", "several",

    "damn", "other words", "the here", "the now", "the present", "the moment", "now", "here", "there",

    "the", "take place", "come to end", "such", "non", "on the left", "on the right", "play role",

}


class IgnoreObjectsModule(FilteringModule):
    def __init__(self):
        self.ignored_objects = IGNORED_OBJECTS

    def validate(self, cluster: Dict[str, Any]) -> bool:
        arg2 = cluster["predicted_object"]
        obj = cluster["object"]

        return arg2 not in self.ignored_objects and obj not in self.ignored_objects

    def get_name(self):
        return f"IgnoreObjectsModule - {len(self.ignored_objects)} objects"
