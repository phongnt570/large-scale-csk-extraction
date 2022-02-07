IGNORED_PO_PAIRS = {
    ("be", "seen"),
    ("be", "used"),
    ("be", "the fact"),
    ("be taken", "care"),
    ("make", "sense"),
    ("make", "sure"),
    ("make", "use"),
    ("pay", "attention"),
    ("be", " capable"),
    ("be", " able"),
    ("come in", " size"),
    ("play", " an important role"),
    ("play", " important role"),
    ("make", " difference"),
    ("do", " thing"),
    ("do", " the same"),
    ("take", " part"),
    ("give", " way"),
    ("give", " a way"),
}

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
