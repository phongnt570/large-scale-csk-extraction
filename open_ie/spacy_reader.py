import logging
from pathlib import Path
from typing import Union, List

import spacy
from spacy.tokens import DocBin, Doc

logger = logging.getLogger(__name__)

logger.info("Loading SpaCy model")
nlp = spacy.load("en_core_web_md")


def read_one_spacy_file(filename: Union[str, Path]) -> List[Doc]:
    doc_bin = DocBin().from_disk(filename)
    return list(doc_bin.get_docs(nlp.vocab))


def read_one_spacy_folder(folder: Union[str, Path], num_files) -> List[Doc]:
    folder = Path(folder)

    filenames = [folder / f"{i:03d}-of-{num_files:03d}.spacy" for i in range(num_files)]

    docs = []
    for filename in filenames:
        docs.extend(read_one_spacy_file(filename))

    return docs
