import gzip
import json
from pathlib import Path
from typing import Union, Any, Dict, List, NamedTuple


def load_one_assertion_file(filename: Union[str, Path]) -> List[Dict[str, Any]]:
    assertions = []
    with gzip.open(filename, "rt") as f:
        for line in f:
            assertions.append(json.loads(line))
    return assertions


class AssertionId(NamedTuple):
    c4_id: int
    part_id: int
    asst_id: int

    def __str__(self):
        return f"{self.c4_id:05d}-{self.part_id:03d}-{self.asst_id:07d}"


def convert_id(aid: str) -> AssertionId:
    toks = aid.split("-")

    assert len(toks) == 3
    assert len(toks[0]) == 5
    assert len(toks[1]) == 3
    assert len(toks[2]) == 7

    c4_id = int(toks[0])
    part_id = int(toks[1])
    asst_id = int(toks[2])

    assert 0 <= c4_id < 1024
    assert 0 <= part_id < 64
    assert asst_id >= 0

    return AssertionId(c4_id=c4_id, part_id=part_id, asst_id=asst_id)
