import gzip
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def read_one_file(filename) -> List[Dict[str, str]]:
    logger.info(f"Reading \"{filename}\"...")
    try:
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error occurred in file {filename}: {e}")
        rows = []

    logger.info(f"Reading \"{filename}\" done! Size: {len(rows)}.")
    return rows
