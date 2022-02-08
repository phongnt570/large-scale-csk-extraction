from typing import Dict, Any

from final_filter.abc_filtering_module import FilteringModule


class PerplexityModule(FilteringModule):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def validate(self, cluster: Dict[str, Any]) -> bool:
        return cluster["perplexity"] <= self.threshold

    def get_name(self):
        return f"PerplexityModule - threshold = {self.threshold}"
