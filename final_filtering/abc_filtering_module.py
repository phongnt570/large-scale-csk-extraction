from typing import Any, Dict


class FilteringModule(object):
    def validate(self, cluster: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def __str__(self):
        return self.get_name()
