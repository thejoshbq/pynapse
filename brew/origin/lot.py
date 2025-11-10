# lot.py

from typing import List
from bean import Sample

class Population:
    def __init__(
        self,
        samples: List[Sample],
        name: str = "Brew Population",
        description: str = None,
    ):
        self._samples = samples
        self._description = description
        self._name = name

    def __str__(self):
        return f"{self._name}:\n{self._description}"

    def get_name(self):
        return self._name
