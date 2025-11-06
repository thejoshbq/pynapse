# lot.py

from typing import List
from bean import Sample

class Population:
    def __init__(
        self,
        samples: List[Sample],
        name: str = "POP",
        authors: List[str] = None,
        description: str = None,
        **kwargs
    ):
        self.name = name
