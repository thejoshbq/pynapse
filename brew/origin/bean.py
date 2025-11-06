# bean.py

from pathlib import Path
import scipy.io as sio
import warnings
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Union

class Sample:
    def __init__(
        self,
        name: str = "Brew Sample",

    ):
        self.name = name
