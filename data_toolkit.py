
from enum import Enum
from typing import List, Union
import random
import statistics

# === Version Control ===
__version__ = "2025.06.23"
__release__ = "r01"
__expires__ = datetime.date(2025, 6, 24)

import datetime
if datetime.date.today() >= __expires__:
    raise RuntimeError(
        f"⚠️ This version ({__version__}, {__release__}) expired on {__expires__}. "
        "Please download the latest toolkit from your teacher."
    )

class DataType(Enum):
    QUALITATIVE = "qualitative"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"

class VariableRole(Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"

class RandomDataGenerator:
    def __init__(
        self,
        count: int,
        min_val: Union[int, float] = None,
        max_val: Union[int, float] = None,
        is_integer: bool = True,
        qualitative_values: List[str] = None
    ):
        self.count = count
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer
        self.qualitative_values = qualitative_values

    def generate(self) -> List[Union[int, float, str]]:
        if self.qualitative_values:
            return [random.choice(self.qualitative_values) for _ in range(self.count)]
        elif self.is_integer:
            return [random.randint(self.min_val, self.max_val) for _ in range(self.count)]
        else:
            return [random.uniform(self.min_val, self.max_val) for _ in range(self.count)]

class DataContainer:
    def __init__(self, name: str, description: str, data_type: DataType):
        self.name = name
        self.description = description
        self.data_type = data_type
        self.role: VariableRole = None
        self.data: List[Union[int, float, str]] = []

    def set_role(self, role: VariableRole):
        self.role = role

    def load_data(self, source: Union[List[Union[int, float, str]], RandomDataGenerator]):
        if isinstance(source, list):
            self.data = source
        elif isinstance(source, RandomDataGenerator):
            self.data = source.generate()
        else:
            raise TypeError("Unsupported data source type")

class SummaryStats:
    def __init__(self, container: DataContainer):
        self.container = container
        if self.container.data_type == DataType.QUALITATIVE:
            raise TypeError("Cannot compute numerical summary on qualitative data.")

    def mean(self):
        return statistics.mean(self.container.data)

    def median(self):
        return statistics.median(self.container.data)

    def mode(self):
        return statistics.mode(self.container.data)
