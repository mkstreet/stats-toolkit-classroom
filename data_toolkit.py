
import random
import statistics
import datetime
from enum import Enum, auto
from typing import List, Union

# === Version Control ===
__version__ = "2025.06.17"
__release__ = "r01"
__expires__ = datetime.date(2025, 6, 22)

if datetime.date.today() > __expires__:
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

class DistributionType(Enum):
    UNIFORM = auto()
    NORMAL = auto()
    LEFT_SKEW = auto()
    RIGHT_SKEW = auto()

class RandomDataGenerator:
    def __init__(
        self,
        count: int,
        min_val: Union[int, float] = 0,
        max_val: Union[int, float] = 100,
        is_integer: bool = True,
        qualitative_values: List[str] = None
    ):
        # === Input validation ===
        MAX_ALLOWED_COUNT = 20000
        if not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        if count > MAX_ALLOWED_COUNT:
            raise ValueError(f"count exceeds safe maximum of {MAX_ALLOWED_COUNT}.")

        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise TypeError("min_val and max_val must be numbers")

        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")

        if qualitative_values is not None:
            if not isinstance(qualitative_values, list):
                raise TypeError("qualitative_values must be a list")
            if not all(isinstance(item, str) for item in qualitative_values):
                raise TypeError("All items in qualitative_values must be strings")
            if len(qualitative_values) == 0:
                raise ValueError("qualitative_values list must not be empty")

        self.count = count
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer
        self.qualitative_values = qualitative_values

    def generate_uniform_data(self):
        if self.is_integer:
            return [random.randint(self.min_val, self.max_val) for _ in range(self.count)]
        else:
            return [random.uniform(self.min_val, self.max_val) for _ in range(self.count)]

    def generate_data(self, distribution=DistributionType.UNIFORM, **kwargs):
        if self.qualitative_values:
            return [random.choice(self.qualitative_values) for _ in range(self.count)]
        if distribution == DistributionType.UNIFORM:
            return self.generate_uniform_data()
        elif distribution == DistributionType.NORMAL:
            mean = kwargs.get('mean', 50)
            stddev = kwargs.get('stddev', 10)
            return [random.gauss(mean, stddev) for _ in range(self.count)]
        elif distribution == DistributionType.RIGHT_SKEW:
            return [self.min_val + (random.random() ** 2) * (self.max_val - self.min_val) for _ in range(self.count)]
        elif distribution == DistributionType.LEFT_SKEW:
            return [self.min_val + (random.random() ** 0.5) * (self.max_val - self.min_val) for _ in range(self.count)]
        else:
            raise ValueError("Unsupported distribution type.")

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
            self.data = source.generate_data()
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
