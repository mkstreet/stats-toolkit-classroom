# unified_data_toolkit.py

"""
Unified Data Toolkit for Student Use
====================================

This module provides tools for generating, storing, and analyzing data
in a structured and pedagogically useful way. It merges two diverged
versions into a single, comprehensive toolkit.

Features:
- Data containers with role and type awareness
- Random data generation (uniform, normal, skewed)
- Scatter plotting and linear model fitting
- Version control with expiration logic
- Extended models: Growth, Elimination, Inverse Estimator
"""

import random
import statistics
import datetime
from enum import Enum, auto
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# === Version Control ===
__version__ = "2025.06.25"
__release__ = "r01"
__expires__ = datetime.date(2025, 7, 2)

if datetime.date.today() > __expires__:
    raise RuntimeError(
        f"\u26a0\ufe0f This version ({__version__}, {__release__}) expired on {__expires__}. "
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


class DataContainer:
    def __init__(self, name: str, description: str, data_type: DataType):
        self.name = name
        self.description = description
        self.data_type = data_type
        self.role: VariableRole = None
        self.data: List[Union[int, float, str]] = []

    def set_role(self, role: VariableRole):
        self.role = role

    def load_data(self, source: Union[List[Union[int, float, str]], 'RandomDataGenerator']):
        if isinstance(source, list):
            self.data = source
        elif isinstance(source, RandomDataGenerator):
            self.data = source.generate_data()
        else:
            raise TypeError("Unsupported data source type")

    def load_time_series(self, x_list: List[float], y_list: List[float]):
        self.data = list(zip(x_list, y_list))

    def _get_xy_data_zip(self):
        return zip(*self.data)

    def getDescription(self):
        return self.description


class PlotData:
    def __init__(self, dc: DataContainer):
        self._dc = dc

    def scatter(self):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        plt.scatter(x_vals, y_vals)
        plt.title(f"{self._dc.getDescription()}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


class FitModel:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        self.x = np.array([x for x, _ in self._dc.data])
        self.y = np.array([y for _, y in self._dc.data])
        self.coeffs = None

    def fit_linear_model(self, start: float, end: float):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        filtered = [(x, y) for x, y in zip(x_vals, y_vals) if start <= x <= end]

        if len(filtered) < 2:
            raise ValueError("Not enough data points in specified range.")

        x_fit, y_fit = zip(*filtered)
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        self.coeffs = (slope, intercept)

        y_pred = np.polyval([slope, intercept], x_fit)
        residuals = np.array(y_fit) - np.array(y_pred)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(y_fit) - np.mean(y_fit))**2)
        r_squared = 1 - ss_res / ss_tot

        plt.scatter(x_vals, y_vals)
        plt.plot(x_fit, y_pred, color='red')
        plt.title("Linear Fit")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

        return {"slope": slope, "intercept": intercept, "r_squared": r_squared}

    def predict(self, x_val):
        if self.coeffs is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.coeffs[0] * x_val + self.coeffs[1]

    def summary(self):
        if self.coeffs is None:
            return "Model not fit yet."
        slope, intercept = self.coeffs
        return f"f(t) = {slope:.3f} * t + {intercept:.3f}"

    def fit_logistic(self):
        guess = [max(self.y), 1, np.median(self.x)]
        self.params, _ = curve_fit(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), self.x, self.y, p0=guess)

    def plot_logistic(self):
        L, k, x0 = self.params
        x_range = np.linspace(min(self.x), max(self.x), 100)
        y_fit = L / (1 + np.exp(-k * (x_range - x0)))
        plt.plot(self.x, self.y, 'bo', label='Data')
        plt.plot(x_range, y_fit, 'r-', label='Logistic Fit')
        plt.title("Logistic Growth Fit")
        plt.legend()
        plt.show()

    def fit_inverse(self):
        guess = [1, 1]
        self.params, _ = curve_fit(lambda x, a, b: a / (x + b), self.x, self.y, p0=guess)

    def predict_inverse(self, y_target):
        a, b = self.params
        return (a / y_target) - b


class RandomDataGenerator:
    def __init__(
        self,
        count: int,
        min_val: Union[int, float] = 0,
        max_val: Union[int, float] = 100,
        is_integer: bool = True,
        qualitative_values: List[str] = None
    ):
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


class SummaryStats:
    def __init__(self, container: DataContainer):
        self.container = container

    def mean(self):
        if self.container.data_type == DataType.QUALITATIVE:
            raise TypeError("Mean is not defined for qualitative data.")
        return statistics.mean(self.container.data)

    def median(self):
        if self.container.data_type == DataType.QUALITATIVE:
            raise TypeError("Median is not defined for qualitative data.")
        return statistics.median(self.container.data)

    def mode(self):
        try:
            return statistics.mode(self.container.data)
        except statistics.StatisticsError:
            return "No unique mode"
