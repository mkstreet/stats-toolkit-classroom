"""
Unified Data Toolkit for Classroom Use
Version: 1.3.0
Release: 2025-06-25
Expires: 2025-08-31
"""

import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime
from typing import List, Optional

__version__ = "1.3.0"
__release__ = "2025-06-25"
__expires__ = "2025-08-31"

# Expiration check
if datetime.now() > datetime.strptime(__expires__, "%Y-%m-%d"):
    print("⚠️ This toolkit has expired. Please download the latest version from your instructor.")

class DataType(Enum):
    QUALITATIVE = 1
    DISCRETE = 2
    CONTINUOUS = 3

class VariableRole(Enum):
    INDEPENDENT = 1
    DEPENDENT = 2

class DistributionType(Enum):
    UNIFORM = 1
    NORMAL = 2
    SKEW_LEFT = 3
    SKEW_RIGHT = 4

class SummaryStats:
    @staticmethod
    def mean(data):
        return statistics.mean(data)
    @staticmethod
    def median(data):
        return statistics.median(data)
    @staticmethod
    def mode(data):
        try:
            return statistics.mode(data)
        except statistics.StatisticsError:
            return "No unique mode"

class RandomGenerator:
    def generate_numeric(self, count: int, min_val: float, max_val: float, dist: DistributionType):
        if count <= 0 or count > 20000:
            raise ValueError("Count must be between 1 and 20000.")
        if dist == DistributionType.UNIFORM:
            return [random.uniform(min_val, max_val) for _ in range(count)]
        elif dist == DistributionType.NORMAL:
            mu = (max_val + min_val) / 2
            sigma = (max_val - min_val) / 6
            return [min(max_val, max(min_val, random.gauss(mu, sigma))) for _ in range(count)]
        elif dist == DistributionType.SKEW_LEFT:
            return [max_val - (random.paretovariate(2) * (max_val - min_val) / 5) for _ in range(count)]
        elif dist == DistributionType.SKEW_RIGHT:
            return [min_val + (random.paretovariate(2) * (max_val - min_val) / 5) for _ in range(count)]
        else:
            raise ValueError("Unsupported distribution type.")

    def generate_qualitative(self, count: int, choices: List[str]):
        if not choices:
            raise ValueError("Choice list is empty.")
        return [random.choice(choices) for _ in range(count)]

class FitModel:
    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept

    def get_time_for_value(self, y: float) -> Optional[float]:
        if self.slope == 0:
            return None
        return (y - self.intercept) / self.slope

    def summary(self):
        print(f"Slope: {self.slope:.4f}")
        print(f"Intercept: {self.intercept:.4f}")

class DataContainer:
    def __init__(self, name: str, description: str, dtype: DataType):
        self.name = name
        self.description = description
        self.dtype = dtype
        self.role = None
        self.x = []
        self.y = []

    def set_role(self, role: VariableRole):
        self.role = role

    def load_data(self, x_vals: List[float], y_vals: List[float]):
        if len(x_vals) != len(y_vals):
            raise ValueError("x and y must be the same length.")
        self.x = x_vals
        self.y = y_vals

    def load_time_series(self, times: List[float], values: List[float]):
        self.load_data(times, values)

    def plot_scatter(self):
        plt.scatter(self.x, self.y)
        plt.title(f"Scatter Plot for {self.name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    def fit_linear_model(self, start: int, end: int) -> FitModel:
        if start < 0 or end > len(self.x) or start >= end:
            raise ValueError("Invalid start/end indices.")
        x_subset = self.x[start:end]
        y_subset = self.y[start:end]
        slope, intercept = np.polyfit(x_subset, y_subset, 1)
        return FitModel(slope, intercept)
