# unified_data_toolkit.py

"""
Unified Data Toolkit for Student Use
====================================

This module helps you collect, generate, analyze, and graph data in simple ways.
It includes tools for working with numbers, categories, graphs, and models.
Students can use it in science, math, or any project that involves data.

What you can do with it:
- Store data with labels and types
- Generate random or patterned data
- Plot scatterplots of data
- Fit straight-line and curved models to data
- Calculate statistics like average, middle value, or most common value

NOTE: This version will stop working after a certain date so students always get the latest version.
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
    UNIFORM = "uniform"
    NORMAL = "normal"
    SKEW_LEFT = "skew_left"
    SKEW_RIGHT = "skew_right"

# --- Data Container ---
class DataContainer:
    def __init__(self, name, description, data_type):
        self.name = name
        self.description = description
        self.data_type = data_type
        self.role = None
        self.data = []
        self.time_series = None

    def set_role(self, role):
        self.role = role

    def load_data(self, data):
        self.data = list(data)

    def load_time_series(self, x, y):
        self.time_series = (list(x), list(y))

    def plot_scatter(self):
        if self.time_series:
            x, y = self.time_series
            plt.scatter(x, y)
            plt.xlabel("Time")
            plt.ylabel(self.description)
            plt.title(f"{self.name} Scatter Plot")
            plt.grid(True)
            plt.show()
        else:
            raise ValueError("No time series loaded to plot.")

    def fit_linear_model(self, start=None, end=None):
        if not self.time_series:
            raise ValueError("No time series data available.")
        x, y = self.time_series
        if start is None: start = 0
        if end is None: end = len(x)
        x_segment = x[start:end]
        y_segment = y[start:end]
        coeffs = np.polyfit(x_segment, y_segment, 1)
        slope, intercept = coeffs
        return LinearModel(slope, intercept)

# --- Linear Model ---
class LinearModel:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def summary(self):
        print(f"y = {self.slope:.3f} * x + {self.intercept:.3f}")

    def predict(self, x_val):
        return self.slope * x_val + self.intercept

    def get_time_for_value(self, target_y):
        if self.slope == 0:
            raise ValueError("Slope is zero; cannot solve for x.")
        return (target_y - self.intercept) / self.slope

# --- Summary Stats ---
class SummaryStats:
    def __init__(self, data):
        self.data = sorted(data)

    def mean(self):
        return sum(self.data) / len(self.data)

    def median(self):
        n = len(self.data)
        mid = n // 2
        if n % 2 == 0:
            return (self.data[mid - 1] + self.data[mid]) / 2
        else:
            return self.data[mid]

    def mode(self):
        freq = {}
        for val in self.data:
            freq[val] = freq.get(val, 0) + 1
        max_count = max(freq.values())
        modes = [val for val, count in freq.items() if count == max_count]
        return modes

# --- Random Generator ---
class RandomGenerator:
    def __init__(self):
        pass

    def generate_numeric(self, count, min_val=0, max_val=100, skew=DistributionType.UNIFORM):
        if count > 20000:
            raise ValueError("Maximum count is 20,000")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")

        if skew == DistributionType.UNIFORM:
            return [random.uniform(min_val, max_val) for _ in range(count)]
        elif skew == DistributionType.NORMAL:
            mean = (min_val + max_val) / 2
            stddev = (max_val - min_val) / 6
            return [max(min(random.gauss(mean, stddev), max_val), min_val) for _ in range(count)]
        elif skew == DistributionType.SKEW_RIGHT:
            data = np.random.beta(a=2, b=5, size=count)
            return list(min_val + (max_val - min_val) * data)
        elif skew == DistributionType.SKEW_LEFT:
            data = np.random.beta(a=5, b=2, size=count)
            return list(min_val + (max_val - min_val) * data)
        else:
            raise ValueError("Unsupported distribution type.")

    def generate_qualitative(self, count, choices):
        if count > 20000:
            raise ValueError("Maximum count is 20,000")
        return [random.choice(choices) for _ in range(count)]
