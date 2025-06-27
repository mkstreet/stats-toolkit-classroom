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
    """Tells what kind of data you are working with."""
    QUALITATIVE = "qualitative"  # Categories, like "red" or "tall"
    DISCRETE = "discrete"        # Whole numbers, like 1 or 2 or 5
    CONTINUOUS = "continuous"    # Any value, like 3.6 or 4.95


class VariableRole(Enum):
    """Tells if a variable is the cause or the effect in a graph or experiment."""
    INDEPENDENT = "independent"  # The thing you change (x-axis)
    DEPENDENT = "dependent"      # The thing you measure (y-axis)


class DistributionType(Enum):
    """Ways to randomly generate numbers."""
    UNIFORM = auto()     # All values are equally likely
    NORMAL = auto()      # Bell curve (most in the middle)
    LEFT_SKEW = auto()   # More high values than low
    RIGHT_SKEW = auto()  # More low values than high


class DataContainer:
    """
    Stores your data along with labels and extra info.

    Parameters:
    - name (str): Short name of the variable (like "height")
    - description (str): Longer description (like "Height of plants in cm")
    - data_type (DataType): Tells if the data is numbers or categories
    """
    def __init__(self, name: str, description: str, data_type: DataType):
        self.name = name
        self.description = description
        self.data_type = data_type
        self.role: VariableRole = None
        self.data: List[Union[int, float, str]] = []

    def set_role(self, role: VariableRole):
        """Set whether this variable is independent (x-axis) or dependent (y-axis)."""
        self.role = role

    def load_data(self, source: Union[List[Union[int, float, str]], 'RandomDataGenerator']):
        """
        Loads data into this container.

        You can give it:
        - A list like [1, 2, 3, 4]
        - A RandomDataGenerator object that makes data for you
        """
        if isinstance(source, list):
            self.data = source
        elif isinstance(source, RandomDataGenerator):
            self.data = source.generate_data()
        else:
            raise TypeError("Unsupported data source type")

    def load_time_series(self, x_list: List[float], y_list: List[float]):
        """
        Loads paired data points like (time, value).

        Make sure:
        - x_list and y_list are the same length
        - each x matches a y (like time and height)
        """
        self.data = list(zip(x_list, y_list))

    def _get_xy_data_zip(self):
        return zip(*self.data)

    def getDescription(self):
        return self.description


class PlotData:
    """
    Draws a scatter plot using the data from a DataContainer.

    To use:
        PlotData(my_container).scatter()
    """
    def __init__(self, dc: DataContainer):
        self._dc = dc

    def scatter(self):
        """Shows a scatterplot of the data on a graph."""
        x_vals, y_vals = self._dc._get_xy_data_zip()
        plt.scatter(x_vals, y_vals)
        plt.title(f"{self._dc.getDescription()}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


class FitModel:
    """
    Fits math models to data in a DataContainer.

    You can:
    - Fit a straight line (linear)
    - Fit a curve (logistic or inverse)
    - Use the model to predict values
    """
    def __init__(self, dc: DataContainer):
        self._dc = dc
        self.x = np.array([x for x, _ in self._dc.data])
        self.y = np.array([y for _, y in self._dc.data])
        self.coeffs = None

    def fit_linear_model(self, start: float, end: float):
        """
        Finds the best-fit line for part of the data.

        Parameters:
        - start: The smallest x to include
        - end: The largest x to include

        Returns:
        A dictionary with slope, intercept, and r-squared score
        """
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
        """Predicts a y value from x using the linear model."""
        if self.coeffs is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.coeffs[0] * x_val + self.coeffs[1]

    def summary(self):
        """Shows the equation of the linear model."""
        if self.coeffs is None:
            return "Model not fit yet."
        slope, intercept = self.coeffs
        return f"f(t) = {slope:.3f} * t + {intercept:.3f}"

    def fit_logistic(self):
        """Fits a logistic growth curve to the data."""
        guess = [max(self.y), 1, np.median(self.x)]
        self.params, _ = curve_fit(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), self.x, self.y, p0=guess)

    def plot_logistic(self):
        """Draws the logistic model on top of the data."""
        L, k, x0 = self.params
        x_range = np.linspace(min(self.x), max(self.x), 100)
        y_fit = L / (1 + np.exp(-k * (x_range - x0)))
        plt.plot(self.x, self.y, 'bo', label='Data')
        plt.plot(x_range, y_fit, 'r-', label='Logistic Fit')
        plt.title("Logistic Growth Fit")
        plt.legend()
        plt.show()

    def fit_inverse(self):
        """Fits an inverse model (like 1/x) to the data."""
        guess = [1, 1]
        self.params, _ = curve_fit(lambda x, a, b: a / (x + b), self.x, self.y, p0=guess)

    def predict_inverse(self, y_target):
        """Gives an x-value that would create the given y-value using the inverse model."""
        a, b = self.params
        return (a / y_target) - b
