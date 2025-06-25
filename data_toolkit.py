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

 
"""


import random
import statistics
import datetime
from enum import Enum, auto
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np



# === Version Control ===
__version__ = "2025.06.25"
__release__ = "r01"
__expires__ = datetime.date(2025, 7, 2)

if datetime.date.today() > __expires__:
    raise RuntimeError(
        f"⚠️ This version ({__version__}, {__release__}) expired on {__expires__}. "
        "Please download the latest toolkit from your teacher."
    )




class DataType(Enum):
    """Defines the types of data a DataContainer can represent."""

    QUALITATIVE = "qualitative"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"



class VariableRole(Enum):
    """Defines the role of a variable in analysis (independent or dependent)."""

    INDEPENDENT = "independent"
    DEPENDENT = "dependent"



class DistributionType(Enum):
    """Defines supported random distributions for data generation."""

    UNIFORM = auto()
    NORMAL = auto()
    LEFT_SKEW = auto()
    RIGHT_SKEW = auto()



class DataContainer:
        """Stores a named dataset with type and role metadata, and provides loading methods."""

        def __init__(self, name: str, description: str, data_type: DataType):
            """
            Parameters:
                name (str): Short label for this dataset.
                description (str): A longer description for context.
                data_type (DataType): Type of data being stored (qualitative, discrete, continuous).
                data (List[tuple]): Paired (x, y) values or single-variable data.
            """

            self.name = name
            self.description = description
            self.data_type = data_type
            self.role: VariableRole = None
            self.data: List[Union[int, float, str]] = []

        def set_role(self, role: VariableRole):
            """Assigns whether the variable is independent or dependent."""
    
            self.role = role

        def load_data(self, source: Union[List[Union[int, float, str]], RandomDataGenerator]):
            """
            Loads data from a list or a generator into this container.

            Parameters:
                source (list or RandomDataGenerator): Data source to load.

            Raises:
                TypeError: If source is an unsupported type.
            """

            if isinstance(source, list):
                self.data = source
            elif isinstance(source, RandomDataGenerator):
                self.data = source.generate_data()
            else:
                raise TypeError("Unsupported data source type")


        def load_time_series(self, x_list: List[float], y_list: List[float]):
            """Loads a series of (x, y) data pairs."""
            self.data = list(zip(x_list, y_list))

        def _get_xy_data_zip(self):
            return zip(*self.data)

        def getDescription(self):
            return self.description




class PlotData:
        _dc = None

        def __init__(self, dc:DataContainer):
            self._dc = description

        def scatter(self):
            x_vals, y_vals = self._dc._get_xy_data_zip()
            plt.scatter(x_vals, y_vals)
            plt.title(f"{self._dc.getDescription()}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.show()


class FitModel:
        _dc = None

        def __init__(self, dc:DataContainer):
            self._dc = description


       def fit_linear_model(self, start: float, end: float):
            """
            Fits and plots a linear model to data between specified x-values.

            Args:
                start (float): Lower bound of x-values to include.
                end (float): Upper bound of x-values to include.

            Returns:
                dict: Dictionary with slope, intercept, and r_squared.
            """
            x_vals, y_vals = self._dc._get_xy_data_zip()
            filtered = [(x, y) for x, y in zip(x_vals, y_vals) if start <= x <= end]

            if len(filtered) < 2:
                raise ValueError("Not enough data points in specified range.")

            x_fit, y_fit = zip(*filtered)
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            y_pred = np.polyval([slope, intercept], x_fit)
            residuals = y_fit - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - ss_res / ss_tot

            plt.scatter(x_vals, y_vals)
            plt.plot(x_fit, y_pred, color='red')
            plt.title("Linear Fit")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.show()

            return {"slope": slope, "intercept": intercept, "r_squared": r_squared}








class SummaryStats:
    """Provides statistical summaries (mean, median, mode) for numeric data in a DataContainer."""

        def __init__(self, container: DataContainer):
        """
        Parameters:
            container (DataContainer): Must contain numeric data.

        Raises:
            TypeError: If data type is QUALITATIVE.
        """

        self.container = container
        if self.container.data_type == DataType.QUALITATIVE:
            raise TypeError("Cannot compute numerical summary on qualitative data.")

        def mean(self):
        """Returns the mean of the data."""

        return statistics.mean(self.container.data)

        def median(self):
        """Returns the median of the data."""

        return statistics.median(self.container.data)

        def mode(self):
        """Returns the mode of the data."""

        return statistics.mode(self.container.data)


class RandomDataGenerator:
    """Generates random datasets using various distributions, optionally qualitative or numeric."""

        def __init__(
        self,

        count: int,
        min_val: Union[int, float] = 0,
        max_val: Union[int, float] = 100,
        is_integer: bool = True,
        qualitative_values: List[str] = None
    ):
        """
        Parameters:
            count (int): Number of values to generate.
            min_val (int or float): Minimum value (numeric only).
            max_val (int or float): Maximum value (numeric only).
            is_integer (bool): Whether to generate integers or floats.
            qualitative_values (list of str, optional): If provided, generates qualitative values from this list.

        Raises:
            ValueError: For invalid count, value ranges, or empty qualitative list.
            TypeError: If types are incorrect.
        """
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
        """Generates uniformly distributed data (int or float based on is_integer)."""

        if self.is_integer:
            return [random.randint(self.min_val, self.max_val) for _ in range(self.count)]
        else:
            return [random.uniform(self.min_val, self.max_val) for _ in range(self.count)]

        def generate_data(self, distribution=DistributionType.UNIFORM, **kwargs):
        """
        Generates data using the specified distribution.

        Parameters:
            distribution (DistributionType): The distribution to use.
            kwargs: Extra parameters for normal distribution (mean, stddev).

        Returns:
            list of generated data values

        Raises:
            ValueError: If unsupported distribution is selected.
        """

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
