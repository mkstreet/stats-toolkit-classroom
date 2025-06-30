<<<<<<< HEAD
# ============================
# Enums
# ============================
from enum import Enum

class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    BINOMIAL = "binomial"
    QUALITATIVE = "qualitative"

class DataType(Enum):
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    PANEL = "panel"

class ModelType(Enum):
    LINEAR = "linear"
    POLYFIT = "polyfit"
    LOGISTIC_GROWTH = "logistic_growth"
    DECAY = "decay"
    INTERPOLATION = "interpolation"


class HeaderOption(Enum):
    HEADERS = "headers"
    NOHEADERS = "noheaders"


# ============================
# Core: DataContainer
# ============================
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


__version__ = "1.3.0"
__release__ = "2025-06-25"
__expires__ = "2025-07-03"

# Expiration check
if datetime.now() > datetime.strptime(__expires__, "%Y-%m-%d"):
    print("⚠️ This toolkit has expired. Please download the latest version from your instructor.")
else:
	print(f"INFO:  The data tool kit will expire after {datetime.strptime(__expires__, "%Y-%m-%d")}")


class VariableRole(Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"

class ValidatedContainer(BaseModel):
    name: str
    description: str
    data_type: DataType

    @validator('name', 'description')
    def non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Must be a non-empty string')
        return v

class DataContainer(ValidatedContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.role: Optional[VariableRole] = None
        self.data: List[Tuple[float, float]] = []
        self.headers: Optional[List[str]] = None
        self.df: Optional[pl.DataFrame] = None
        self.data_matrix: Optional[List[Tuple[float, ...]]] = None

    def set_role(self, role: VariableRole):
        if not isinstance(role, VariableRole):
            raise TypeError("Role must be a VariableRole")
        self.role = role

    def load_time_series(self, x_list: List[float], y_list: List[float]):
        if len(x_list) != len(y_list):
            raise ValueError("x and y must be same length.")
        self.data = list(zip(x_list, y_list))

    def load_univariate(self, y_list: List[float]):
        self.data = list(enumerate(y_list))

    def load_from_csv(self, path: str, has_headers: bool = True,
                      x_col: Union[int, str] = 0, y_col: Union[int, str] = 1):
        df = pl.read_csv(path, has_header=has_headers)
        self.df = df
        self.headers = df.columns

        if isinstance(x_col, int):
            x = df[:, x_col].to_list()
=======
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
>>>>>>> 9b3e2876ff147fbdebff4d92d3be26317e7e32b4
        else:
            x = df[x_col].to_list()

<<<<<<< HEAD
        if isinstance(y_col, int):
            y = df[:, y_col].to_list()
        else:
            y = df[y_col].to_list()

        if len(x) != len(y):
            raise ValueError("X and Y columns must be the same length")

        self.data = list(zip(x, y))

    def load_multivariate_from_csv(self, path: str, header_option: HeaderOption = HeaderOption.HEADERS,
                                    cols: Optional[Union[List[int], List[str]]] = None):
        has_headers = header_option == HeaderOption.HEADERS
        df = pl.read_csv(path, has_header=has_headers)
        self.df = df
        self.headers = df.columns

        if cols is None:
            sub_df = df
        else:
            try:
                sub_df = df.select(cols)
            except Exception as e:
                raise ValueError(f"Failed to select columns {cols}: {e}")

        self.data_matrix = [tuple(row) for row in sub_df.rows()]

    def _get_xy_data_zip(self):
        return zip(*self.data)



# ============================
# Plotting
# ============================
class PlotData:
    def __init__(self, dc: DataContainer):
        self._dc = dc

    def scatter(self, title: Optional[str] = None):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        x = list(x_vals)
        y = list(y_vals)
        plt.scatter(x, y)
        plt.title(title if title else self._dc.description)
        plt.xlabel(self._dc.name + " X")
        plt.ylabel(self._dc.name + " Y")
        plt.grid(True)
        plt.show()

    def plot_model_fit(self, model, title: Optional[str] = None):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        x = np.array(list(x_vals))
        y = np.array(list(y_vals))
        y_fit = model.predict_y(x)
        plt.plot(x, y, 'bo', label='Data')
        plt.plot(x, y_fit, 'r-', label=model.summary())
        plt.title(title if title else self._dc.description)
        plt.xlabel(self._dc.name + " X")
        plt.ylabel(self._dc.name + " Y")
        plt.legend()
        plt.grid(True)
        plt.show()


# ============================
# Models
# ============================
from scipy.stats import linregress

class LinearModel:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        self.result = None

    def fit(self):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        self.result = linregress(list(x_vals), list(y_vals))

    def predict_y(self, x):
        return self.result.slope * np.array(x) + self.result.intercept

    def summary(self):
        return f"y = {self.result.slope:.2f}x + {self.result.intercept:.2f}"


class PolyFitModel:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        self.best_model = None
        self.best_degree = None
        self.coeffs = None

    def fit(self):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        x = np.array(list(x_vals))
        y = np.array(list(y_vals))
        best_rmse = float("inf")
        for deg in [2, 3, 4, 5]:
            coeffs = np.polyfit(x, y, deg)
            y_pred = np.polyval(coeffs, x)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                self.best_degree = deg
                self.coeffs = coeffs

    def predict_y(self, x):
        return np.polyval(self.coeffs, x)

    def summary(self):
        return f"polyfit deg={self.best_degree}, coeffs={np.round(self.coeffs, 2)}"


class LogisticGrowthModel:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        # Placeholder logic for logistic fitting
    def fit(self):
        pass
    def predict_y(self, x):
        return x  # Placeholder
    def summary(self):
        return "logistic growth (placeholder)"


class DecayModel:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        # Placeholder logic for decay fitting
    def fit(self):
        pass
    def predict_y(self, x):
        return x  # Placeholder
    def summary(self):
        return "decay model (placeholder)"


class InverseEstimator:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        # Placeholder logic for inverse estimation
    def estimate_x_for_y(self, y):
        return y  # Placeholder


# ============================
# Fit Interface
# ============================
class FitModel:
    def __init__(self, dc: DataContainer, model_type: ModelType):
        self.dc = dc
        self.model_type = model_type
        self.model = self._build_model(model_type)

    def _build_model(self, model_type: ModelType):
        if model_type == ModelType.LINEAR:
            return LinearModel(self.dc)
        elif model_type == ModelType.POLYFIT:
            return PolyFitModel(self.dc)
        elif model_type == ModelType.LOGISTIC_GROWTH:
            return LogisticGrowthModel(self.dc)
        elif model_type == ModelType.DECAY:
            return DecayModel(self.dc)
        elif model_type == ModelType.INTERPOLATION:
            return InverseEstimator(self.dc)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self):
        if hasattr(self.model, "fit"):
            self.model.fit()

    def summary(self):
        if hasattr(self.model, "summary"):
            return self.model.summary()
        return f"{self.model_type.value} model has no summary."

    def predict_y(self, x):
        if hasattr(self.model, "predict_y"):
            return self.model.predict_y(x)
        raise NotImplementedError("This model does not support y prediction.")

    def predict_x(self, y):
        if hasattr(self.model, "predict_x"):
            return self.model.predict_x(y)
        elif hasattr(self.model, "estimate_x_for_y"):
            return self.model.estimate_x_for_y(y)
        raise NotImplementedError("This model does not support x prediction.")
=======
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
>>>>>>> 9b3e2876ff147fbdebff4d92d3be26317e7e32b4
