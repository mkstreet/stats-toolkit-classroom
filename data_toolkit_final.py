# ============================
# Enums
# ============================
from enum import Enum
from typing import List, Tuple, Optional, Union
from datetime import datetime
from pydantic import BaseModel, validator
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import curve_fit

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
# Expiration check
# ============================
__version__ = "1.3.0"
__release__ = "2025-06-25"
__expires__ = "2025-07-03"

if datetime.now() > datetime.strptime(__expires__, "%Y-%m-%d"):
    print("\u26a0\ufe0f This toolkit has expired. Please download the latest version from your instructor.")
else:
    print(f"INFO:  The data tool kit will expire after {datetime.strptime(__expires__, '%Y-%m-%d')}")

# ============================
# Extensions to DataContainer
# ============================
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

    def getDescription(self):
        return self.description

    def _get_xy_data_zip(self):
        return zip(*self.data)

# ============================
# LogisticGrowthModel with plot
# ============================
class LogisticGrowthModel:
    def __init__(self, dc: DataContainer):
        self._dc = dc
        self.params = None

    def fit(self):
        x_vals, y_vals = self._dc._get_xy_data_zip()
        x = np.array(list(x_vals))
        y = np.array(list(y_vals))
        guess = [max(y), 1, np.median(x)]
        self.params, _ = curve_fit(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), x, y, p0=guess)

    def predict_y(self, x):
        L, k, x0 = self.params
        return L / (1 + np.exp(-k * (x - x0)))

    def summary(self):
        L, k, x0 = self.params
        return f"logistic: L={L:.2f}, k={k:.2f}, x0={x0:.2f}"

    def plot_logistic(self):
        L, k, x0 = self.params
        x_vals, y_vals = self._dc._get_xy_data_zip()
        x = np.array(list(x_vals))
        y = np.array(list(y_vals))
        x_range = np.linspace(min(x), max(x), 100)
        y_fit = L / (1 + np.exp(-k * (x_range - x0)))
        plt.plot(x, y, 'bo', label='Data')
        plt.plot(x_range, y_fit, 'r-', label='Logistic Fit')
        plt.title("Logistic Growth Fit")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.show()
