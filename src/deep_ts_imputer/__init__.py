"""deep-ts-imputer: deep-learning time-series imputation framework.

A domain-agnostic toolkit for reconstructing missing values in multivariate
time series using recurrent and convolutional-recurrent neural networks.

Originally developed for hydrological / water-quality reconstruction in the
Seine estuary (Janbain et al., Water/MDPI 2023), generalised here for any
multivariate time-series imputation task.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deep-ts-imputer")
except PackageNotFoundError:  # package not installed
    __version__ = "0.1.0"

__all__ = ["__version__"]
