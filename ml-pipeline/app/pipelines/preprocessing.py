from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "IQRClipper":
        data = pd.DataFrame(X).copy()
        for column in data.columns:
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[column] = float(q1 - self.factor * iqr)
            self.upper_bounds_[column] = float(q3 + self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(X).copy()
        for column in data.columns:
            lower = self.lower_bounds_.get(column, -np.inf)
            upper = self.upper_bounds_.get(column, np.inf)
            data[column] = data[column].clip(lower=lower, upper=upper)
        return data
