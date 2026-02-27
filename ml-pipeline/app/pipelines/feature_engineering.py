from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(X).copy()
        numeric_columns = list(data.select_dtypes(include=["number"]).columns)

        if numeric_columns:
            data["numeric_sum"] = data[numeric_columns].sum(axis=1)
            data["numeric_mean"] = data[numeric_columns].mean(axis=1)

        data["missing_count"] = data.isna().sum(axis=1)
        return data
