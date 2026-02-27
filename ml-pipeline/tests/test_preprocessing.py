from __future__ import annotations

import pandas as pd

from app.pipelines.feature_engineering import FeatureEngineer
from app.pipelines.preprocessing import IQRClipper


def test_iqr_clipper_clips_outliers():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 200]})

    clipper = IQRClipper(factor=1.5)
    clipper.fit(data)
    transformed = clipper.transform(data)

    assert transformed["x"].max() < 200


def test_feature_engineer_adds_expected_columns():
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "category": ["x", "y"],
        }
    )
    transformer = FeatureEngineer()
    transformed = transformer.fit_transform(data)

    assert "numeric_sum" in transformed.columns
    assert "numeric_mean" in transformed.columns
    assert "missing_count" in transformed.columns
    assert transformed["numeric_sum"].tolist() == [4.0, 6.0]
