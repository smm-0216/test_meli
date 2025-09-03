import joblib
from pathlib import Path
from typing import Sequence, List, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FeatureEngineering:
    """
    Runtime Feature Engineering for production.

    Only the encoder is loaded from disk. 
    The following are provided via constructor:
      - kept_categories (set of valid categories for 'J')
      - final_columns (list of final feature names used by the model)

    Processing steps (same as training):
      1) Drop ['C', 'K']
      2) Convert ['Q','R','Monto'] (object with commas) to float
      3) Map rare categories in 'J' -> 'OTHERS' using kept_categories
      4) Apply OneHotEncoder on 'J'
      5) Concatenate encoded features and drop 'J'
      6) Reorder to final_columns (missing -> 0, extra -> removed)
    """

    def __init__(
        self,
        encoder_path: str | Path,
        kept_categories: Set[str],
        final_columns: List[str],
        drop_cols: Sequence[str] = ("C", "K"),
        cat_col: str = "J",
        numeric_object_cols: Sequence[str] = ("Q", "R", "Monto"),
    ) -> None:
        self.encoder_path = Path(encoder_path)
        self.drop_cols = tuple(drop_cols)
        self.cat_col = cat_col
        self.numeric_object_cols = tuple(numeric_object_cols)
        self.kept_categories_: Set[str] = set(map(str, kept_categories)) | {"OTHERS"}
        self.final_columns_: List[str] = list(final_columns)
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {self.encoder_path}")
        self.encoder_: OneHotEncoder = joblib.load(self.encoder_path)
        self.ohe_feature_names_: np.ndarray = self.encoder_.get_feature_names_out([self.cat_col])

    def _clean_numeric_objects(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numeric_object_cols:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .replace({"": np.nan})
                    .astype(float)
                )
        return X

    def _map_rare(self, s: pd.Series) -> pd.Series:
        s = s.astype(str)
        return s.where(s.isin(self.kept_categories_), other="OTHERS")

    def _align_to_final_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.final_columns_ if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0.0
        return X[self.final_columns_]

    def run(self, payload: dict) -> pd.DataFrame:
        if self.encoder_ is None or self.ohe_feature_names_ is None:
            raise RuntimeError("Encoder not properly loaded.")
        if self.cat_col not in payload:
            raise ValueError(f"Missing categorical column '{self.cat_col}' in payload.")

        X = pd.DataFrame([payload])
        cols_to_drop = [c for c in self.drop_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        X = self._clean_numeric_objects(X)
        X[self.cat_col] = self._map_rare(X[self.cat_col])
        ohe = self.encoder_.transform(X[[self.cat_col]])
        ohe_df = pd.DataFrame(ohe, columns=self.ohe_feature_names_, index=X.index)
        X = pd.concat([X.drop(columns=[self.cat_col]), ohe_df], axis=1)
        X = self._align_to_final_columns(X)
        return X
