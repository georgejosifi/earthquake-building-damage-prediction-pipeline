import pandas as pd
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)


class ZeroTransformer(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return pd.DataFrame()

class ColumnRenameTransformer(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def __init__(self, renames: list[str]):
        self.renames = renames

    def fit(self, X: pd.DataFrame, y=None):
        # Nothing to fit
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X.columns = self.renames
        return X
