import pandas as pd
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)


# Example custom transformer to use in pipeline
class SumColumnsTransformer(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if X.shape[1] <= 1:
            return X

        res = X.iloc[:,0]
        for col in X.columns[1:]:
            res += X[col]
        
        return res
