import pandas as pd
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)


class InfrequentCategoryMergerTransformer(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def __init__(self, threshhold):
        self.threshhold = threshhold
        self.merge_value = "other"
        
    def fit(self, X: pd.DataFrame, y = None):
        self.values_to_merge_in_each_column = {}
        for col in X.columns:
            value_counts = X[col].value_counts(normalize= True)
            if len(value_counts[value_counts < self.threshhold].index.tolist()) > 0:
                self.values_to_merge_in_each_column[col] = value_counts[value_counts < self.threshhold].index.tolist()
            
        return self
        
        
    def transform(self, X: pd.DataFrame, y= None):
        for col in self.values_to_merge_in_each_column.keys():
            if not isinstance(self.values_to_merge_in_each_column[col][0],str):
                self.merge_value = -1
            else:
                self.merge_value = "other"
                
            X[col] = X[col].replace(to_replace= self.values_to_merge_in_each_column[col], value= self.merge_value)
            
        return X
    