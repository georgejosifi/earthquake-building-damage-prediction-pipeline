import pandas as pd
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)


municipality_coordinates = {
    28: (27.6572, 84.0590),
    19: (28.6029, 83.3362),
    29: (27.8940, 83.1220),
    2: (28.2774, 83.5816),
    23: (26.9835, 87.3215),
    24: (27.6174, 87.3016),
    14: (28.0889, 83.2934),
    15: (27.1780, 87.0524),
    30: (28.2246, 83.6987),
    5: (27.8253, 83.6348),
    1: (28.2622, 84.0167),
    13: (27.3240, 86.5047),
    11: (27.6588, 85.3247),
    3: (27.6710, 85.4298),
    22: (27.9447, 84.2279),
    25: (28.2765, 84.3542),
    16: (27.6992, 86.7416),
    0: (27.5291, 84.3542),
    9: (28.0197, 83.8049),
    12: (28.1755, 85.3963),
    18: (27.1838, 86.7819),
    6: (27.5285, 85.6435),
    26: (27.5546, 85.0233),
    10: (27.9711, 84.8985),
    17: (27.9512, 85.6846),
    8: (28.2964, 84.8568),
    7: (27.9194, 85.1661),
    20: (27.2569, 85.9713),
    21: (27.7784, 86.1752),
    4: (27.3193, 86.1039),
    27: (27.7172, 85.3240)
}

class AddMunicipalityCoordinatesTransformer(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X["north_coordinate"] = X["geo_level_1_id"].map(
            {k: municipality_coordinates[k][0] for k in municipality_coordinates})
        X["east_coordinate"] = X["geo_level_1_id"].map(
            {k: municipality_coordinates[k][1] for k in municipality_coordinates})

        return X
