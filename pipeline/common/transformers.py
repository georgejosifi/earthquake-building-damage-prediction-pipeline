from enum import Enum

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import (Binarizer, KBinsDiscretizer, MaxAbsScaler,
                                   MinMaxScaler, Normalizer, OneHotEncoder,
                                   OrdinalEncoder, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.random_projection import GaussianRandomProjection

import category_encoders as ce

import custom_transformers.aggregate_transformers as aggregate_transformers
import custom_transformers.util_transformers as util_transformers
import custom_transformers.infrequent_category_merger_transformers as i_c_m_t
import custom_transformers.misc_transformers as misc_transformers


class ImputationAlgorithm(str, Enum):
    SimpleImputer = "SimpleImputer"
    KNNImputer = "KNNImputer"
    IterativeImputer = "IterativeImputer"


imputation_algorithms = {
    ImputationAlgorithm.SimpleImputer: SimpleImputer,
    ImputationAlgorithm.KNNImputer: KNNImputer,
    ImputationAlgorithm.IterativeImputer: IterativeImputer
}


class PreprocessingAlgorithm(str, Enum):
    Binarizer = "Binarizer"
    KBinsDiscretizer = "KBinsDiscretizer"
    MaxAbsScaler = "MaxAbsScaler"
    MinMaxScaler = "MinMaxScaler"
    OneHotEncoder = "OneHotEncoder"
    OrdinalEncoder = "OrdinalEncoder"
    QuantileTransformer = "QuantileTransformer"
    RobustScaler = "RobustScaler"
    StandardScaler = "StandardScaler"
    Normalizer = "Normalizer"
    TargetEncoder = "TargetEncoder"
    MEstimateEncoder = "MEstimateEncoder"
    LeaveOneOutEncoder = "LeaveOneOutEncoder"
    GLMMEncoder = "GLMMEncoder"
    HelmertEncoder = "HelmertEncoder"
    JamesSteinEncoder = "JamesSteinEncoder"
    CatBoostEncoder = "CatBoostEncoder"


preprocessing_algorithms = {
    PreprocessingAlgorithm.Binarizer: Binarizer,
    PreprocessingAlgorithm.KBinsDiscretizer: KBinsDiscretizer,
    PreprocessingAlgorithm.MaxAbsScaler: MaxAbsScaler,
    PreprocessingAlgorithm.MinMaxScaler: MinMaxScaler,
    PreprocessingAlgorithm.OneHotEncoder: OneHotEncoder,
    PreprocessingAlgorithm.OrdinalEncoder: OrdinalEncoder,
    PreprocessingAlgorithm.QuantileTransformer: QuantileTransformer,
    PreprocessingAlgorithm.RobustScaler: RobustScaler,
    PreprocessingAlgorithm.StandardScaler: StandardScaler,
    PreprocessingAlgorithm.Normalizer: Normalizer,
    PreprocessingAlgorithm.TargetEncoder: ce.TargetEncoder,
    PreprocessingAlgorithm.MEstimateEncoder: ce.MEstimateEncoder,
    PreprocessingAlgorithm.LeaveOneOutEncoder: ce.LeaveOneOutEncoder,
    PreprocessingAlgorithm.GLMMEncoder: ce.GLMMEncoder,
    PreprocessingAlgorithm.HelmertEncoder: ce.HelmertEncoder,
    PreprocessingAlgorithm.JamesSteinEncoder: ce.JamesSteinEncoder,
    PreprocessingAlgorithm.CatBoostEncoder: ce.CatBoostEncoder
}


class DimensionalityReductionAlgorithm(str, Enum):
    PCA = "PCA"
    FeatureAgglomeration = "FeatureAgglomeration"
    GaussianRandomProjection = "GaussianRandomProjection"


dimensionality_reduction_algorithms = {
    DimensionalityReductionAlgorithm.PCA: PCA,
    DimensionalityReductionAlgorithm.FeatureAgglomeration: FeatureAgglomeration,
    DimensionalityReductionAlgorithm.GaussianRandomProjection: GaussianRandomProjection
}


class AggregateTransformerAlgorithm(str, Enum):
    SumColumns = "SumColumns"


aggregate_transformer_algorithms = {
    AggregateTransformerAlgorithm.SumColumns: aggregate_transformers.SumColumnsTransformer
}


class UtilTransformerAlgorithm(str, Enum):
    ZeroTransformer = "ZeroTransformer"


util_transformer_algorithms = {
    UtilTransformerAlgorithm.ZeroTransformer: util_transformers.ZeroTransformer
}


class InfrequentCategoryMergerTransformerAlgorithm(str, Enum):
    InfrequentCategoryMerger = "InfrequentCategoryMerger"
    
infrequent_category_merger_algorithms = {
    InfrequentCategoryMergerTransformerAlgorithm.InfrequentCategoryMerger: i_c_m_t.InfrequentCategoryMergerTransformer
}

class MiscTransformerAlgorithm(str, Enum):
    AddMunicipalityCoordinates = "AddMunicipalityCoordinates"

misc_transformer_algorithms = {
    MiscTransformerAlgorithm.AddMunicipalityCoordinates: misc_transformers.AddMunicipalityCoordinatesTransformer
}

class TransformerType(str, Enum):
    Imputation = "Imputation"
    Preprocessing = "Preprocessing"
    Aggregate = "Aggregate"
    DimensionalityReduction = "DimensionalityReduction"
    Util = "Util"
    InfrequentCatMerger = "InfrequentCatMerger"
    Misc = "Misc"


transformer_algorithms = {
    TransformerType.Imputation: (ImputationAlgorithm, imputation_algorithms),
    TransformerType.Preprocessing: (PreprocessingAlgorithm, preprocessing_algorithms),
    TransformerType.Aggregate: (AggregateTransformerAlgorithm, aggregate_transformer_algorithms),
    TransformerType.DimensionalityReduction: (
        DimensionalityReductionAlgorithm, dimensionality_reduction_algorithms),
    TransformerType.Util: (UtilTransformerAlgorithm,
                           util_transformer_algorithms),
    TransformerType.InfrequentCatMerger: (InfrequentCategoryMergerTransformerAlgorithm, infrequent_category_merger_algorithms),
    TransformerType.Misc: (MiscTransformerAlgorithm, misc_transformer_algorithms)
}


def get_transformer_algoritm(transformer_name: str, algorithm_name: str, algorithm_parameters):
    try:
        algorithm_types, algorithm_dict = transformer_algorithms[TransformerType[transformer_name]]
    except:
        print(
            f"Error: Transformer type must be one of the following: {[e.value for e in TransformerType]}, but {transformer_name} was given!")
        exit()

    try:
        algorithm = algorithm_dict[algorithm_types[algorithm_name]]
    except:
        print(
            f"Error: Algorithm type for {transformer_name} must be one of the following: {[e.value for e in algorithm_types]}, but {algorithm_name} was given!")
        exit()
    instance = algorithm(**algorithm_parameters)
    return instance
