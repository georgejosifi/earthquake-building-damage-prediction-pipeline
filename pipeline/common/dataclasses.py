from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class ImbalanceAlgorithmConfig:
    sampler_name: str
    sampler_algorithm: str
    algorithm_parameters: dict[str, Any] = field(default_factory=dict)
    

@dataclass
class TransformerConfig:
    name: str
    features: list[str]
    transformer_name: str
    transformer_algorithm: str
    new_column_names: list[str] = field(default_factory=list)
    algorithm_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnTransformerStepConfig:
    step_name: str
    col_transform_parameters: dict[str, Any]
    transformers: list[TransformerConfig]
    
@dataclass
class ClassificationStepConfig:
    classification_algorithm: str
    algorithm_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    validation: bool
    imbalanced_learn: bool
    data_load_dir: str
    results_dir: str
    train_values_file: str
    train_labels_file: str
    test_values_file: str
    output_file: str
    steps_dir: str
    imbalanced_learn_dir: str


@dataclass
class LoadedData:
    train_values: pd.DataFrame
    train_labels: pd.DataFrame
    test_values: pd.DataFrame
