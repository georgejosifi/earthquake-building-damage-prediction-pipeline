from enum import Enum
from pathlib import Path

import dacite
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline

from common.dataclasses import (ColumnTransformerStepConfig, LoadedData,
                                PipelineConfig, ClassificationStepConfig, ImbalanceAlgorithmConfig)
from common.transformers import get_transformer_algoritm
from common.imbalanced_data_samplers import get_sampler_algoritm
from common.classifiers import get_classification_algorithm
from custom_transformers.util_transformers import ColumnRenameTransformer

def get_imbalanced_learning_algorithm(pipeline_config: PipelineConfig):
    imb_learn_path = Path(pipeline_config.imbalanced_learn_dir)
    with imb_learn_path.open('r') as f:
        imbalance_algo_config : ImbalanceAlgorithmConfig = dacite.from_dict(data_class= ImbalanceAlgorithmConfig, 
        data = yaml.load(f,yaml.FullLoader))
            
    imbalance_algo = get_sampler_algoritm(
    imbalance_algo_config.sampler_name, imbalance_algo_config.sampler_algorithm, imbalance_algo_config.algorithm_parameters)
    
    return imbalance_algo
    
    
    
def load_data(pipeline_config: PipelineConfig) -> LoadedData:
    base_dir = Path(__file__).parent.parent
    data_load_dir = base_dir / pipeline_config.data_load_dir
    train_values = pd.read_csv(
        data_load_dir / pipeline_config.train_values_file)
    train_labels = pd.read_csv(
        data_load_dir / pipeline_config.train_labels_file)
    test_values = pd.read_csv(data_load_dir / pipeline_config.test_values_file)

    loaded_data = LoadedData(
        train_values=train_values, train_labels=train_labels, test_values=test_values)

    return loaded_data


def create_pipeline(pipeline_config: PipelineConfig):
    base_dir = Path(__file__).parent.parent
    sorted_steps_path = sorted(Path(base_dir / pipeline_config.steps_dir).iterdir())
    classification_step_config: ClassificationStepConfig
    transformer_step_configs: list[ColumnTransformerStepConfig] = []
    for i, step in enumerate(sorted_steps_path):
        with step.open("r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config["step_name"] = step.name.split("_")[1].split(".")[0]
        if config["step_name"] == "classification":
            classification_step_config = dacite.from_dict(
                data_class=ClassificationStepConfig, data=config, config=dacite.Config(cast=[Enum]))
            print(f"Loaded classifier")
        else:
            config = dacite.from_dict(
                data_class=ColumnTransformerStepConfig, data=config, config=dacite.Config(cast=[Enum]))
            transformer_step_configs.append(config)
            print(f"Loaded step {i + 1}: {config.step_name}")

    column_transformers = []
    for step_cfg in transformer_step_configs:
        transformers = []
        for trans_cfg in step_cfg.transformers:
            transformer = get_transformer_algoritm(
                trans_cfg.transformer_name, trans_cfg.transformer_algorithm, trans_cfg.algorithm_parameters)
            if trans_cfg.new_column_names:
                transformer = make_pipeline(transformer, ColumnRenameTransformer(
                    trans_cfg.new_column_names), "passthrough")
            transformed_regex = "|".join(f"{w}" for w in trans_cfg.features)
            transformers.append(
                (trans_cfg.name, transformer, make_column_selector(transformed_regex)))

        col_trans = ColumnTransformer(
            transformers=transformers, verbose_feature_names_out=False, **step_cfg.col_transform_parameters)
        col_trans.set_output(transform="pandas")
        column_transformers.append((step_cfg.step_name, col_trans))

    classifier = get_classification_algorithm(
        classification_step_config.classification_algorithm, classification_step_config.algorithm_parameters)

    pipeline = Pipeline(
        [*column_transformers, ("classification", classifier)], verbose=True)
    return pipeline
