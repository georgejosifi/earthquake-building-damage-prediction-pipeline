from pathlib import Path

import dacite
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

from common.dataclasses import PipelineConfig, LoadedData
from common.util import create_pipeline, load_data
from common.util import get_imbalanced_learning_algorithm

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

import numpy as np
np.random.seed(31)

def validate(pipeline, data: LoadedData, pipeline_config: PipelineConfig):              #shto nje bool imbalanced learn
    # train_values, validation_values, train_labels, validation_labels = train_test_split(
    #     data.train_values, data.train_labels)
    # pipeline.fit(X=train_values, y=train_labels["damage_grade"])
    # predictions = pipeline.predict(validation_values)
    # validation_labels = validation_labels["damage_grade"]
    # acc = accuracy_score(validation_labels, predictions)
    # mcc = matthews_corrcoef(validation_labels, predictions)
    # print(f"Accuracy score: {acc}")
    # print(f"Matthews score: {mcc}")
    
    
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    acc_scores = []
    mcc_scores = []
    
    i = 0
    for train_index, test_index in skf.split(data.train_values, data.train_labels["damage_grade"]):
        print(f"Starting fold {i}")
        i += 1
        
        X_train, X_test = data.train_values.loc[train_index], data.train_values.loc[test_index]
        y_train, y_test = data.train_labels.loc[train_index]["damage_grade"], data.train_labels.loc[test_index]["damage_grade"]
        
        if pipeline_config.imbalanced_learn:
            sampling_algorithm = get_imbalanced_learning_algorithm(pipeline_config)
            print(sampling_algorithm)
            classification_step = pipeline.steps.pop(-1)
            X_train = pipeline.fit_transform(X_train)
            X_train, y_train = sampling_algorithm.fit_resample(X_train, y_train)
            classification_step[1].fit(X_train, y_train)
            pipeline.steps.append(classification_step)
            
        else:
            pipeline.fit(X_train, y_train)
        
        # for _, transform in pipeline.steps[:-1]:
        #     transform.fit(X_train)
        #     X_train = transform.transform(X_train)

        # # print(X_train.columns)
        # # smote = SVMSMOTE(random_state=31, n_jobs=12, )
        # # X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # # print(pipeline.steps[-1])
                    
        # pipeline.steps[-1][1].fit(X_train, y_train)
        
        
        
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        acc_scores.append(acc)
        mcc_scores.append(mcc)
        
    print(f"Accuracy: {np.mean(acc_scores)}, std: {np.std(acc_scores)} ")
    print(f"MCC: {np.mean(mcc_scores)}, std: {np.std(mcc_scores)} ")

def predict(pipeline, pipeline_config: PipelineConfig, data: LoadedData):
    pipeline.fit(X=data.train_values, y=data.train_labels["damage_grade"])
    predictions = pipeline.predict(data.test_values)
    res = pd.concat([data.test_values["building_id"], pd.Series(
        predictions, name="damage_grade")], axis=1)
    output_file = Path(__file__).parent / pipeline_config.results_dir/ pipeline_config.output_file
               
    pd.DataFrame.to_csv(res, output_file, index=False)


def main():
    config_dir = Path(__file__).parent / "config" / "config.yaml"
    with config_dir.open("r") as f:
        pipeline_config: PipelineConfig = dacite.from_dict(
            data_class=PipelineConfig, data=yaml.load(f, yaml.FullLoader))
    data = load_data(pipeline_config)
    pipeline = create_pipeline(pipeline_config)

    if pipeline_config.validation:
        validate(pipeline, data, pipeline_config)
    else:
        predict(pipeline, pipeline_config, data)


if __name__ == "__main__":
    main()
