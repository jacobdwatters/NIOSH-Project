import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from SIG_SUB_Regression_Models import lgb_regression_objective, rf_regression_objective, ridge_objective, nn_regression_objective, rmse

from sklearn.model_selection import train_test_split
import violation_common

import pickle
from pprint import pprint

import optuna


categorical_cols = ['PRIMARY_OR_MILL', 'COAL_METAL_IND', 'MINE_TYPE', 'VIOLATOR_TYPE_CD']
numerical_cols = ['VIOLATOR_INSPECTION_DAY_CNT', 'VIOLATOR_VIOLATION_CNT', 'YEAR_OCCUR']

target = 'PROPOSED_PENALTY'

feature_ranking = [
    'YEAR_OCCUR',
    'VIOLATOR_VIOLATION_CNT',
    'VIOLATOR_INSPECTION_DAY_CNT',
    'MINE_TYPE',
    'PRIMARY_OR_MILL',
    'COAL_METAL_IND',
    'VIOLATOR_TYPE_CD'
]

def hp_tune(train_hp, validate_hp):

    # train_hp, _ = violation_common.scale_selected_columns(train_hp, cols_to_scale=numerical_cols + [target], preprocessor=scaler)
    # validate_hp, _ = violation_common.scale_selected_columns(validate_hp, cols_to_scale=numerical_cols + [target], preprocessor=scaler)


    # model types is a list of tuples of model names and objective functions and number of trials
    model_types = [('lightgbm', lgb_regression_objective, 20),
                   #('random_forest', rf_regression_objective, 10),
                   #('ridge_regression', ridge_objective, 10),
                   #('neural_network', nn_regression_objective, 5)
                   ]

    # list of tuples of metric names and functions
    # [('metric_name', metric_function), ...]
    metrics = [('mse', mean_squared_error),
               ('mae', mean_absolute_error),
               ('r2', r2_score),
               ('rmse', rmse),
               ('explained_variance_score', explained_variance_score)]

    # results[dataset_type][model_name][sig_sub] = study
    hp_validation_results = {model_name[0]: None for model_name in model_types}
                
    for model_name, objective, n_trials in model_types:
        print(f'Model: {model_name}')
        study = optuna.create_study(direction="minimize")

        trial_train_data = train_hp
        trial_validate_data = validate_hp

        study.optimize(lambda trial: objective(trial,
                                                data_train=trial_train_data,
                                                data_validate=trial_validate_data,
                                                metrics=metrics,
                                                objective_metric='mse',
                                                categorical_cols=categorical_cols,
                                                numerical_cols=numerical_cols,
                                                target=target,
                                                feature_ranking=feature_ranking),
                        n_trials=n_trials)

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print("  User attrs:")
        pprint(trial.user_attrs)
            
        hp_validation_results[model_name] = study


    for model_name, objective, _ in model_types:
        print(f'Model: {model_name}')
        print("Best hyperparameters:")
        pprint(hp_validation_results[model_name].best_params)
        print("Metrics:")
        pprint(hp_validation_results[model_name].best_trial.user_attrs)
        print()

    # Save results with pickle
    with open('data/hp_validation_results_one_stage.pkl', 'wb') as f:
        pickle.dump(hp_validation_results, f)

    with open('data/hp_validation_results_one_stage.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        print(hp_validation_results)

def evaluate_model(objective, trial, data_train, data_validate, metrics):
    # data_train, _ = violation_common.scale_selected_columns(data_train, cols_to_scale=numerical_cols + [target], preprocessor=scaler)
    # data_validate, _ = violation_common.scale_selected_columns(data_validate, cols_to_scale=numerical_cols + [target], preprocessor=scaler)

    objective(trial, data_train, data_validate, metrics=metrics, objective_metric='mse', categorical_cols=categorical_cols, numerical_cols=numerical_cols, target=target, feature_ranking=feature_ranking, just_predict=False)
    
    return trial.user_attrs


if __name__ == '__main__':
    train_full = pd.read_csv('data/after_2010_train_full.csv', index_col=0)
    test = pd.read_csv('data/after_2010_test.csv', index_col=0)
    train_hp = pd.read_csv('data/after_2010_train_hp.csv', index_col=0)
    validate_hp = pd.read_csv('data/after_2010_validate_hp.csv', index_col=0)
    train_smote_full = pd.read_csv('data/after_2010_train_smote_full.csv', index_col=0)
    train_hp_smote = pd.read_csv('data/after_2010_train_smote_hp.csv', index_col=0)
    

    # hp_tune(train_hp, validate_hp)

    # list of tuples of metric names and functions
    # [('metric_name', metric_function), ...]
    metrics = [('mse', mean_squared_error),
               ('mae', mean_absolute_error),
               ('r2', r2_score),
               ('rmse', rmse),
               ('explained_variance_score', explained_variance_score)]
    
    model_types = [('lightgbm', lgb_regression_objective, 20),
                #    ('random_forest', rf_regression_objective, 10),
                #    ('ridge_regression', ridge_objective, 10),
                #    ('neural_network', nn_regression_objective, 5)
                ]

    best_regression_trial = None
    with open('data/hp_validation_results_one_stage.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
    #     # print(hp_validation_results)
        for model_name, _, _ in model_types:
            print(f'{model_name}:')
            print("Best hyperparameters:")
            pprint(hp_validation_results[model_name].best_params)
            print("Metrics:")
            pprint(hp_validation_results[model_name].best_trial.user_attrs)

        best_regression_trial = hp_validation_results['lightgbm'].best_trial
    
    # print(best_regression_trial.params)

    fresh_trial = optuna.trial.create_trial(params=best_regression_trial.params, value=best_regression_trial.value, distributions=best_regression_trial.distributions)

    eval_results = evaluate_model(lgb_regression_objective, fresh_trial, train_full, test, metrics=metrics)


    pprint(eval_results)