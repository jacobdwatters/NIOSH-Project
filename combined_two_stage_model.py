from pprint import pprint
import pandas as pd
import numpy as np

import violation_common
import lightgbm as lgb
from skranger.ensemble import RangerForestRegressor
from sklearn.preprocessing import StandardScaler

from SIG_SUB_Classifier import lgb_objective as classifier_objective
from SIG_SUB_Regression_Models import lgb_regression_objective as regression_y_objective
from SIG_SUB_Regression_Models import rf_regression_objective as regression_n_objective

import violation_common

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

import pickle



categorical_cols = ['PRIMARY_OR_MILL', 'COAL_METAL_IND', 'MINE_TYPE', 'VIOLATOR_TYPE_CD']
numerical_cols = ['VIOLATOR_INSPECTION_DAY_CNT', 'VIOLATOR_VIOLATION_CNT', 'YEAR_OCCUR']

target_classifier = 'SIG_SUB'
target_regression = 'PROPOSED_PENALTY'

feature_ranking = [
    'YEAR_OCCUR',
    'VIOLATOR_VIOLATION_CNT',
    'VIOLATOR_INSPECTION_DAY_CNT',
    'MINE_TYPE',
    'PRIMARY_OR_MILL',
    'COAL_METAL_IND',
    'VIOLATOR_TYPE_CD'
]


def predict_two_stage(classifier_trial, regression_y_trial, regression_n_trial, data_train, data_validate, metrics, perfect_classifier=False):

    data_train, scaler = violation_common.scale_selected_columns(data_train, cols_to_scale=numerical_cols + [target_regression])
    data_validate, _ = violation_common.scale_selected_columns(data_validate, cols_to_scale=numerical_cols + [target_regression], preprocessor=scaler)

    classifier_preds = classifier_objective(classifier_trial, data_train, data_validate, metrics=None, objective_metric=None, just_predict=True)
    if perfect_classifier:
        classifier_preds = data_validate['SIG_SUB'].values == 'Y'
    regression_y_preds = regression_y_objective(regression_y_trial, data_train[data_train['SIG_SUB'] == 'Y'], data_validate, metrics=None, objective_metric=None, just_predict=True, target='PROPOSED_PENALTY', categorical_cols=categorical_cols, numerical_cols=numerical_cols, feature_ranking=feature_ranking)
    regression_n_preds = regression_n_objective(regression_n_trial, data_train[data_train['SIG_SUB'] == 'N'], data_validate, metrics=None, objective_metric=None, just_predict=True, target='PROPOSED_PENALTY', categorical_cols=categorical_cols, numerical_cols=numerical_cols, feature_ranking=feature_ranking)
    
    combined_preds = np.zeros_like(classifier_preds) * np.nan

    regression_y_indices = classifier_preds.astype(bool)
    regression_n_indices = (1 - classifier_preds).astype(bool)

    combined_preds[regression_y_indices] = regression_y_preds[regression_y_indices]
    combined_preds[regression_n_indices] = regression_n_preds[regression_n_indices]
    
    true_y = data_validate['PROPOSED_PENALTY'].values

    metric_results = dict()

    for metric_name, metric_func in metrics:
        metric_value = metric_func(true_y, combined_preds)
        metric_results[metric_name] = metric_value
        
    return metric_results


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


if __name__ == '__main__':
    best_classifier_trial = None
    with open('data/hp_validation_results_classifiers.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        best_classifier_trial = hp_validation_results['non_smote']['lightgbm'].best_trial
    
    best_regression_y_trial = None
    best_regression_n_trial = None
    with open('data/hp_validation_results_regressors.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        best_regression_y_trial = hp_validation_results['lightgbm']['Y'].best_trial
        best_regression_n_trial = hp_validation_results['random_forest']['N'].best_trial

    train_full = pd.read_csv('data/after_2010_train_full.csv', index_col=0)
    test = pd.read_csv('data/after_2010_test.csv', index_col=0)
    train_hp = pd.read_csv('data/after_2010_train_hp.csv', index_col=0)
    validate_hp = pd.read_csv('data/after_2010_validate_hp.csv', index_col=0)
    train_smote_full = pd.read_csv('data/after_2010_train_smote_full.csv', index_col=0)
    train_hp_smote = pd.read_csv('data/after_2010_train_smote_hp.csv', index_col=0)

    # list of tuples of metric names and functions
    # [('metric_name', metric_function), ...]
    metrics = [('mse', mean_squared_error),
               ('mae', mean_absolute_error),
               ('r2', r2_score),
               ('rmse', rmse),
               ('explained_variance_score', explained_variance_score)]

    metrics = predict_two_stage(best_classifier_trial, best_regression_y_trial, best_regression_n_trial, train_full, test, metrics=metrics, perfect_classifier=False)
    pprint(metrics)