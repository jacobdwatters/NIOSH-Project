import numpy as np
import pandas as pd

from skranger.ensemble import RangerForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import Ridge
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader

import violation_common

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import pickle
from pprint import pprint

import optuna




def lgb_regression_objective(trial, data_train, data_validate, metrics, objective_metric, categorical_cols, numerical_cols, target, feature_ranking, just_predict=False):
    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    param = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # same pre-processing and encoding steps as before
    # X_train, y_train, preprocessor, target_transformer = violation_common.encode_and_scale(data_train, target=target, contionous_target=True, to_keep=feature_ranking[0:num_features], categorical_cols=categorical_cols, numerical_cols=numerical_cols, preprocessor=None, target_transformer=None)
    # X_validate, y_validate, _, _ = violation_common.encode_and_scale(data_validate, target=target, to_keep=feature_ranking[0:num_features], preprocessor=preprocessor, target_transformer=target_transformer)

    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))
    numerical_cols_to_keep.append(target)

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    model = LGBMRegressor(**param)

    model.fit(X_train, y_train)
    preds = model.predict(X_validate)

    if just_predict:
        return preds, preprocessor

    # calculate metrics
    for metric_name, metric_func in metrics:
        metric_value = metric_func(y_validate, preds)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]

# Repeat the process with other regressors. Here is an example with RandomForest
def rf_regression_objective(trial, data_train, data_validate, metrics, objective_metric, categorical_cols, numerical_cols, target, feature_ranking, just_predict=False):
    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 64, 128),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_node_size": trial.suggest_int("min_node_size", 1, 5),
        "mtry": trial.suggest_int("mtry", 0, num_features),  # num of features
        "sample_fraction": trial.suggest_float("sample_fraction", 0.5, 1.0),
    }

    # same pre-processing and encoding steps as before
    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))
    numerical_cols_to_keep.append(target)

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    model = RangerForestRegressor(**param)

    model.fit(X_train, y_train)
    preds = model.predict(X_validate)

    if just_predict:
        return preds

    # calculate metrics
    for metric_name, metric_func in metrics:
        metric_value = metric_func(y_validate, preds)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]



def ridge_objective(trial, data_train, data_validate, metrics, objective_metric, categorical_cols, numerical_cols, target, feature_ranking):
    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    param = {
        "alpha": trial.suggest_float("alpha", 0.1, 10),  # regularization strength
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "solver": trial.suggest_categorical("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    }

    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))
    numerical_cols_to_keep.append(target)

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    model = Ridge(**param)

    model.fit(X_train, y_train)
    preds = model.predict(X_validate)

    # calculate metrics
    for metric_name, metric_func in metrics:
        metric_value = metric_func(y_validate, preds)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]


class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


def nn_regression_objective(trial, data_train, data_validate, metrics, objective_metric, categorical_cols, numerical_cols, target, feature_ranking):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))

    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))
    numerical_cols_to_keep.append(target)

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    input_size = X_train.shape[1]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, trial.suggest_int(f'n_units_l{i}', 10, 500)))
        else:
            layers.append(nn.Linear(trial.suggest_int(f'n_units_l{i-1}', 10, 500), 
                                    trial.suggest_int(f'n_units_l{i}', 10, 500)))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(trial.suggest_int(f'n_units_l{n_layers-1}', 10, 500), 1))

    model = nn.Sequential(*layers).to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model_train_x, early_stopping_x, model_train_y, early_stopping_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create data loaders
    model_train_dataset = RegressionDataset(model_train_x, model_train_y)
    val_dataset = RegressionDataset(X_validate, y_validate)
    early_stopping_dataset = RegressionDataset(early_stopping_x, early_stopping_y)
    train_loader = DataLoader(model_train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    early_stopping_loader = DataLoader(early_stopping_dataset, batch_size=32, shuffle=False)

    early_stopping_best = np.inf
    early_stopping_strikes = 0

    # Training loop
    for epoch in range(100):
        model.train()
        for batch in tqdm(train_loader):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).ravel()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # early stopping
        model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            for batch in early_stopping_loader:
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).ravel()
                preds.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
            
        early_stop_mse = mean_squared_error(targets, preds)
        print(f'Early stopping mse: {early_stop_mse}')
        
        if early_stop_mse < early_stopping_best - 0.005:
            early_stopping_best = early_stop_mse
            early_stopping_strikes = 0
        else:
            if early_stop_mse < early_stopping_best:
                early_stopping_best = early_stop_mse
            early_stopping_strikes += 1
            if early_stopping_strikes == 2:
                break

    # Validation
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
    # calculate metrics
    for metric_name, metric_func in metrics:
        metric_value = metric_func(y_validate, preds)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def hp_tune(train_hp, validate_hp):

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

    # model types is a list of tuples of model names and objective functions and number of trials
    model_types = [('lightgbm', lgb_regression_objective, 20),
                   ('random_forest', rf_regression_objective, 10),
                   ('ridge_regression', ridge_objective, 10),
                   ('neural_network', nn_regression_objective, 5)]

    # list of tuples of metric names and functions
    # [('metric_name', metric_function), ...]
    metrics = [('mse', mean_squared_error),
               ('mae', mean_absolute_error),
               ('r2', r2_score),
               ('rmse', rmse),
               ('explained_variance_score', explained_variance_score)]

    # results[dataset_type][model_name][sig_sub] = study
    hp_validation_results = {model_name[0]: {'Y': None, 'N': None} for model_name in model_types}
                
    for model_name, objective, n_trials in model_types:
        for sig_sub in ['Y', 'N']:
            print(f'Model: {model_name}, SIG_SUB: {sig_sub}')
            study = optuna.create_study(direction="minimize")

            trial_train_data = train_hp
            trial_validate_data = validate_hp
            trial_train_data = trial_train_data[trial_train_data['SIG_SUB'] == sig_sub]
            trial_validate_data = trial_validate_data[trial_validate_data['SIG_SUB'] == sig_sub]

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
                
            hp_validation_results[model_name][sig_sub] = study


    for model_name, objective, _ in model_types:
        for sig_sub in ['Y', 'N']:
            print(f'Model: {model_name}, SIG_SUB: {sig_sub}')
            print("Best hyperparameters:")
            pprint(hp_validation_results[model_name][sig_sub].best_params)
            print("Metrics:")
            pprint(hp_validation_results[model_name][sig_sub].best_trial.user_attrs)
            print()

    # Save results with pickle
    with open('data/hp_validation_results_regressors.pkl', 'wb') as f:
        pickle.dump(hp_validation_results, f)

    with open('data/hp_validation_results_regressors.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        print(hp_validation_results)

if __name__ == '__main__':
    # train_full = pd.read_csv('data/after_2010_train_full.csv', index_col=0)
    # test = pd.read_csv('data/after_2010_test.csv', index_col=0)
    train_hp = pd.read_csv('data/after_2010_train_hp.csv', index_col=0)
    validate_hp = pd.read_csv('data/after_2010_validate_hp.csv', index_col=0)
    # train_smote_full = pd.read_csv('data/after_2010_train_smote_full.csv', index_col=0)
    # train_hp_smote = pd.read_csv('data/after_2010_train_smote_hp.csv', index_col=0)

    model_types = [('lightgbm', lgb_regression_objective, 20),
                   ('random_forest', rf_regression_objective, 10),
                   ('ridge_regression', ridge_objective, 10),
                   ('neural_network', nn_regression_objective, 5)]

    with open('data/hp_validation_results_regressors.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        # print(hp_validation_results)


    for model_name, objective, _ in model_types:
        for sig_sub in ['Y', 'N']:
            print(f'Model: {model_name}, SIG_SUB: {sig_sub}')
            # print("Best hyperparameters:")
            # pprint(hp_validation_results[model_name][sig_sub].best_params)
            print("Metrics:")
            pprint(hp_validation_results[model_name][sig_sub].best_trial.user_attrs)
            print()

    # hp_tune(train_hp, validate_hp)