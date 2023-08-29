
import numpy as np
import pandas as pd

import violation_common

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, cohen_kappa_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from skranger.ensemble import RangerForestClassifier
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader

import pickle

from tqdm import tqdm

from pprint import pprint
import optuna


categorical_cols = ['PRIMARY_OR_MILL', 'COAL_METAL_IND', 'MINE_TYPE', 'VIOLATOR_TYPE_CD']
numerical_cols = ['VIOLATOR_INSPECTION_DAY_CNT', 'VIOLATOR_VIOLATION_CNT', 'YEAR_OCCUR']

target = 'SIG_SUB'

feature_ranking = [
    'YEAR_OCCUR',
    'VIOLATOR_VIOLATION_CNT',
    'VIOLATOR_INSPECTION_DAY_CNT',
    'MINE_TYPE',
    'PRIMARY_OR_MILL',
    'COAL_METAL_IND',
    'VIOLATOR_TYPE_CD'
]


def lgb_objective(trial, data_train, data_validate, metrics, objective_metric, just_predict=False):

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -2,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    categorical_cols_to_keep.append(target)
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    model = lgb.LGBMClassifier(**param)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_validate)[:, 1]
    pred_labels = model.predict(X_validate)

    if just_predict:
        return pred_labels

    # calculate metrics
    for metric_name, metric_func, requires_proba in metrics:
        metric_value = None
        if requires_proba:
            metric_value = metric_func(y_validate, preds)
        else:
            metric_value = metric_func(y_validate, pred_labels)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]


def rf_objective(trial, data_train, data_validate, metrics, objective_metric):
    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 64, 128),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_node_size": trial.suggest_int("min_node_size", 1, 5),
        "mtry": trial.suggest_int("mtry", 0, num_features),  # num of features
        "split_rule": trial.suggest_categorical("split_rule", ["gini", "extratrees"]),
        "sample_fraction": trial.suggest_float("sample_fraction", 0.5, 1.0),
    }
    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    categorical_cols_to_keep.append(target)
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    model = RangerForestClassifier(**param)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_validate)[:, 1]
    pred_labels = model.predict(X_validate)

    # calculate metrics
    for metric_name, metric_func, requires_proba in metrics:
        metric_value = None
        if requires_proba:
            metric_value = metric_func(y_validate, preds)
        else:
            metric_value = metric_func(y_validate, pred_labels)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]


def logistic_objective(trial, data_train, data_validate, metrics, objective_metric):
    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    param = {
        "C": trial.suggest_float("C", 1e-10, 1e10, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
    }

    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    categorical_cols_to_keep.append(target)
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))

    X_train, y_train, preprocessor = violation_common.df_to_model_ready(data_train, categorical_cols_to_keep, numerical_cols_to_keep, target)
    X_validate, y_validate, _ = violation_common.df_to_model_ready(data_validate, categorical_cols_to_keep, numerical_cols_to_keep, target, preprocessor=preprocessor)

    model = LogisticRegression(**param)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_validate)[:, 1]
    pred_labels = model.predict(X_validate)

    # calculate metrics
    for metric_name, metric_func, requires_proba in metrics:
        metric_value = None
        if requires_proba:
            metric_value = metric_func(y_validate, preds)
        else:
            metric_value = metric_func(y_validate, pred_labels)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]


class ClassifierDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.unsqueeze(torch.tensor(self.labels[idx], dtype=torch.float), dim=0)


def nn_objective(trial, data_train, data_validate, metrics, objective_metric):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define neural network architecture
    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))

    num_features = trial.suggest_int('num_features', 1, len(categorical_cols) + len(numerical_cols))
    cols_to_keep = feature_ranking[:num_features]
    categorical_cols_to_keep = list(set(cols_to_keep) & set(categorical_cols))
    categorical_cols_to_keep.append(target)
    numerical_cols_to_keep = list(set(cols_to_keep) & set(numerical_cols))

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
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers).to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model_train_x, early_stopping_x, model_train_y, early_stopping_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create data loaders
    model_train_dataset = ClassifierDataset(model_train_x, model_train_y)
    val_dataset = ClassifierDataset(X_validate, y_validate)
    early_stopping_dataset = ClassifierDataset(early_stopping_x, early_stopping_y)
    train_loader = DataLoader(model_train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    early_stopping_loader = DataLoader(early_stopping_dataset, batch_size=32, shuffle=False)

    early_stopping_best = 0
    early_stopping_strikes = 0

    # Training loop
    for epoch in range(100):
        model.train()
        for batch in tqdm(train_loader):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
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
                outputs = model(features)
                preds.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
            
        early_stop_auc = roc_auc_score(targets, preds)
        print(f'Early stopping auc: {early_stop_auc}')
        
        if early_stop_auc > early_stopping_best + 0.01:
            early_stopping_best = early_stop_auc
            early_stopping_strikes = 0
        else:
            if early_stop_auc > early_stopping_best:
                early_stopping_best = early_stop_auc
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

    pred_labels = np.array(preds) > 0.5
        
    # calculate metrics
    for metric_name, metric_func, requires_proba in metrics:
        metric_value = None
        if requires_proba:
            metric_value = metric_func(y_validate, preds)
        else:
            metric_value = metric_func(y_validate, pred_labels)
        trial.set_user_attr(metric_name, metric_value)

    return trial.user_attrs[objective_metric]


def hp_tune(train_hp, validate_hp, train_hp_smote):
    dataset_types = ['non_smote', 'smote']

    # model types is a list of tuples of model names and objective functions and number of trials
    model_types = [('lightgbm', lgb_objective, 10),
                   ('random_forest', rf_objective, 10),
                   ('logistic_regression', logistic_objective, 10),
                   ('neural_network', nn_objective, 5)]

    # list of tuples of metric names and functions along with whether they require a probability prediction
    # [('metric_name', metric_function, requires_probability_prediction), ...]
    metrics = [('roc_auc_score', roc_auc_score, True),
               ('accuracy_score', accuracy_score, False),
               ('balanced_accuracy_score', balanced_accuracy_score, False),
               ('f1_score', f1_score, False),
               ('precision_score', precision_score, False),
               ('recall_score', recall_score, False),
               ('cohen_kappa_score', cohen_kappa_score, False),
               ('confusion_matrix', confusion_matrix, False)]

    # results[dataset_type][model_name] = study
    hp_validation_results = {dataset_type: {model_name[0]: None for model_name in model_types} for dataset_type in dataset_types}

    for dataset_type in dataset_types:
        for model_name, objective, n_trials in model_types:
            print(f'SMOTE: {dataset_type}, Model: {model_name}')
            study = optuna.create_study(direction="maximize")
            trial_train_data = train_hp_smote if dataset_type == 'smote' else train_hp
            study.optimize(lambda trial: objective(trial,
                                                   data_train=trial_train_data,
                                                   data_validate=validate_hp,
                                                   metrics=metrics,
                                                   objective_metric='roc_auc_score'),
                           n_trials=n_trials)

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                
            hp_validation_results[dataset_type][model_name] = study

    for dataset_type in dataset_types:
        for model_name, objective, _ in model_types:
            print(f'SMOTE: {dataset_type}, Model: {model_name}')
            print("Best hyperparameters:")
            pprint(hp_validation_results[dataset_type][model_name].best_params)
            print("Metrics:")
            pprint(hp_validation_results[dataset_type][model_name].best_trial.user_attrs)
            print()

    # Save results with pickle
    with open('data/hp_validation_results_classifiers.pkl', 'wb') as f:
        pickle.dump(hp_validation_results, f)

    with open('data/hp_validation_results_classifiers.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        print(hp_validation_results)


if __name__ == '__main__':
    # Load data
    train_full = pd.read_csv('data/after_2010_train_full.csv', index_col=0)
    test = pd.read_csv('data/after_2010_test.csv', index_col=0)
    train_hp = pd.read_csv('data/after_2010_train_hp.csv', index_col=0)
    validate_hp = pd.read_csv('data/after_2010_validate_hp.csv', index_col=0)
    train_smote_full = pd.read_csv('data/after_2010_train_smote_full.csv', index_col=0)
    train_hp_smote = pd.read_csv('data/after_2010_train_smote_hp.csv', index_col=0)
    dataset_types = ['non_smote', 'smote']

    # model types is a list of tuples of model names and objective functions and number of trials
    model_types = [('lightgbm', lgb_objective, 10),
                   ('random_forest', rf_objective, 10),
                   ('logistic_regression', logistic_objective, 10),
                   ('neural_network', nn_objective, 5)]
    with open('data/hp_validation_results_classifiers.pkl', 'rb') as f:
        hp_validation_results = pickle.load(f)
        for dataset_type in dataset_types:
            for model_name, objective, _ in model_types:
                print(f'SMOTE: {dataset_type}, Model: {model_name}')
                # print("Best hyperparameters:")
                # pprint(hp_validation_results[dataset_type][model_name].best_params)
                # print("Metrics:")
                # pprint(hp_validation_results[dataset_type][model_name].best_trial.user_attrs)
                print('Confusion Matrix:')
                pprint(hp_validation_results[dataset_type][model_name].best_trial.user_attrs['confusion_matrix'])
                print()
    # hp_tune(train_hp, validate_hp, train_hp_smote)