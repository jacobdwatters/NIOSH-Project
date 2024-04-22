import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# import lightgbm as lgb
import lightgbm as lgb

# Load data
print('Loading data...')
processed_data = pd.read_csv('data/processed_contest_data.csv')
# after 2010
processed_data = processed_data[processed_data['YEAR_OCCUR'] >= 2010]
print('Data loaded.')

# Assuming retrieve_raw_violation_data_contest and process_raw_contest_data are defined
# raw_data = retreive_raw_violation_data_contest()
# processed_data = process_raw_contest_data(raw_data)

def process_data(processed_data):
    print('Processing data...')
    # Encode categorical variables
    categorical_columns = ['MINE_TYPE', 'COAL_METAL_IND', 'SIG_SUB', 'SECTION_OF_ACT', 'LIKELIHOOD', 'INJ_ILLNESS', 'NEGLIGENCE', 'ENFORCEMENT_AREA', 'SPECIAL_ASSESS', 'PRIMARY_OR_MILL']


    ohe_encoder = OneHotEncoder(sparse_output=False)
    encoded_features = ohe_encoder.fit_transform(processed_data[categorical_columns])

    # Normalize numerical variables
    numerical_columns = ['NO_AFFECTED', 'PROPOSED_PENALTY', 'VIOLATOR_VIOLATION_CNT', 'VIOLATOR_INSPECTION_DAY_CNT', 'YEAR_OCCUR']

    scaler = StandardScaler()

    numerical_features = scaler.fit_transform(processed_data[numerical_columns])

    # Split data into features and targets
    X = np.concatenate((encoded_features, numerical_features), axis=1)
    # replace 'Y' with 1 and 'N' with 0
    y = processed_data['CONTESTED_IND'].replace({'Y': 1, 'N': 0}).values

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, X_test, y_train, y_test)

# Process data
X_train, X_test, y_train, y_test = process_data(processed_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=2**10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2**10, shuffle=False)

print('Data processed.')


# Define Neural Network
class ContestClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ContestClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# # Initialize model, optimizer, and loss function
# input_dim = X_train_tensor.shape[1]
# model = ContestClassifier(input_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()

# # Training Loop
# n_epochs = 10
# train_losses = []
# test_losses = []

# print("Training model...")

# for epoch in range(n_epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output.squeeze(), target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     # Initialize variables for metrics
#     true_labels_train = []
#     predicted_labels_train = []

#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in train_loader:
#             output = model(data).squeeze()
            
#             pred = (output > 0.5).float()
#             correct += pred.eq(target.view_as(pred)).sum().item()

#             true_labels_train.extend(target.cpu().numpy())
#             predicted_labels_train.extend(pred.cpu().numpy())

#     train_accuracy = 100. * correct / len(train_loader.dataset)

#     # Calculate additional metrics for training data
#     balanced_acc_train = balanced_accuracy_score(true_labels_train, predicted_labels_train)
#     precision_train = precision_score(true_labels_train, predicted_labels_train)
#     recall_train = recall_score(true_labels_train, predicted_labels_train)
#     f1_train = f1_score(true_labels_train, predicted_labels_train)

#     print(f"Train Metrics: Balanced Accuracy: {balanced_acc_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1: {f1_train:.4f}")

#     # Initialize variables for metrics
#     true_labels = []
#     predicted_labels = []

#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data).squeeze()
#             test_loss += criterion(output, target).item()
            
#             pred = (output > 0.5).float()
#             correct += pred.eq(target.view_as(pred)).sum().item()
            
#             true_labels.extend(target.cpu().numpy())
#             predicted_labels.extend(pred.cpu().numpy())

#     test_loss /= len(test_loader)
#     test_accuracy = 100. * correct / len(test_loader.dataset)

#     # Calculate additional metrics for test data
#     balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels)
#     recall = recall_score(true_labels, predicted_labels)
#     f1 = f1_score(true_labels, predicted_labels)

#     print(f"Test Metrics: Balanced Accuracy: {balanced_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# # Calculate train accuracy
# model.eval()
# correct = 0
# with torch.no_grad():
#     for data, target in train_loader:
#         output = model(data).squeeze()
#         pred = (output > 0.5).float()
#         correct += pred.eq(target.view_as(pred)).sum().item()

# train_accuracy = 100. * correct / len(train_loader.dataset)
# print(f"Final Train Accuracy: {train_accuracy:.2f}%")

# Record results here temporarily
# 94.32% on test data
# 93.12% N

# Train Metrics: Balanced Accuracy: 0.6429, Precision: 0.7428, Recall: 0.2933, F1: 0.4206
# Test Metrics: Balanced Accuracy: 0.6391, Precision: 0.7345, Recall: 0.2858, F1: 0.4115


# train lightgbm model

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Calculate train accuracy
preds = model.predict(X_train)
train_accuracy = 100. * np.mean(preds == y_train)

print(f"Final Train Accuracy: {train_accuracy:.2f}%")

# Calculate test accuracy
preds = model.predict(X_test)
test_accuracy = 100. * np.mean(preds == y_test)

print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Calculate additional metrics for test data
balanced_acc = balanced_accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Test Metrics: Balanced Accuracy: {balanced_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# now test on different mine types
for mine_type in processed_data['MINE_TYPE'].unique():
    print()
    print(f"Mine Type: {mine_type}")
    # Process data
    X_train, X_test, y_train, y_test = process_data(processed_data[processed_data['MINE_TYPE'] == mine_type])

    # Train model
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    # Calculate train accuracy
    preds = model.predict(X_train)
    train_accuracy = 100. * np.mean(preds == y_train)

    print(f"Final Train Accuracy: {train_accuracy:.2f}%")

    # Calculate test accuracy
    preds = model.predict(X_test)
    test_accuracy = 100. * np.mean(preds == y_test)

    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Calculate additional metrics for test data
    balanced_acc = balanced_accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Test Metrics: Balanced Accuracy: {balanced_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")