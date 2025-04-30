import pandas as pan
import numpy as num
import seaborn as seab
import matplotlib.pyplot as mplt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === loading the BERT features dataset ===
print("Loadinging the BERT features CSV...")
bertdatafile = pan.read_csv("bert_features.csv")
X = bertdatafile.drop(columns=['label']).values
y = bertdatafile['label'].values


le = LabelEncoder()
y = le.fit_transform(y)

# === Splitting the dataset into training and testing sets ===
print("Splitting the dataset into training and testing sets...")
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === Converting the data to PyTorch tensors ===
'''Converting the training data (X_train and y_train) to tensors (the training data is used for training the model)
        Note that the labels (y_train) are converted to long type for classification tasks.
        & the features (X_train) are converted to float type for numerical data.

    The test data is NOT used to train the model ([tensors] will be used for future evaluation)'''

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# Convert the test data to tensors
# Note: The test data is not used for training, but it's converted to tensors for evaluation later.
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Creating dataLoader for training and testing sets
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# === Defining the Popularity model ===
''' Popularity class model is simple feedforward neural network w/one hidden layer; the input dimension is 
the number of features in the dataset, hidden dimension is set to 128, & the output dimension is 
3 (for the three classes of popularity)'''

class PopularityClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
