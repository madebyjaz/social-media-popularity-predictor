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
