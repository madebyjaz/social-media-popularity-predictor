import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as num
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# === Loading the BERT features dataset ===
df = pd.read_csv("bert_features.csv")
X = df.drop(columns=['label']).values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# === Re-defining the trained popularity model ===
'''The model is re-defined to ensure that the same architecture is used for evaluation. '''
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
    
# === Load trained model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PopularityClassifier(input_dim=X_test.shape[1]).to(device)
model.load_state_dict(torch.load("popularity_classifier.pt"))
model.eval()


# === Making predictions ===
'''The model is used to make predictions on the test set. 
    The predictions are converted to class labels using the label encoder.'''
all_preds = []
with torch.no_grad():
    for i in range(0, len(X_test_tensor), 32):
        batch = X_test_tensor[i:i+32].to(device)
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
# Convert predictions to class labels
all_preds = le.inverse_transform(all_preds)
# === Saving predictions ===        
'''The predictions are saved to a CSV file for further analysis.'''
predictions_df = pd.DataFrame(all_preds, columns=['predictions'])
predictions_df.to_csv("predictions.csv", index=False)


# === Load saved predictions and test labels ===
print("🔍 Loading saved test labels and predictions...")
y_test = num.load("y_test.npy")
all_preds = num.load("all_preds.npy")

# === Load training losses ===
print("📈 Loading training loss values...")
train_losses = num.load("training_losses.npy")
