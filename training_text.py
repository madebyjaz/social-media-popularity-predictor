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
    

# Check if GPU is available and use it if possible if not use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PopularityClassifier(input_dim=X_train.shape[1]).to(device)

# === Defining the loss function and optimizer ===

''' For multi-class classification use CrossEntropyLoss. It:
1. Combines softmax and negative log-likelihood loss in one single class function.

Also use Adam optimizer for faster convergence. It:
1. Is an adaptive learning rate optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
3. Is computationally efficient and requires little memory. '''

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Training the model ===
'''The model is trained using the Adam optimizer and CrossEntropyLoss for multi-class classification.
    The model is trained for 100 epochs'''

print("Training the bert model...")
epochs = 100 #trainig starting out maybe 400 next iteration
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dl)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device)
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

# Saving the test data and predictions
print("\n ✅Done! Now saving the test data & predictions to numpy files...")
num.save("y_test.npy", y_test)
num.save("all_preds.npy", num.array(all_preds))


# === Saving the trained model ===
torch.save(model.state_dict(), "popularity_classifier.pt")
print("\n✅ Andd we're Done with the training! The model was saved to popularity_classifier.pt")


# === Saving the training losses ===
'''The training losses are saved to a numpy file for future reference. 
    This can be useful for plotting the training loss curve or for resuming training later.'''
print("\n ✅Done! Now saving the training losses to train_losses.npy...")
num.save("training_losses.npy", num.array(train_losses))


