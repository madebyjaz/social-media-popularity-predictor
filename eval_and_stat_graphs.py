import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
import numpy as num
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# === Loading the BERT features dataset ===
bert_df = pd.read_csv("bert_features.csv")
X = bert_df.drop(columns=['label']).values
y = bert_df['label'].values

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
print("\n📈 Loading training loss values...")
train_losses = num.load("training_losses.npy")

# === Evaluate model performance ===
print("\nEvaluating the model performance...")
'''The model is evaluated using accuracy score and classification report. 
    The classification report includes precision, recall, and F1-score for each class.'''
# Calculate accuracy
accuracy = accuracy_score(y_test, all_preds)
print(f"\n🎯 Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
# Print classification report 
print(classification_report(y_test, all_preds, target_names=le.classes_))

# classification report as dict
report_dict = classification_report(y_test, all_preds, target_names=le.classes_, output_dict=True)

# Converting to DF & filter only the class metrics (exclude avg/total)
report_df = pd.DataFrame(report_dict).transpose().iloc[:-3]
# Reseting index & melting for plotting
report_melted = report_df.reset_index().melt(id_vars="index", value_vars=["precision", "recall", "f1-score"])
report_melted.columns = ["Class", "Metric", "Score"]

# === Classification Report Bar Chart ===
'''The classification report is visualized using a bar chart.
    The chart shows precision, recall, and F1-score for each class.'''
print("\n📊 Plotting classification report...")

# Using seaborn to create a bar plot for the classification report
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x="Class", y="Score", hue="Metric", data=report_melted, color="red", palette="Set2")
plt.title("Classification Metrics per Class")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(title="Metric")

plt.tight_layout()
plt.savefig("classification_report_barplot.png")

plt.show()
print("✅Classification report bar plot saved as classification_report_barplot.png")

# === Plotting Training Loss Charts ===
'''The training loss values are plotted to visualize the model's performance during training.
    X-axis represents the epoch number, & Y-axis is the loss value.'''
print("📉 Plotting training loss charts...")

epochs = list(range(1, len(train_losses) + 1))
df = pd.DataFrame({
    "Epoch": epochs,
    "Loss": train_losses
})
df["Epoch Group"] = (df["Epoch"] - 1) // 10 * 10 + 1  # For grouped bar chart

sns.set(style="whitegrid")
plt.figure(figsize=(14, 20))

# Using a line plot to visualize the training loss over epochs
plt.subplot(4, 1, 1)
sns.lineplot(x="Epoch", y="Loss", data=df, marker='o', color='blue', label='Training Loss')
plt.title("Line Plot: Training Loss Over Epochs")

plt.legend()
plt.tight_layout()

plt.savefig("training_loss.png")
plt.show()
print("✅Training loss scatter plot saved as training_loss.png")


# Using a scatter plot to visualize the training loss over epochs
sns.scatterplot(x="Epoch", y="Loss", data=df, color='purple', label='Training Loss Scatter')
plt.title("Training Loss Over Epoch-wise visualized with Scatter Plot")
plt.legend()
plt.tight_layout()

plt.savefig("training_loss_scatter.png")
plt.show()
print("✅Training loss scatter plot saved as training_loss_scatter.png")

# Using a bar plot to visualize the training loss over epochs  (Grouped by 10 Epochs)
plt.subplot(4, 1, 3)
sns.barplot(x="Epoch Group", y="Loss", data=df, errorbar=None, color="steelblue", label='Avg Loss')
plt.xlabel("Epoch Group")

plt.legend()
plt.tight_layout()
plt.title("Avg. Training Loss per 10-Epoch Group via Bar Plot")

plt.savefig("training_loss_bar.png")

plt.show()
print("✅Training loss bar plot saved as training_loss_bar.png")

# Using a histogram to visualize the distribution of training loss values
plt.subplot(4, 1, 4)
sns.histplot(df["Loss"], bins=20, kde=True, color='green', label='Loss Distribution')

plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.legend()
plt.title("Histogram: Distribution of Training Loss")

plt.tight_layout()
plt.savefig("training_loss_histogram.png")

plt.show()
print("✅Training loss histogram saved as training_loss_histogram.png")

# Using a box plot to visualize the distribution of training loss values
plt.subplot(4, 1, 4)
sns.boxplot(x="Loss", data=df, color='pink', label='Loss Distribution')
plt.legend()
plt.tight_layout()
plt.title("Box Plot: Distribution of Training Loss")
plt.savefig("training_loss_box.png")
plt.show()
print("✅Training loss box plot saved as training_loss_box.png")


# === Plotting Confusion Matrix =

'''The confusion matrix is visualized using a heatmap.
    The heatmap shows the number of true positives, false positives, true negatives, and false negatives for each class.'''
print("\n📊 Plotting confusion matrix...")

# Generate confusion matrix
cm = confusion_matrix(y_test, all_preds)

# Plotting the confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("✅Confusion matrix saved as confusion_matrix.png")



# === Confusion Matrix Bar Chart ===
'''The confusion matrix is visualized using a bar chart ( just another way to visualize it).
    The chart shows the number of true positives, false positives, true negatives, & false negatives for each class.'''
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
cm_df = cm_df.reset_index().melt(id_vars="index", value_vars=le.classes_)
cm_df.columns = ["True Class", "Predicted Class", "Count"]
plt.figure(figsize=(10, 6))
sns.barplot(data=cm_df, x="True Class", y="Count", hue="Predicted Class")
plt.title("Confusion Matrix Bar Chart")
plt.ylabel("Count")
plt.ylim(0, cm.max() + 5)
plt.grid(True)
plt.tight_layout()
plt.savefig("confusion_matrix_bar.png")
plt.show()
print("✅Confusion matrix bar chart saved as confusion_matrix_bar.png")


# === Classification Metrics Bar Chart ===
'''The classification report is visualized using a bar chart.
    The chart shows precision, recall, & F1-score for each class.'''
report = classification_report(y_test, all_preds, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().drop('accuracy', errors='ignore')


print("\nMacro avg:", report['macro avg'])
print("Weighted avg:", report['weighted avg'])

print("\n📊 Plotting classification metrics bar chart...")

plt.figure(figsize=(10, 6))
sns.barplot(data=report_df.loc[le.classes_][['precision', 'recall', 'f1-score']])
plt.title("Model Performance per Class")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.grid(True)

plt.tight_layout()
plt.savefig("classification_metrics.png")

plt.show()
print("✅Classification metrics bar chart saved as classification_metrics.png")

# === Stacked Histogram ===
'''The stacked histogram shows the distribution of predicted classes for each true class.
    This can help visualize how well the model is performing across different classes.'''

pred_df = pd.DataFrame({
    'True': le.inverse_transform(y_test),
    'Predicted': le.inverse_transform(all_preds)
})

plt.figure(figsize=(10, 6))
sns.histplot(data=pred_df, x="Predicted", hue="True", multiple="stack", shrink=0.8, palette="Set2")
plt.title("Stacked Histogram of Predictions by True Class")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.grid(True)
plt.legend(title="True Class")

plt.tight_layout()
plt.savefig("stacked_histogram.png")

plt.show()
print("✅Stacked Historgram saved as stacked_histogram.png")

# === Stacked Bar Chart ===
'''The stacked bar chart shows the distribution of predicted classes for each true class (Predicted class distribution per true class)
    This can help spot misclassifications & visualize how well the model is performing across different classes.'''
plt.figure(figsize=(10, 6))
sns.countplot(data=pred_df, x="Predicted", hue="True", palette="Set2")
plt.title("Stacked Bar Chart of Predictions by True Class")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.grid(True)
plt.legend(title="True Class")
plt.tight_layout()
plt.savefig("stacked_bar_chart.png")
plt.show()
print("✅Stacked bar chart saved as stacked_bar_chart.png")     

# === Predicted Class Distribution ===
'''The predicted class distribution shows the proportion of each predicted class(Flat count/proportion of predicted classes only)
    This can help in detecting prediction imbalance & visualize the model's performance across different classes.'''

# Convert predictions to a Series
pred_series = pd.Series(all_preds)

# Calculate the distribution of predicted classes
pred_dist = pred_series.value_counts(normalize=True).reset_index()
pred_dist.columns = ['Predicted Class', 'Proportion']

plt.figure(figsize=(10, 6))
sns.barplot(data=pred_dist, x='Predicted Class', y='Proportion', color='salmon')


plt.xlabel("Predicted Class")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title="Predicted Class")

plt.tight_layout()
plt.title("Predicted Class Distribution")
plt.savefig("predicted_class_distribution.png")
plt.show()

print("✅Predicted class distribution saved as predicted_class_distribution.png")


# === Semantic Scatterplot (requires matching original CSV) ===
'''The scatterplot visualizes the relationship between text length and likes, colored by popularity class.
    The size of the points represents the number of repins. '''
try:
    orig_df = pd.read_csv("pinterest_cleaned.csv")
    if all(col in orig_df.columns for col in ["text_len", "likes", "popularity_class", "repin_count"]):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=orig_df,
            x="text_len",
            y="likes",
            hue="popularity_class",
            size="repin_count",
            sizes=(20, 200),
            palette="coolwarm"
        )
        plt.title("Scatterplot: Likes vs Text Length by Popularity")
        plt.xlabel("Text Length")
        plt.ylabel("Likes")
        plt.legend(title="Popularity Class")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("semantic_scatterplot.png")
        
        plt.show()
        print("✅Semantic scatterplot saved as semantic_scatterplot.png")
except FileNotFoundError:
    print("'pinterest_cleaned.csv' not found or missing columns. Scatterplot not generated.")

# === Saving the evaluation results to text file ===
with open("eval_metrics_summary.txt", "w") as f:
    f.write("🎯 Evaluation Summary\n")
    f.write("="*40 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write("Macro Average:\n")
    for k, v in report['macro avg'].items():
        f.write(f"  {k}: {v:.4f}\n")

    f.write("\nWeighted Average:\n")
    for k, v in report['weighted avg'].items():
        f.write(f"  {k}: {v:.4f}\n")

    f.write("\n📊 Full Classification Report:\n")
    f.write(classification_report(y_test, all_preds, target_names=le.classes_))

    f.write("\n🧾 Confusion Matrix:\n")
    cm_str = pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string()
    f.write(cm_str)
    
    f.write("\n📉 Training Loss Summary:\n")
    f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
    f.write(f"Best Training Loss: {min(train_losses):.4f} (Epoch {num.argmin(train_losses) + 1})\n")
    f.write(f"Standard Deviation of Training Loss: {num.std(train_losses):.4f}\n")
    f.write("="*40 + "\n")

    f.write("\n📉 Average Training Loss (every 10 epochs):\n")
    for i in range(0, len(train_losses), 10):
        avg = num.mean(train_losses[i:i+10])
        f.write(f"Epochs {i+1}-{i+10}: {avg:.4f}\n")
