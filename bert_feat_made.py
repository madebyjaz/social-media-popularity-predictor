import pandas as pn
import numpy as num
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

# === Loading the cleaned data ===
# This extracts BERT features from the preprocess and cleaned Pinterest dataset and saves them to a CSV file.
# Don't forget: Run this script in the same directory as the Pinterest dataset.
print("Alright, loading the cleaned Pinterest dataset...")
df = pn.read_csv("pinterest_cleaned.csv")
texts = df['description'].fillna("").tolist()

# === Extracting BERT features ===
'''Load the DistilBERT tokenizer and model.

The DistilBERT model is a smaller (by 40%), faster (by 60%), and lighter, overall being a more effient version of BERT. 
    The tokenizer converts text into tokens that the model can understand.
    The model processes the tokens and outputs a fixed-size vector representation of the text. '''

print("✅Done! Now loading the DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print("✅Done! Now loading the DistilBERT model...")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

bert_features = []

print("Extracting the features...")
for text in tqdm(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        pooled_np = pooled.squeeze().cpu().numpy()
        bert_features.append(pooled_np)

# Save features immediately to CSV (to avoid memory issues with large datasets)
print("✅Done! Now saving the features to CSV file...")
# Convert the list of features to a NumPy array and then to a DataFrame
bert_array = num.array(bert_features)
bert_df = pn.DataFrame(bert_array)

''' Add the label column (popularity class from the OG dataset) to the DataFrame
    Note that the labels are not BERT features, but they are included for reference. '''
bert_df['label'] = df['popularity_class'].values
bert_df.to_csv("bert_features.csv", index=False)

print("✅ Anddd we're Finished! BERT features saved to bert_features.csv")
