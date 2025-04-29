import pandas as pn
import numpy as num
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel



# === Loading the cleaned data ===
print("Loading the cleaned Pinterest dataset...")
df = pn.read_csv("pinterest_cleaned.csv")
texts = df['description'].fillna("").tolist()

# === Loading DistilBERT ===
print("Loading the DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings = [] #generating the embeddings

print("Generating BERT embeddings...")
for text in tqdm(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(pooled.squeeze().cpu().numpy())

# === Saving the features to CSV ===
print("Saving the embeddings as a CSV...")
embeddings = num.vstack(embeddings)
bert_df = pn.DataFrame(embeddings)
bert_df['label'] = df['popularity_class'].values
bert_df.to_csv("bert_features.csv", index=False)

print("✅ And we're Finished! BERT features saved to bert_features.csv")