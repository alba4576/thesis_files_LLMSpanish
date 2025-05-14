import torch
from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report

# Label mapping
# label2id = {'ES': 0, 'MX': 1, 'VE': 2}
label2id = {'YUC': 0, 'Mexico_City': 1, 'MXC': 2}
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

# country data
# df1 = pd.read_csv('es_100.csv')
# df2 = pd.read_csv('mex_100.csv')
# df3 = pd.read_csv('ven_100.csv')

# regional data
df1 = pd.read_csv('yuc_10.csv')
df2 = pd.read_csv('cdmx_10.csv')
df3 = pd.read_csv('mxc_10.csv')

total_df = pd.concat([df1, df2, df3], ignore_index=True)
total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)
total_df['label'] = total_df['location'].map(label2id)

texts = total_df['text'].tolist()
true_labels = total_df['label'].tolist()

# Initialize zero-shot-classification pipeline
pipe = pipeline("zero-shot-classification", model="meta-llama/Llama-3.2-1B")

# Candidate labels
candidate_labels = list(label2id.keys())

# Run inference in batches
batch_size = 16
preds = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    results = pipe(batch_texts, candidate_labels=candidate_labels)
    
    # The output is a list of dicts, one per input
    for result in results:
        top_label = result['labels'][0]
        preds.append(label2id.get(top_label, -1))

# Evaluate
filtered_preds = [p for p in preds if p != -1]
filtered_actuals = [a for p, a in zip(preds, true_labels) if p != -1]

print("\nClassification Report:")
print(classification_report(filtered_actuals, filtered_preds, target_names=label_list))

# Save results
results_df = pd.DataFrame({
    "text": texts,
    "actual": [id2label[i] for i in true_labels],
    "predicted": [id2label.get(i, "UNK") for i in preds]
})
filename = "llama_zero_shot_regional_predictions.csv"

results_df.to_csv(filename, index=False)
print(f"Saved to {filename}")
