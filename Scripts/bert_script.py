import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from datetime import datetime

device = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    "cuda" if torch.cuda.is_available() else 
    "cpu"
)

# country datasets
# df1 = pd.read_csv('es_100.csv')
# df2 = pd.read_csv('mex_100.csv')
# df3 = pd.read_csv('ven_100.csv')

# regional datasets 
df1 = pd.read_csv('yuc_10.csv')
df2 = pd.read_csv('cdmx_10.csv')
df3 = pd.read_csv('mxc_10.csv')

#dataframe combined
total_df = pd.concat([df1, df2, df3], ignore_index=True)
shuffled_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)

#mapping
# label2id = {'ES': 0, 'MX': 1, 'VE': 2}
label2id = {'YUC': 0, 'Mexico_City': 1, 'MXC': 2}
id2label = {v: k for k, v in label2id.items()}
shuffled_df['label'] = shuffled_df['location'].map(label2id)

# 80/10/10 split: train, val, test
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    shuffled_df['text'].tolist(), shuffled_df['label'].tolist(), 
    test_size=0.2, stratify=shuffled_df['label'], random_state=42
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, 
    test_size=0.5, stratify=temp_labels, random_state=42
)

# Tokenizer, model
bert_model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        x = self.dropout(pooled)
        return self.fc(x)

model = BERTClassifier(bert_model_name, num_classes=3).to(device)

class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# Hyperparameters
max_length = 128
batch_size = 16
epochs = 4 
lr = 2e-5
weight_decay = 0.01

# Data, loaders
train_dataset = DialectDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = DialectDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = DialectDataset(test_texts, test_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

scaler = None
# Mixed precision (for GPU)
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

# Early stopping
best_val_loss = float('inf')
best_model_state = None
patience = 2
epochs_no_improve = 0

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    return preds, actuals

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", dynamic_ncols=True)

    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        if scaler:
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        pbar.set_postfix(loss=loss.item())

    # Validation
    val_preds, val_actuals = evaluate(model, val_loader, device)
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_actuals, val_preds)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    print(classification_report(val_actuals, val_preds, target_names=id2label.values()))

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_model_state)

# Test evaluation
test_preds, test_actuals = evaluate(model, test_loader, device)
test_acc = accuracy_score(test_actuals, test_preds)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(classification_report(test_actuals, test_preds, target_names=id2label.values()))

# Save model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f"trained_model_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
