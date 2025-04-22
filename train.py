import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.append("src")

from data_loader import load_imdb_dataset, IMDBDataset
from model import RobertaClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_train, df_test, _ = load_imdb_dataset()
df_train['label'] = df_train['label'].astype(int)
df_test['label'] = df_test['label'].astype(int)

train_texts = df_train['text'].tolist()
train_labels = df_train['label'].tolist()
test_texts = df_test['text'].tolist()
test_labels = df_test['label'].tolist()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_len=128)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

model = RobertaClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].long().to(device)  # 注意：CrossEntropy 需要 long 类型标签

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            pred_labels = torch.argmax(logits, dim=1)
            preds += pred_labels.cpu().tolist()
            trues += labels.cpu().tolist()

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    print("Prediction breakdown:", Counter(preds))
    return acc, f1

EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    acc, f1 = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f} | Accuracy = {acc:.4f} | F1 = {f1:.4f}")

os.makedirs("outputs/saved_model", exist_ok=True)
torch.save(model.state_dict(), "outputs/saved_model/roberta_cls.pt")
print("✅ Model saved to outputs/saved_model/roberta_cls.pt")
