import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

sys.path.append(os.path.abspath("src"))

from data_loader import load_imdb_dataset, IMDBDataset
from model import RobertaClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RobertaClassifier()
model.load_state_dict(torch.load("outputs/saved_model/roberta_attn.pt", map_location=device))
model.to(device)
model.eval()

_, df_test, _ = load_imdb_dataset()
test_texts = df_test['text'].tolist()
test_labels = df_test['label'].astype(int).tolist()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=16)

all_preds, all_trues = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds += preds.cpu().tolist()
        all_trues += labels.cpu().tolist()

acc = accuracy_score(all_trues, all_preds)
f1 = f1_score(all_trues, all_preds)
print(f" Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

cm = confusion_matrix(all_trues, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

os.makedirs("outputs/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix on IMDB Test Set")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix.png")
print("Confusion matrix saved to outputs/figures/confusion_matrix.png")
plt.show()
