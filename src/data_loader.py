import pandas as pd
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset

splits = {
    'train': 'plain_text/train-00000-of-00001.parquet',
    'test': 'plain_text/test-00000-of-00001.parquet',
    'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'
}
def load_imdb_dataset():
    prefix = "hf://datasets/stanfordnlp/imdb/"
    df_train = pd.read_parquet(prefix + splits["train"])
    df_test = pd.read_parquet(prefix + splits["test"])
    df_unsup = pd.read_parquet(prefix + splits["unsupervised"])
    return df_train, df_test, df_unsup

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
