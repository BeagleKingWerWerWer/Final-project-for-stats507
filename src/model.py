import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaClassifier(nn.Module):
    def __init__(self, pretrained_model='roberta-base'):
        super(RobertaClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits
