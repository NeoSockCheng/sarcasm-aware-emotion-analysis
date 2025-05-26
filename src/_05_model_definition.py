import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, pretrained_model_name="vinai/bertweet-base", num_emotions=8):
        super(MultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        # Emotion classification head
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)
        # Sarcasm classification head (binary classification)
        self.sarcasm_classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, task):
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token embedding (first token)
        pooled_output = outputs.last_hidden_state[:, 0]

        # Choose output head based on task
        if task == "emotion":
            logits = self.emotion_classifier(pooled_output)
        elif task == "sarcasm":
            logits = self.sarcasm_classifier(pooled_output)
        else:
            raise ValueError("Task must be 'emotion' or 'sarcasm'")

        return logits