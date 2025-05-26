import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = self.labels[idx]  # emotion labels
        return item


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = self.labels[idx]  # sarcasm labels
        return item


def load_emotion_datasets(tokenizer):
    train_df = pd.read_csv("data/preprocessed/emotion_train.csv")
    val_df = pd.read_csv("data/preprocessed/emotion_validation.csv")
    test_df = pd.read_csv("data/preprocessed/emotion_test.csv")

    return (
        EmotionDataset(train_df["text"].tolist(), train_df["emotion_label"].tolist(), tokenizer),
        EmotionDataset(val_df["text"].tolist(), val_df["emotion_label"].tolist(), tokenizer),
        EmotionDataset(test_df["text"].tolist(), test_df["emotion_label"].tolist(), tokenizer),
    )


def load_sarcasm_datasets(tokenizer):
    train_df = pd.read_csv("data/preprocessed/sarcasm_train.csv")
    val_df = pd.read_csv("data/preprocessed/sarcasm_validation.csv")
    test_df = pd.read_csv("data/preprocessed/sarcasm_test.csv")

    return (
        SarcasmDataset(train_df["text"].tolist(), train_df["sarcasm_label"].tolist(), tokenizer),
        SarcasmDataset(val_df["text"].tolist(), val_df["sarcasm_label"].tolist(), tokenizer),
        SarcasmDataset(test_df["text"].tolist(), test_df["sarcasm_label"].tolist(), tokenizer),
    )
    
def get_dataloaders(emotion_batch_size=64, sarcasm_batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    # Load datasets
    emotion_train, emotion_val, emotion_test = load_emotion_datasets(tokenizer)
    sarcasm_train, sarcasm_val, sarcasm_test = load_sarcasm_datasets(tokenizer)

    # Create DataLoaders
    emotion_train_loader = DataLoader(emotion_train, batch_size=emotion_batch_size, shuffle=True)
    emotion_val_loader = DataLoader(emotion_val, batch_size=emotion_batch_size, shuffle=False)
    emotion_test_loader = DataLoader(emotion_test, batch_size=emotion_batch_size, shuffle=False)

    sarcasm_train_loader = DataLoader(sarcasm_train, batch_size=sarcasm_batch_size, shuffle=True)
    sarcasm_val_loader = DataLoader(sarcasm_val, batch_size=sarcasm_batch_size, shuffle=False)
    sarcasm_test_loader = DataLoader(sarcasm_test, batch_size=sarcasm_batch_size, shuffle=False)

    return emotion_train_loader, sarcasm_train_loader, emotion_val_loader, sarcasm_val_loader, emotion_test_loader, sarcasm_test_loader