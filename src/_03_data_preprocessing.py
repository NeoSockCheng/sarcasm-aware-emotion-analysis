from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os

def preprocess_for_multitask(emotion_df, sarcasm_df):
    # Prepare labels for multi-task (-1 for missing)
    emotion_texts = emotion_df['text'].tolist()
    emotion_labels = emotion_df['emotion_label'].tolist()
    sarcasm_labels_for_emotion = [-1] * len(emotion_labels)

    sarcasm_texts = sarcasm_df['text'].tolist()
    sarcasm_labels = sarcasm_df['sarcasm_label'].tolist()
    emotion_labels_for_sarcasm = [-1] * len(sarcasm_labels)

    combined_texts = emotion_texts + sarcasm_texts
    combined_emotion_labels = emotion_labels + emotion_labels_for_sarcasm
    combined_sarcasm_labels = sarcasm_labels_for_emotion + sarcasm_labels

    combined_df = pd.DataFrame({
        'text': combined_texts,
        'emotion_label': combined_emotion_labels,
        'sarcasm_label': combined_sarcasm_labels
    })
    return combined_df

def encode_emotion_labels(emotion_df):
    encoder = LabelEncoder()
    emotion_df['emotion_label'] = encoder.fit_transform(emotion_df['emotion_label'])
    return emotion_df, encoder

def save_splits(df_dict, prefix):
    os.makedirs("data/preprocessed", exist_ok=True)
    for split_name, df in df_dict.items():
        path = f"data/preprocessed/{prefix}_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {split_name} split to {path}")

def main():
    # Load datasets
    emotion_df = pd.read_csv("data/cleaned/emotion_cleaned.csv")
    sarcasm_train = pd.read_csv("data/cleaned/sarcasm_train_cleaned.csv")
    sarcasm_val = pd.read_csv("data/cleaned/sarcasm_validation_cleaned.csv")
    sarcasm_test = pd.read_csv("data/cleaned/sarcasm_test_cleaned.csv")

    # Combine all sarcasm data
    sarcasm_all = pd.concat([sarcasm_train, sarcasm_val, sarcasm_test], ignore_index=True)

    # Encode emotion labels
    emotion_df, emotion_encoder = encode_emotion_labels(emotion_df)

    # Save emotion label map
    emotion_label_map = {int(idx): label for idx, label in enumerate(emotion_encoder.classes_)}
    with open("data/preprocessed/emotion_label_map.json", "w") as f:
        json.dump(emotion_label_map, f, indent=4)
    print("Saved emotion label map.")

    # Split sarcasm data: 80% train, then 10% val, 10% test from remaining 20%
    sarcasm_train_split, sarcasm_temp = train_test_split(sarcasm_all, test_size=0.2, random_state=42)
    sarcasm_val_split, sarcasm_test_split = train_test_split(sarcasm_temp, test_size=0.5, random_state=42)

    # Split emotion data: same ratios
    emotion_train_split, emotion_temp = train_test_split(emotion_df, test_size=0.2, random_state=42)
    emotion_val_split, emotion_test_split = train_test_split(emotion_temp, test_size=0.5, random_state=42)

    # Save individual splits
    save_splits({
        "train": sarcasm_train_split,
        "validation": sarcasm_val_split,
        "test": sarcasm_test_split
    }, prefix="sarcasm")

    save_splits({
        "train": emotion_train_split,
        "validation": emotion_val_split,
        "test": emotion_test_split
    }, prefix="emotion")

    # Create combined splits (emotion + sarcasm)
    combined_train = preprocess_for_multitask(emotion_train_split, sarcasm_train_split)
    combined_val = preprocess_for_multitask(emotion_val_split, sarcasm_val_split)
    combined_test = preprocess_for_multitask(emotion_test_split, sarcasm_test_split)

    save_splits({
        "train": combined_train,
        "validation": combined_val,
        "test": combined_test
    }, prefix="combined")

if __name__ == "__main__":
    main()
