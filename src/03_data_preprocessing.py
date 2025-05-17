from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def preprocess_for_multitask(emotion_df, sarcasm_df, save_csv_path=None):
    # Emotion dataset labels
    emotion_texts = emotion_df['text'].tolist()
    emotion_labels = emotion_df['emotion_label'].tolist()
    sarcasm_labels_for_emotion = [-1] * len(emotion_labels)

    # Sarcasm dataset labels
    sarcasm_texts = sarcasm_df['text'].tolist()
    sarcasm_labels = sarcasm_df['sarcasm_label'].tolist()
    emotion_labels_for_sarcasm = [-1] * len(sarcasm_labels)

    # Combine texts and labels
    combined_texts = emotion_texts + sarcasm_texts
    combined_emotion_labels = emotion_labels + emotion_labels_for_sarcasm
    combined_sarcasm_labels = sarcasm_labels_for_emotion + sarcasm_labels

    # Save combined raw data to CSV if path given
    if save_csv_path:
        import pandas as pd
        combined_df = pd.DataFrame({
            'text': combined_texts,
            'emotion_label': combined_emotion_labels,
            'sarcasm_label': combined_sarcasm_labels
        })
        combined_df.to_csv(save_csv_path, index=False)
        print(f"Saved combined data to {save_csv_path}")

    return combined_texts, combined_emotion_labels, combined_sarcasm_labels

# Encode string emotion labels to integers
def encode_emotion_labels(emotion_df):
    encoder = LabelEncoder()
    emotion_df['emotion_label'] = encoder.fit_transform(emotion_df['emotion_label'])
    return emotion_df, encoder

# Prepare train/val datasets with sarcasm split
def preprocess_split(emotion_df, sarcasm_all):
    # Split emotion data
    emo_train, emo_temp = train_test_split(emotion_df, test_size=0.2, random_state=42)
    emo_val, emo_test = train_test_split(emo_temp, test_size=0.5, random_state=42)  # 0.1 each

    # Split sarcasm data with same ratios
    sarc_train, sarc_temp = train_test_split(sarcasm_all, test_size=0.2, random_state=42)
    sarc_val, sarc_test = train_test_split(sarc_temp, test_size=0.5, random_state=42)  # 0.1 each

    train_dataset = preprocess_for_multitask(emo_train, sarc_train)
    val_dataset = preprocess_for_multitask(emo_val, sarc_val)
    test_dataset = preprocess_for_multitask(emo_test, sarc_test)
    return train_dataset, val_dataset, test_dataset

def main():
    # Load CSVs
    emotion_df = pd.read_csv("data/cleaned/emotion_cleaned.csv")
    sarcasm_train = pd.read_csv("data/cleaned/sarcasm_train_cleaned.csv")
    sarcasm_val = pd.read_csv("data/cleaned/sarcasm_validation_cleaned.csv")
    sarcasm_test = pd.read_csv("data/cleaned/sarcasm_test_cleaned.csv")

    # Combine all sarcasm data
    sarcasm_all = pd.concat([sarcasm_train, sarcasm_val, sarcasm_test], ignore_index=True)

    # Encode emotion labels
    emotion_df, emotion_encoder = encode_emotion_labels(emotion_df)

    # Save emotion label map (int → emotion string) as JSON for reference
    emotion_label_map = {int(idx): label for idx, label in enumerate(emotion_encoder.classes_)}
    with open("data/preprocessed/emotion_label_map.json", "w") as f:
        json.dump(emotion_label_map, f, indent=4)
    print(f"Saved emotion label map to data/preprocessed/emotion_label_map.json")

    # Combine and save CSV
    preprocess_for_multitask(emotion_df, sarcasm_all, save_csv_path="data/preprocessed/combined_multitask_dataset.csv")

if __name__ == "__main__":
    main()
