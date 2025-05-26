import os
import pandas as pd
from datasets import load_dataset

def save_to_csv(df, filename):
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{filename}", index=False)
    print(f"Saved {filename} to data/raw/")

def main():
    # Load emotion dataset from GitHub
    emotion_url = 'https://raw.githubusercontent.com/SannketNikam/Emotion-Detection-in-Text/main/data/emotion_dataset_raw.csv'
    emotion_df = pd.read_csv(emotion_url)
    print("Emotion dataset sample:")
    print(emotion_df.head())
    save_to_csv(emotion_df, 'emotion_raw.csv')

    # Load sarcasm dataset from Hugging Face
    sarcasm_ds = load_dataset("tweet_eval", "irony")
    print("Sarcasm dataset structure:")
    print(sarcasm_ds)

    # Convert sarcasm train split to DataFrame and save
    sarcasm_train_df = pd.DataFrame(sarcasm_ds['train'])
    save_to_csv(sarcasm_train_df, 'sarcasm_train_raw.csv')

    # Optional: Save test and validation splits if needed
    sarcasm_test_df = pd.DataFrame(sarcasm_ds['test'])
    save_to_csv(sarcasm_test_df, 'sarcasm_test_raw.csv')

    sarcasm_val_df = pd.DataFrame(sarcasm_ds['validation'])
    save_to_csv(sarcasm_val_df, 'sarcasm_validation_raw.csv')

if __name__ == "__main__":
    main()
