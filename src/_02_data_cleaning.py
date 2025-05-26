import re
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Replace URLs with [URL]
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    # Replace numbers with [NUM]
    text = re.sub(r"\b\d+\b", "[NUM]", text)
    # Remove redundant mentions (keep only the first @user in a sequence)
    text = re.sub(r'(@\w+)(\s+@\w+)+', r'\1', text)
    # Strip multiple spaces, tabs, or newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_text(text, max_tokens=218):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text

def clean_and_truncate_dataframe(df):
    # Ensure 'text' column exists
    if 'text' not in df.columns:
        if 'Text' in df.columns:
            df = df.rename(columns={'Text': 'text'})
        else:
            raise ValueError("No 'text' or 'Text' column found in the dataframe.")
    # Rename Emotion to emotion_label if exists
    if 'Emotion' in df.columns:
        df = df.rename(columns={'Emotion': 'emotion_label'})
    # Rename label to sarcasm_label if exists
    if 'label' in df.columns:
        df = df.rename(columns={'label': 'sarcasm_label'})
    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)
    # Truncate to 218 tokens
    df["text"] = df["text"].apply(lambda x: truncate_text(x, max_tokens=218))
    # Remove duplicated text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df

def clean_and_export(input_path, output_path):
    df = pd.read_csv(input_path)
    df_clean = clean_and_truncate_dataframe(df)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned file saved to {output_path}")

def main():
    # Example usage for multiple files
    files_to_clean = [
        ("data/raw/emotion_raw.csv", "data/cleaned/emotion_cleaned.csv"),
        ("data/raw/sarcasm_test_raw.csv", "data/cleaned/sarcasm_test_cleaned.csv"),
        ("data/raw/sarcasm_train_raw.csv", "data/cleaned/sarcasm_train_cleaned.csv"),
        ("data/raw/sarcasm_validation_raw.csv", "data/cleaned/sarcasm_validation_cleaned.csv"),
    ]
    for input_path, output_path in files_to_clean:
        try:
            clean_and_export(input_path, output_path)
        except Exception as e:
            print(f"Failed to clean {input_path}: {e}")

if __name__ == "__main__":
    main()