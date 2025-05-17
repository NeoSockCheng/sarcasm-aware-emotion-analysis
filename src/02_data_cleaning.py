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
    # Remove non-standard/control characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)
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
        ("data/raw/emotion_raw.csv", "your_train_cleaned.csv"),
        ("sarcasm_test_raw.csv", "your_test_cleaned.csv"),
        ("sarcasm_train_raw.csv", "your_val_cleaned.csv"),
        ("sarcasm_validation_raw.csv", "your_val_cleaned.csv"),
    ]
    for input_path, output_path in files_to_clean:
        try:
            clean_and_export(input_path, output_path)
        except Exception as e:
            print(f"Failed to clean {input_path}: {e}")

if __name__ == "__main__":
    main()