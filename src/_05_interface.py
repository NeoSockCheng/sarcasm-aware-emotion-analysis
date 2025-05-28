import gradio as gr
import pandas as pd
from datetime import datetime
import os
import torch
from transformers import AutoTokenizer
from _04_multitask_model import MultiTaskModel
import json

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = MultiTaskModel().to(device)

# Load the saved model weights
model_path = "results/train_val/best_multitask_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load emotion label map
with open("data/preprocessed/emotion_label_map.json", "r") as f:
    emotion_label_map = json.load(f)

# Global variable to store history
history = []

def predict(text):
    # Tokenize input
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get predictions
    with torch.no_grad():
        emotion_logits = model(input_ids, attention_mask, task="emotion")
        sarcasm_logits = model(input_ids, attention_mask, task="sarcasm")
    
    # Process emotion prediction
    emotion_probs = torch.softmax(emotion_logits, dim=1)
    emotion_pred = torch.argmax(emotion_probs).item()
    emotion_label = emotion_label_map[str(emotion_pred)]
    emotion_conf = emotion_probs[0][emotion_pred].item()
    
    # Process sarcasm prediction
    sarcasm_probs = torch.softmax(sarcasm_logits, dim=1)
    sarcasm_pred = torch.argmax(sarcasm_probs).item()
    sarcasm_label = "Yes" if sarcasm_pred == 1 else "No"
    sarcasm_conf = sarcasm_probs[0][sarcasm_pred].item()
    
    return emotion_label, emotion_conf, sarcasm_label, sarcasm_conf

def analyze(text):
    if not text.strip():
        return "Please enter some text", "", ""
    
    try:
        emotion_label, emotion_conf, sarcasm_label, sarcasm_conf = predict(text)
        
        # Format outputs with confidence
        emotion_output = f"Emotion: {emotion_label} (Confidence: {emotion_conf*100:.1f}%)"
        sarcasm_output = f"Sarcasm: {sarcasm_label} (Confidence: {sarcasm_conf*100:.1f}%)"
        
        # Add to history (store only last 5 entries)
        history.append((text, emotion_output, sarcasm_output))
        if len(history) > 5:
            history.pop(0)
        
        # Prepare history display
        history_text = "\n".join([
            f"{i+1}. {item[0]}\n   → {item[1]}\n   → {item[2]}\n" 
            for i, item in enumerate(reversed(history))
        ]) if history else "No history yet"
        
        return emotion_output, sarcasm_output, history_text
    
    except Exception as e:
        return f"Error: {str(e)}", "", ""

def save_to_csv(text, emotion_output, sarcasm_output):
    # Create directory if it doesn't exist
    os.makedirs("flagged", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    data = {
        "text": [text],
        "output_0": [emotion_output],
        "output_1": [sarcasm_output],
        "timestamp": [timestamp]
    }
    
    try:
        # Try to read existing file
        df = pd.read_csv("flagged/user_input.csv")
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df], ignore_index=True)
    except FileNotFoundError:
        # Create new file if it doesn't exist
        df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv("flagged/user_input.csv", index=False)
    return "Data saved successfully!"

with gr.Blocks() as app:
    gr.Markdown("## Emotion & Sarcasm Analyzer")
    gr.Markdown("This tool analyzes text for both emotion (8 categories) and sarcasm using a BERTweet-based multitask model.")
    
    # Input section
    input_text = gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text")
    analyze_btn = gr.Button("Analyze")
    
    # Output section
    with gr.Row():
        with gr.Column():
            emotion_output = gr.Textbox(label="Emotion Detection")
            sarcasm_output = gr.Textbox(label="Sarcasm Detection")
    
    # Flag button section
    flag_btn = gr.Button("Flag/Save Results")
    save_status = gr.Textbox(label="Save Status", interactive=False)
    
    # History section
    gr.Markdown("### History (last 5 analyses)")
    history_output = gr.Textbox(label="", interactive=False)
    
    # Analysis button click
    analyze_btn.click(
        fn=analyze,
        inputs=input_text,
        outputs=[emotion_output, sarcasm_output, history_output]
    )
    
    # Flag button click
    flag_btn.click(
        fn=save_to_csv,
        inputs=[input_text, emotion_output, sarcasm_output],
        outputs=save_status
    )

app.launch()