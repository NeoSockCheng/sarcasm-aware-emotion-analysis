import gradio as gr
import pandas as pd
from datetime import datetime
import os

# Global variable to store history
history = []

def analyze(text):
    # Mock analysis with confidence levels
    emotion = "Happy"
    sarcasm = "No"
    emotion_conf = 0.92  # 92% confidence
    sarcasm_conf = 0.15  # 15% confidence
    
    # Format outputs with confidence
    emotion_output = f"Emotion: {emotion} (Confidence: {emotion_conf*100:.1f}%)"
    sarcasm_output = f"Sarcasm: {sarcasm} (Confidence: {sarcasm_conf*100:.1f}%)"
    
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

'''
import gradio as gr

def analyze(text):
    return "Emotion: Happy", "Sarcasm: No"

iface = gr.Interface(fn=analyze, 
                     inputs=gr.Textbox(lines=2, placeholder="Enter text here..."), 
                     outputs=["text", "text"])

iface.launch()
'''