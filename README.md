# Sarcasm-Aware Emotion Analysis 🎭

A transformer-based multi-task learning model that performs simultaneous emotion classification and sarcasm detection on social media text using **BERTweet**. Built with PyTorch and deployed via Gradio.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project implements a **multi-task learning approach** where a single BERTweet-based model learns to:
- Classify emotions into 8 categories (anger, disgust, fear, joy, neutral, sadness, shame, surprise)
- Detect sarcasm/irony (binary classification)

By training both tasks jointly with a shared encoder, the model achieves better contextual understanding of social media text.

## Key Features

- **Multi-Task Architecture**: Shared BERTweet encoder with dual task-specific classification heads
- **Strong Performance**: 75.1% F1 for emotion classification, 80.1% F1 for sarcasm detection
- **Production-Ready**: Interactive Gradio web interface for real-time predictions
- **Modular Pipeline**: Clean separation of data loading, cleaning, and preprocessing
- **Twitter-Optimized**: Specialized tokenization for tweets (handles @USER, HTTPURL, emojis)

## Technical Stack

- **Model**: BERTweet (vinai/bertweet-base)
- **Framework**: PyTorch, Transformers
- **Deployment**: Gradio
- **Data Processing**: Pandas, Scikit-learn
- **Datasets**: HuggingFace Datasets (tweet_eval), Custom emotion dataset

## Results

| Task | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Emotion Classification | 75.7% | 75.9% | 74.7% | **75.1%** |
| Sarcasm Detection | 80.3% | 80.1% | 80.1% | **80.1%** |

**Combined Macro F1**: 77.6%

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NeoSockCheng/sarcasm-aware-emotion-analysis.git
cd sarcasm-aware-emotion-analysis

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# 1. Download datasets
python src/_01_data_loading.py

# 2. Clean and normalize text
python src/_02_data_cleaning.py

# 3. Preprocess and create splits
python src/_03_data_preprocessing.py

# 4. Train model (via Jupyter notebook)
# Open and run: notebooks/Multitask_model.ipynb

# 5. Launch web interface
python src/_05_interface.py
```

## Project Structure

```
├── data/
│   ├── raw/              # Original datasets
│   ├── cleaned/          # Normalized text
│   └── preprocessed/     # Train/val/test splits
├── notebooks/            # EDA and model training
├── src/
│   ├── _01_data_loading.py
│   ├── _02_data_cleaning.py
│   ├── _03_data_preprocessing.py
│   ├── _04_multitask_model.py
│   └── _05_interface.py  # Gradio deployment
└── results/              # Metrics and confusion matrices
```

## Model Architecture

```
Input Text → BERTweet Tokenizer → BERTweet Encoder → Dropout(0.3)
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                         ↓
            Emotion Head (8 classes)              Sarcasm Head (binary)
```

## Use Cases

- Social media sentiment analysis with sarcasm awareness
- Customer feedback analysis
- Content moderation
- Brand monitoring on Twitter
- Mental health monitoring in online communities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **BERTweet**: Pre-trained language model for English Tweets ([VinAI Research](https://github.com/VinAIResearch/BERTweet))
- **Datasets**: tweet_eval (HuggingFace), Emotion Detection dataset (SannketNikam)

## Contact

For questions or collaboration opportunities, feel free to reach out via [GitHub Issues](https://github.com/NeoSockCheng/sarcasm-aware-emotion-analysis/issues).

---

*Developed as a demonstration of multi-task learning and transformer-based NLP for social media analysis.*