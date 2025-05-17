# sarcasm-aware-emotion-analysis
A transformer-based NLP model that enhances tweet emotion classification by incorporating sarcasm detection using multi-task learning to improve contextual understanding.

# Project Initiation
1. Git clone the repo
2. Navigate to the project root (sarcas-aware-emotion-analysis), create a new venv:
```bash
python -m venv venv
source venv\Scripts\activate
# or `source venv/bin/activate` (Linux/macOS)
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

# After first time project initiation
Everytime before running any code in the project:
```bash
cd /path/to/your/project
source venv\Scripts\activate
```
Everytime after you installed/updated any package:
```bash
pip freeze > requirements.txt
```
Everytime after pulling latest update from main:
```bash
pip install -r requirements.txt
```

Always make sure you are on your venv when installing or running anything...
Buttt it is up to you to use .py or .ipynb. These steps is for you to run .py locally, if you only use .ipynb so just run the notebook on the collab, download and upload in this project.