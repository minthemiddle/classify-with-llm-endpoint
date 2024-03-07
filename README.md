# Classify message with LLM endpoint

Pass messages in a CSV.  
Reads prompt from text file.  

Replaces placeholder in prompt with real message.  

Classify with an LLM classification API.  
Saves back class to CSV in column `label_classification_llm`.  

**Usage**  
`python -m venv venv`  
`pip install requirements.txt`  
`cp config.ini.example config.ini` and add your LLM API Key.  
`touch prompt.txt`, create prompt, make sure it has a `{{INPUT}}` placeholder.  
`python3 classify-with-llm-endpoint.py FILE.csv`
