# Mini Offline Chatbot

## Setup
1. Create venv
2. Install dependencies (`pip install -r requirements.txt`)
3. Run: `python chatbot.py --data_dir ./data`

## Method
Used **TF-IDF + cosine similarity** for efficient offline retrieval.  
Chosen for simplicity, transparency, and no need for heavy models.

# Mini Offline Chatbot

## Project Overview
This project is a **Mini Offline Chatbot** that answers user queries from a small text corpus stored locally.  
It works fully **offline** and uses **TF-IDF vectorization** and **cosine similarity** to find the most relevant paragraph for each query.  

---

## Project Structure

mini_chatbot/
│
├── chatbot.py # Main chatbot program
├── data/ # Folder containing text files (knowledge base)
│ ├── file1.txt
│ ├── file2.txt
│ └── file3.txt
├── outputs/ # Folder for storing query logs and outputs
│ └── session.jsonl
├── README.md # This documentation file
├── provenance.md # List of data sources and licenses
├── requirements.txt # Python packages required
└── venv/ # Local virtual environment (optional, not pushed)
