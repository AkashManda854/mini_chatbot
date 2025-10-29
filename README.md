# Mini Offline Chatbot 🤖

A lightweight, intelligent chatbot that retrieves information from local documents using TF-IDF vectorization and cosine similarity - fully offline, no internet required!

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9.1-green.svg)](https://www.nltk.org/)

## 🌟 Features

- ✅ **Fully Offline** - Works without internet connectivity
- ✅ **Smart Retrieval** - Uses TF-IDF + Cosine Similarity for accurate results
- ✅ **Fast Response** - Sub-second query processing
- ✅ **Session Logging** - Tracks all interactions with timestamps
- ✅ **Easy to Extend** - Simply add text files to expand knowledge base
- ✅ **Lightweight** - Minimal dependencies, ~70 lines of code

## 🚀 Quick Start

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AkashManda854/mini_chatbot.git
   cd mini_chatbot
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Run the chatbot:**
```bash
python chatbot.py --data_dir ./data
```

**Example interaction:**
```
🤖 Offline Chatbot ready! Type 'exit' to quit.

You: What is artificial intelligence?

📄 From file: file1.txt.txt
💬 Best match (score=0.50):
Artificial intelligence (AI) is the simulation of human intelligence in machines...

You: exit
Goodbye 👋
```

## 📁 Project Structure

```
mini_chatbot/
├── chatbot.py              # Main chatbot application
├── data/                   # Knowledge base (text files)
│   ├── file1.txt.txt      # AI content
│   ├── file2.txt.txt      # Cybersecurity content
│   └── file3.txt.txt      # Climate change content
├── outputs/                # Session logs
│   └── session.jsonl      # Query logs in JSON Lines format
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LINKEDIN_DESCRIPTION.md # Professional LinkedIn summary
├── TECHNOLOGY_DETAILS.md  # Comprehensive technical documentation
├── QUICK_REFERENCE.md     # Copy-paste LinkedIn posts & tips
├── Readme.md.txt          # Original readme
└── provenance.md.txt      # Data sources and licenses
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.x | Core programming |
| **ML Library** | Scikit-learn 1.3.2 | TF-IDF vectorization, cosine similarity |
| **NLP Library** | NLTK 3.9.1 | Text tokenization and processing |
| **Vectorization** | TF-IDF | Convert text to numerical vectors |
| **Similarity Metric** | Cosine Similarity | Measure query-document relevance |

## 🔬 How It Works

### 1. **Text Vectorization (TF-IDF)**
   - Converts documents into numerical vectors
   - Balances term frequency with document rarity
   - Filters out common words (stop words)

### 2. **Similarity Computation**
   - Compares query vector with all document vectors
   - Uses cosine similarity for relevance scoring
   - Ranks results by similarity score

### 3. **Result Retrieval**
   - Returns top-K most similar paragraphs
   - Displays best match with confidence score
   - Logs interaction for future analysis

## 📊 Use Cases

- 📚 **Knowledge Base Search** - Internal documentation systems
- 🔍 **Offline FAQ** - Customer support without internet
- 🎓 **Educational Tool** - Learn about information retrieval
- 📝 **Document Search** - Find relevant paragraphs in large texts
- 🔒 **Secure Environments** - Air-gapped systems requiring offline operation

## 📈 Performance

- **Query Speed**: < 100ms for small datasets (< 1000 documents)
- **Memory Usage**: Efficient sparse matrix representation
- **Scalability**: Handles thousands of documents efficiently
- **Accuracy**: High relevance for exact term matches

## 🎓 Learning Resources

Want to understand the technology in depth? Check out:

- **[LINKEDIN_DESCRIPTION.md](LINKEDIN_DESCRIPTION.md)** - Professional summary for LinkedIn
- **[TECHNOLOGY_DETAILS.md](TECHNOLOGY_DETAILS.md)** - Deep dive into algorithms and architecture
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - LinkedIn posts, resume tips, interview points

## 🔧 Customization

### Add Your Own Documents

1. Place `.txt` files in the `data/` directory
2. Each file should contain text content (any topic)
3. The chatbot automatically indexes new files on startup

### Adjust Retrieved Results

```python
# In chatbot.py, change top_k parameter
results = retrieve(query, vectorizer, matrix, docs, top_k=5)  # Returns 5 results
```

### Modify Logging Format

```python
# In log_interaction function
log = {
    "timestamp": datetime.datetime.now().isoformat(),
    "query": query,
    "top1": top1,
    # Add custom fields here
}
```

## 🚀 Future Enhancements

Potential improvements:
- [ ] BM25 algorithm for better ranking
- [ ] Sentence transformers for semantic search
- [ ] Web UI with Flask/Streamlit
- [ ] Multi-language support
- [ ] Query expansion and spell correction
- [ ] Real-time document updates
- [ ] Export conversations to PDF/HTML

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new features
- Improve documentation
- Fix bugs
- Suggest enhancements

## 📄 License

This project is open source and available for educational purposes.

## 📞 Contact

**Akash Manda**
- GitHub: [@AkashManda854](https://github.com/AkashManda854)
- Repository: [mini_chatbot](https://github.com/AkashManda854/mini_chatbot)

## 🙏 Acknowledgments

- Data sources documented in `provenance.md.txt`
- Built with Scikit-learn and NLTK
- Inspired by information retrieval systems

## 📚 Documentation Index

- **Quick Start**: See above ⬆️
- **LinkedIn Summary**: [LINKEDIN_DESCRIPTION.md](LINKEDIN_DESCRIPTION.md)
- **Technical Deep Dive**: [TECHNOLOGY_DETAILS.md](TECHNOLOGY_DETAILS.md)
- **Reference Guide**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

**⭐ If you find this project helpful, please give it a star!**

Made with ❤️ for learning NLP and Information Retrieval
