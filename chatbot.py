import os
import argparse
import json
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt', quiet=True)

def load_data(data_dir):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split into paragraphs
                paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
                for para in paragraphs:
                    docs.append({"file": fname, "para": para})
    return docs

def build_index(docs):
    corpus = [d["para"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

def retrieve(query, vectorizer, matrix, docs, top_k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    results = [{"file": docs[i]["file"], "para": docs[i]["para"], "score": float(sims[i])} for i in top_idx]
    return results

def log_interaction(query, top1, top3):
    os.makedirs("outputs", exist_ok=True)
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "top1": top1,
        "top3": top3
    }
    with open("outputs/session.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

def main(data_dir):
    docs = load_data(data_dir)
    vectorizer, matrix = build_index(docs)
    print("\n🤖 Offline Chatbot ready! Type 'exit' to quit.\n")
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break
        
        results = retrieve(query, vectorizer, matrix, docs, top_k=3)
        top1 = results[0]
        print(f"\n📄 From file: {top1['file']}")
        print(f"💬 Best match (score={top1['score']:.2f}):\n{top1['para']}\n")
        
        log_interaction(query, top1, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to text data folder")
    args = parser.parse_args()
    main(args.data_dir)
