import os
import math
import json
import argparse
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class VectorSpaceModel:
    def __init__(self, corpus_path, index_dir):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.term_dictionary = defaultdict(lambda: {"df": 0, "postings": {}})
        self.document_lengths = {}
        self.total_documents = 0
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.document_id_map = {}

    def preprocess_text(self, text):
        words = word_tokenize(text.lower())
        return [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stopwords]

    def build_index(self):
        for doc_id, filename in enumerate(sorted(os.listdir(self.corpus_path)), start=1):
            if filename.endswith('.txt'):
                self.total_documents += 1
                self.document_id_map[str(doc_id)] = filename
                filepath = os.path.join(self.corpus_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    words = self.preprocess_text(content)
                    
                    term_freq = defaultdict(int)
                    for word in words:
                        term_freq[word] += 1
                    
                    doc_length = 0
                    for term, freq in term_freq.items():
                        if str(doc_id) not in self.term_dictionary[term]["postings"]:
                            self.term_dictionary[term]["df"] += 1
                        self.term_dictionary[term]["postings"][str(doc_id)] = freq
                        
                        log_tf = 1 + math.log10(freq)
                        doc_length += log_tf ** 2
                    
                    self.document_lengths[doc_id] = math.sqrt(doc_length)

    def save_index(self):
        os.makedirs(self.index_dir, exist_ok=True)
        
        with open(os.path.join(self.index_dir, 'term_dictionary.json'), 'w', encoding='utf-8') as f:
            json.dump(self.term_dictionary, f)
        
        with open(os.path.join(self.index_dir, 'document_lengths.json'), 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in self.document_lengths.items()}, f)
        
        with open(os.path.join(self.index_dir, 'total_documents.txt'), 'w', encoding='utf-8') as f:
            f.write(str(self.total_documents))

        with open(os.path.join(self.index_dir, 'document_id_map.json'), 'w', encoding='utf-8') as f:
            json.dump(self.document_id_map, f)

    def load_index(self):
        with open(os.path.join(self.index_dir, 'term_dictionary.json'), 'r', encoding='utf-8') as f:
            self.term_dictionary = json.load(f)
        
        with open(os.path.join(self.index_dir, 'document_lengths.json'), 'r', encoding='utf-8') as f:
            self.document_lengths = {int(k): v for k, v in json.load(f).items()}
        
        with open(os.path.join(self.index_dir, 'total_documents.txt'), 'r', encoding='utf-8') as f:
            self.total_documents = int(f.read().strip())

        with open(os.path.join(self.index_dir, 'document_id_map.json'), 'r', encoding='utf-8') as f:
            self.document_id_map = json.load(f)

    def search(self, query):
        query_terms = self.preprocess_text(query)
        query_vector = defaultdict(float)

        # Calculate query weights (ltc scheme)
        for term in set(query_terms):
            if term in self.term_dictionary:
                tf = 1 + math.log10(query_terms.count(term))
                idf = math.log10(self.total_documents / self.term_dictionary[term]["df"])
                query_vector[term] = tf * idf

        # Normalize query vector
        query_length = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
        for term in query_vector:
            query_vector[term] /= query_length

        # Calculate document scores
        scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_id_str, freq in self.term_dictionary[term]["postings"].items():
                doc_id = int(doc_id_str)
                if doc_id in self.document_lengths:
                    # Document weight (lnc scheme)
                    doc_weight = (1 + math.log10(freq)) / self.document_lengths[doc_id]
                    scores[doc_id] += query_weight * doc_weight

        # Sort documents by score (descending) and then by document ID (ascending)
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        # Map document IDs to filenames and return top 10 results
        return [(self.document_id_map.get(str(doc_id), f"Unknown-{doc_id}"), score) for doc_id, score in ranked_docs[:10]]

def main():
    parser = argparse.ArgumentParser(description="Vector Space Model for Information Retrieval")
    parser.add_argument('--corpus', default='corpus', help='Path to the corpus directory (default: corpus)')
    parser.add_argument('--index', default='index', help='Path to the index directory (default: index)')
    parser.add_argument('--build', action='store_true', help='Build the index')
    parser.add_argument('--search', action='store_true', help='Perform search')
    args = parser.parse_args()

    vsm = VectorSpaceModel(args.corpus, args.index)

    if args.build:
        print("Building index...")
        vsm.build_index()
        vsm.save_index()
        print("Index built and saved successfully.")

    if args.search:
        print("Loading index...")
        vsm.load_index()
        print("Index loaded successfully.")

        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            results = vsm.search(query)
            print("\nTop 10 most relevant documents:")
            for i, (filename, score) in enumerate(results, 1):
                print(f"{i}. {filename} (Score: {score:.4f})")

if __name__ == "__main__":
    main()