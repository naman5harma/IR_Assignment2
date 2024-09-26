import os
import math
import json
import argparse
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class VectorSpaceModel:
    def __init__(self, corpus_path, index_dir):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.dictionary = defaultdict(lambda: {"df": 0, "postings": []})
        self.doc_lengths = {}
        self.N = 0  # Total number of documents
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.doc_id_map = {}  # Map filenames to document IDs

    def preprocess(self, text):
        """Tokenize, remove stopwords, and stem the text."""
        tokens = word_tokenize(text.lower())
        return [self.stemmer.stem(token) for token in tokens if token.isalnum() and token not in self.stopwords]

    def build_index(self):
        """Build the inverted index and calculate document lengths."""
        for doc_id, filename in enumerate(sorted(os.listdir(self.corpus_path)), start=1):
            if filename.endswith('.txt'):
                self.N += 1
                self.doc_id_map[filename] = doc_id
                filepath = os.path.join(self.corpus_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    tokens = self.preprocess(content)
                    
                    term_freq = defaultdict(int)
                    for token in tokens:
                        term_freq[token] += 1
                    
                    doc_length = 0
                    for term, freq in term_freq.items():
                        if doc_id not in [posting[0] for posting in self.dictionary[term]["postings"]]:
                            self.dictionary[term]["df"] += 1
                        self.dictionary[term]["postings"].append((doc_id, freq))
                        
                        log_tf = 1 + math.log10(freq)
                        doc_length += log_tf ** 2
                    
                    self.doc_lengths[doc_id] = math.sqrt(doc_length)

        for term_info in self.dictionary.values():
            term_info["postings"].sort(key=lambda x: x[0])

    def save_index(self):
        """Save the index to files."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        with open(os.path.join(self.index_dir, 'dictionary.json'), 'w', encoding='utf-8') as f:
            json.dump({term: {"df": info["df"]} for term, info in self.dictionary.items()}, f, ensure_ascii=False)
        
        with open(os.path.join(self.index_dir, 'postings.txt'), 'w', encoding='utf-8') as f:
            for term, info in self.dictionary.items():
                postings_str = ','.join(f'{doc_id}:{freq}' for doc_id, freq in info['postings'])
                f.write(f"{term}:{postings_str}\n")
        
        with open(os.path.join(self.index_dir, 'doc_lengths.json'), 'w', encoding='utf-8') as f:
            json.dump(self.doc_lengths, f)
        
        with open(os.path.join(self.index_dir, 'N.txt'), 'w', encoding='utf-8') as f:
            f.write(str(self.N))

        with open(os.path.join(self.index_dir, 'doc_id_map.json'), 'w', encoding='utf-8') as f:
            json.dump(self.doc_id_map, f, ensure_ascii=False)


    def load_index(self):
        """Load the index from files."""
        with open(os.path.join(self.index_dir, 'dictionary.json'), 'r', encoding='utf-8') as f:
            self.dictionary = defaultdict(lambda: {"df": 0, "postings": []}, json.load(f))
        
        with open(os.path.join(self.index_dir, 'postings.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                term, postings_str = line.strip().split(':', 1)
                self.dictionary[term]["postings"] = [tuple(map(int, posting.split(':'))) for posting in postings_str.split(',')]
        
        with open(os.path.join(self.index_dir, 'doc_lengths.json'), 'r', encoding='utf-8') as f:
            self.doc_lengths = {int(k): v for k, v in json.load(f).items()}
        
        with open(os.path.join(self.index_dir, 'N.txt'), 'r', encoding='utf-8') as f:
            self.N = int(f.read().strip())

        with open(os.path.join(self.index_dir, 'doc_id_map.json'), 'r', encoding='utf-8') as f:
            self.doc_id_map = json.load(f)


    def search(self, query):
        """Perform a search using the vector space model."""
        query_tokens = self.preprocess(query)
        query_vector = defaultdict(float)

        for token in set(query_tokens):
            if token in self.dictionary:
                tf = 1 + math.log10(query_tokens.count(token))
                idf = math.log10(self.N / self.dictionary[token]["df"])
                query_vector[token] = tf * idf

        query_length = math.sqrt(sum(weight**2 for weight in query_vector.values()))
        for term in query_vector:
            query_vector[term] /= query_length

        scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_id, freq in self.dictionary[term]["postings"]:
                doc_weight = (1 + math.log10(freq)) / self.doc_lengths[doc_id]
                scores[doc_id] += query_weight * doc_weight

        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        return ranked_docs[:10]

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
            for i, (doc_id, score) in enumerate(results, 1):
                filename = next(name for name, id in vsm.doc_id_map.items() if id == doc_id)
                print(f"{i}. {filename} (Score: {score:.4f})")

if __name__ == "__main__":
    main()