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
        self.term_dictionary = defaultdict(lambda: {"document_frequency": 0, "postings": []})
        self.document_lengths = {}
        self.total_documents = 0  # Total number of documents (N in the formula)
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.document_id_map = {}  # Map filenames to document IDs

    def preprocess_text(self, text):
        """Tokenize, remove stopwords, and stem the text."""
        words = word_tokenize(text.lower())
        return [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stopwords]

    def build_index(self):
        """Build the inverted index and calculate document lengths."""
        for doc_id, filename in enumerate(sorted(os.listdir(self.corpus_path)), start=1):
            if filename.endswith('.txt'):
                self.total_documents += 1
                self.document_id_map[filename] = doc_id
                filepath = os.path.join(self.corpus_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    preprocessed_words = self.preprocess_text(content)
                    
                    term_frequency = defaultdict(int)
                    for word in preprocessed_words:
                        term_frequency[word] += 1
                    
                    document_length = 0
                    for term, freq in term_frequency.items():
                        if doc_id not in [posting[0] for posting in self.term_dictionary[term]["postings"]]:
                            self.term_dictionary[term]["document_frequency"] += 1
                        self.term_dictionary[term]["postings"].append((doc_id, freq))
                        
                        # Calculate log term frequency for document length
                        log_term_freq = 1 + math.log10(freq)
                        document_length += log_term_freq ** 2
                    
                    # Store the square root of the sum of squared log term frequencies
                    self.document_lengths[doc_id] = math.sqrt(document_length)

        # Sort postings by document ID
        for term_info in self.term_dictionary.values():
            term_info["postings"].sort(key=lambda x: x[0])

    def save_index(self):
        """Save the index to files."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Save term dictionary (excluding postings)
        with open(os.path.join(self.index_dir, 'term_dictionary.json'), 'w', encoding='utf-8') as f:
            json.dump({term: {"document_frequency": info["document_frequency"]} 
                       for term, info in self.term_dictionary.items()}, f, ensure_ascii=False)
        
        # Save postings
        with open(os.path.join(self.index_dir, 'postings.txt'), 'w', encoding='utf-8') as f:
            for term, info in self.term_dictionary.items():
                postings_str = ','.join(f'{doc_id}:{freq}' for doc_id, freq in info["postings"])
                f.write(f"{term}:{postings_str}\n")
        
        # Save document lengths
        with open(os.path.join(self.index_dir, 'document_lengths.json'), 'w', encoding='utf-8') as f:
            json.dump(self.document_lengths, f)
        
        # Save total number of documents
        with open(os.path.join(self.index_dir, 'total_documents.txt'), 'w', encoding='utf-8') as f:
            f.write(str(self.total_documents))

        # Save document ID map
        with open(os.path.join(self.index_dir, 'document_id_map.json'), 'w', encoding='utf-8') as f:
            json.dump(self.document_id_map, f, ensure_ascii=False)

    def load_index(self):
        """Load the index from files."""
        # Load term dictionary
        with open(os.path.join(self.index_dir, 'term_dictionary.json'), 'r', encoding='utf-8') as f:
            self.term_dictionary = defaultdict(lambda: {"document_frequency": 0, "postings": []}, json.load(f))
        
        # Load postings
        with open(os.path.join(self.index_dir, 'postings.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                term, postings_str = line.strip().split(':', 1)
                self.term_dictionary[term]["postings"] = [tuple(map(int, posting.split(':'))) for posting in postings_str.split(',')]
        
        # Load document lengths
        with open(os.path.join(self.index_dir, 'document_lengths.json'), 'r', encoding='utf-8') as f:
            self.document_lengths = {int(k): v for k, v in json.load(f).items()}
        
        # Load total number of documents
        with open(os.path.join(self.index_dir, 'total_documents.txt'), 'r', encoding='utf-8') as f:
            self.total_documents = int(f.read().strip())

        # Load document ID map
        with open(os.path.join(self.index_dir, 'document_id_map.json'), 'r', encoding='utf-8') as f:
            self.document_id_map = json.load(f)

    def search(self, query):
        """Perform a search using the vector space model with lnc.ltc scheme."""
        query_terms = self.preprocess_text(query)
        query_vector = defaultdict(float)

        # Calculate query vector weights (ltc scheme)
        for term in set(query_terms):
            if term in self.term_dictionary:
                # l: log term frequency for query
                query_term_freq = 1 + math.log10(query_terms.count(term))
                # t: inverse document frequency
                inverse_doc_freq = math.log10(self.total_documents / self.term_dictionary[term]["document_frequency"])
                # Combine log tf and idf
                query_vector[term] = query_term_freq * inverse_doc_freq

        # c: Cosine normalization for query vector
        query_vector_length = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
        for term in query_vector:
            query_vector[term] /= query_vector_length

        # Calculate document scores
        document_scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_id, term_freq in self.term_dictionary[term]["postings"]:
                # lnc scheme for documents:
                # l: log term frequency
                doc_term_weight = 1 + math.log10(term_freq)
                # n: no idf used for documents
                # c: cosine normalization (divide by document length)
                doc_term_weight /= self.document_lengths[doc_id]
                
                # Accumulate scores
                document_scores[doc_id] += query_weight * doc_term_weight

        # Sort documents by score (descending) and then by document ID (ascending)
        ranked_docs = sorted(document_scores.items(), key=lambda x: (-x[1], x[0]))

        return ranked_docs[:10]  # Return top 10 documents

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
                filename = next(name for name, id in vsm.document_id_map.items() if id == doc_id)
                print(f"{i}. {filename} (Score: {score:.4f})")

if __name__ == "__main__":
    main()