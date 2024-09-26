import os
import math
import json
import argparse
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class EnhancedVSM:
    """
    Enhanced Vector Space Model for Information Retrieval
    
    This class implements a vector space model using the lnc.ltc weighting scheme:
    - lnc for documents: log term frequency, no IDF, cosine normalization
    - ltc for queries: log term frequency, IDF, cosine normalization
    
    The vector space model represents documents and queries as vectors in a high-dimensional space,
    where each dimension corresponds to a term in the corpus. Similarity between a query and a document
    is computed using the cosine similarity of their respective vectors.
    """

    def __init__(self, corpus_path, index_dir):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.term_index = defaultdict(lambda: {"doc_freq": 0, "occurrences": {}})
        self.doc_magnitudes = {}
        self.corpus_size = 0
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.doc_id_lookup = {}

    def preprocess_text(self, text):
        """
        Preprocess the input text: lowercase, remove punctuation, tokenize, lemmatize, and remove stop words.
        This step is crucial for both indexing and querying to ensure consistent term representation.
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize the text
        tokens = word_tokenize(text)
        # Lemmatize and remove stop words
        return [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]

    def construct_index(self):
        """
        Construct the inverted index and calculate document magnitudes.
        The inverted index maps each term to the documents it appears in, along with term frequencies.
        Document magnitudes are used for cosine normalization during retrieval.
        """
        for doc_id, filename in enumerate(sorted(os.listdir(self.corpus_path)), start=1):
            if filename.endswith('.txt'):
                self.corpus_size += 1
                self.doc_id_lookup[str(doc_id)] = filename
                filepath = os.path.join(self.corpus_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    tokens = self.preprocess_text(content)
                    
                    if not tokens:
                        continue  # Skip empty documents
                    
                    term_freq = defaultdict(int)
                    for token in tokens:
                        term_freq[token] += 1
                    
                    doc_magnitude = 0
                    for term, freq in term_freq.items():
                        if str(doc_id) not in self.term_index[term]["occurrences"]:
                            self.term_index[term]["doc_freq"] += 1
                        self.term_index[term]["occurrences"][str(doc_id)] = freq
                        
                        # Calculate log term frequency for document vector
                        log_tf = 1 + math.log10(freq)
                        doc_magnitude += log_tf ** 2
                    
                    # Store the magnitude (length) of the document vector for later normalization
                    self.doc_magnitudes[doc_id] = math.sqrt(doc_magnitude)

    def persist_index(self):
        """Save the constructed index and related data to files for later use."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        with open(os.path.join(self.index_dir, 'term_index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.term_index, f)
        
        with open(os.path.join(self.index_dir, 'doc_magnitudes.json'), 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in self.doc_magnitudes.items()}, f)
        
        with open(os.path.join(self.index_dir, 'corpus_size.txt'), 'w', encoding='utf-8') as f:
            f.write(str(self.corpus_size))

        with open(os.path.join(self.index_dir, 'doc_id_lookup.json'), 'w', encoding='utf-8') as f:
            json.dump(self.doc_id_lookup, f)

    def load_index(self):
        """Load the previously constructed index and related data from files."""
        with open(os.path.join(self.index_dir, 'term_index.json'), 'r', encoding='utf-8') as f:
            self.term_index = json.load(f)
        
        with open(os.path.join(self.index_dir, 'doc_magnitudes.json'), 'r', encoding='utf-8') as f:
            self.doc_magnitudes = {int(k): v for k, v in json.load(f).items()}
        
        with open(os.path.join(self.index_dir, 'corpus_size.txt'), 'r', encoding='utf-8') as f:
            self.corpus_size = int(f.read().strip())

        with open(os.path.join(self.index_dir, 'doc_id_lookup.json'), 'r', encoding='utf-8') as f:
            self.doc_id_lookup = json.load(f)

    def execute_query(self, query):
        """
        Execute a search query using the vector space model.
        
        This method implements the core of the vector space model:
        1. Preprocess the query
        2. Compute the query vector using the ltc scheme
        3. Compute document vectors using the lnc scheme
        4. Calculate cosine similarity between the query vector and document vectors
        5. Rank documents based on their similarity scores
        """
        query_terms = self.preprocess_text(query)
        query_vector = defaultdict(float)
        doc_vectors = defaultdict(lambda: defaultdict(float))

        # Compute query vector weights (ltc scheme)
        for term in set(query_terms):
            if term in self.term_index:
                # l: log term frequency for query
                tf = 1 + math.log10(query_terms.count(term))
                # t: inverse document frequency
                idf = math.log10(self.corpus_size / self.term_index[term]["doc_freq"])
                # Combine log tf and idf
                query_vector[term] = tf * idf

        # c: Cosine normalization for query vector
        query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
        if query_magnitude == 0:
            return []  # No valid terms in query
        for term in query_vector:
            query_vector[term] /= query_magnitude

        # Compute document vector weights (lnc scheme)
        for term in query_vector:
            for doc_id_str, freq in self.term_index[term]["occurrences"].items():
                doc_id = int(doc_id_str)
                if doc_id in self.doc_magnitudes:
                    # l: log term frequency for documents
                    doc_vectors[doc_id][term] = 1 + math.log10(freq)

        # c: Cosine normalization for document vectors
        for doc_id, vector in doc_vectors.items():
            magnitude = self.doc_magnitudes[doc_id]
            for term in vector:
                vector[term] /= magnitude

        # Compute cosine similarity between query and documents
        similarity_scores = {}
        for doc_id, doc_vector in doc_vectors.items():
            # Dot product of query and document vectors
            score = sum(query_vector[term] * doc_vector[term] for term in query_vector if term in doc_vector)
            similarity_scores[doc_id] = score

        # Rank documents by similarity score (descending) and then by document ID (ascending)
        ranked_docs = sorted(similarity_scores.items(), key=lambda x: (-x[1], x[0]))

        # Return top 10 results with document names and scores
        return [(self.doc_id_lookup.get(str(doc_id), f"Unknown-{doc_id}"), score) for doc_id, score in ranked_docs[:10]]

def main():
    parser = argparse.ArgumentParser(description="Enhanced Vector Space Model for Information Retrieval")
    parser.add_argument('--corpus', default='corpus', help='Path to the corpus directory (default: corpus)')
    parser.add_argument('--index', default='index', help='Path to the index directory (default: index)')
    parser.add_argument('--build', action='store_true', help='Build the index')
    parser.add_argument('--search', action='store_true', help='Perform search')
    args = parser.parse_args()

    vsm = EnhancedVSM(args.corpus, args.index)

    if args.build:
        print("Constructing index...")
        vsm.construct_index()
        vsm.persist_index()
        print("Index constructed and saved successfully.")

    if args.search:
        print("Loading index...")
        vsm.load_index()
        print("Index loaded successfully.")

        while True:
            query = input("\nEnter your search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            results = vsm.execute_query(query)
            if results:
                print("\nTop 10 most relevant documents:")
                for i, (filename, score) in enumerate(results, 1):
                    print(f"{i}. {filename} (Similarity: {score:.4f})")
            else:
                print("No relevant documents found.")

if __name__ == "__main__":
    main()