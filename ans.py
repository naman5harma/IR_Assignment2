import math
from collections import defaultdict

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = {}
        self.postings = defaultdict(list)
        self.doc_lengths = {}

    def add_document(self, doc_id, text):
        terms = text.lower().split()
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1

        doc_length = len(terms)
        self.doc_lengths[doc_id] = doc_length

        for term, freq in term_freq.items():
            if term not in self.dictionary:
                self.dictionary[term] = len(self.dictionary)
            self.postings[term].append((doc_id, freq))

    def compute_tfidf(self, term, doc_id, freq):
        tf = 1 + math.log10(freq)
        idf = math.log10(len(self.doc_lengths) / len(self.postings[term]))
        return tf * idf

    def search(self, query):
        query_terms = query.lower().split()
        query_vector = defaultdict(float)
        for term in query_terms:
            if term in self.dictionary:
                query_vector[term] = self.compute_tfidf(term, 'query', 1)

        scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_id, freq in self.postings[term]:
                doc_weight = self.compute_tfidf(term, doc_id, freq)
                scores[doc_id] += query_weight * doc_weight

        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

def read_corpus(file_path):
    documents = {}
    with open(file_path, 'r') as file:
        for line in file:
            doc_id, content = line.strip().split('\t', 1)
            documents[int(doc_id)] = content
    return documents

# Main execution
if __name__ == "__main__":
    corpus_path = "path_to_your_corpus_file.txt"  # Replace with actual path
    vsm = VectorSpaceModel()

    # Read and add documents from corpus
    corpus = read_corpus(corpus_path)
    for doc_id, content in corpus.items():
        vsm.add_document(doc_id, content)

    # Perform a search
    query = input("Enter your search query: ")
    results = vsm.search(query)

    print("Search results:")
    for doc_id, score in results:
        print(f"Document {doc_id}: Score {score}")
        print(f"Content: {corpus[doc_id][:100]}...")  # Print first 100 chars of the document
        print()import math
from collections import defaultdict

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = {}
        self.postings = defaultdict(list)
        self.doc_lengths = {}

    def add_document(self, doc_id, text):
        terms = text.lower().split()
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1

        doc_length = len(terms)
        self.doc_lengths[doc_id] = doc_length

        for term, freq in term_freq.items():
            if term not in self.dictionary:
                self.dictionary[term] = len(self.dictionary)
            self.postings[term].append((doc_id, freq))

    def compute_tfidf(self, term, doc_id, freq):
        tf = 1 + math.log10(freq)
        idf = math.log10(len(self.doc_lengths) / len(self.postings[term]))
        return tf * idf

    def search(self, query):
        query_terms = query.lower().split()
        query_vector = defaultdict(float)
        for term in query_terms:
            if term in self.dictionary:
                query_vector[term] = self.compute_tfidf(term, 'query', 1)

        scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_id, freq in self.postings[term]:
                doc_weight = self.compute_tfidf(term, doc_id, freq)
                scores[doc_id] += query_weight * doc_weight

        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

def read_corpus(file_path):
    documents = {}
    with open(file_path, 'r') as file:
        for line in file:
            doc_id, content = line.strip().split('\t', 1)
            documents[int(doc_id)] = content
    return documents

# Main execution
if __name__ == "__main__":
    corpus_path = "path_to_your_corpus_file.txt"  # Replace with actual path
    vsm = VectorSpaceModel()

    # Read and add documents from corpus
    corpus = read_corpus(corpus_path)
    for doc_id, content in corpus.items():
        vsm.add_document(doc_id, content)

    # Perform a search
    query = input("Enter your search query: ")
    results = vsm.search(query)

    print("Search results:")
    for doc_id, score in results:
        print(f"Document {doc_id}: Score {score}")
        print(f"Content: {corpus[doc_id][:100]}...")  # Print first 100 chars of the document
        print()