import math
from collections import defaultdict

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = {}  # {term: doc_freq}
        self.postings = defaultdict(list)  # {term: [(doc_id, term_freq), ...]}
        self.doc_lengths = {}  # {doc_id: length}

    def add_document(self, doc_id, text):
        terms = text.lower().split()
        self.doc_lengths[doc_id] = len(terms)
        
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
        
        for term, freq in term_freq.items():
            if term not in self.dictionary:
                self.dictionary[term] = 0
            self.dictionary[term] += 1
            self.postings[term].append((doc_id, freq))

    def compute_tfidf(self, term, doc_id, freq):
        tf = 1 + math.log10(freq)
        idf = math.log10(len(self.doc_lengths) / self.dictionary[term])
        return tf * idf

    def search(self, query):
        query_terms = query.lower().split()
        scores = defaultdict(float)
        
        for term in query_terms:
            if term in self.dictionary:
                idf = math.log10(len(self.doc_lengths) / self.dictionary[term])
                for doc_id, freq in self.postings[term]:
                    tf = 1 + math.log10(freq)
                    tfidf = tf * idf
                    scores[doc_id] += tfidf

        # Normalize scores by document length
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