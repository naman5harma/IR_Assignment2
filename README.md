# IR_Assignment2

# Enhanced Vector Space Model for Information Retrieval

This project implements an Enhanced Vector Space Model (VSM) for information retrieval using the lnc.ltc weighting scheme. It provides both a Python script (.py) and a Jupyter Notebook (.ipynb) for flexibility in usage.

## Requirements

- Python 3.6+
- NLTK
- Jupyter Notebook (for .ipynb version)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/naman5harma/IR_ASSIGNMENT2.git
   cd IR_ASSIGNMENT2
   ```

2. Install the required packages:

   ```
   pip install nltk jupyter
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### Running the Python Script (.py)

1. Ensure your corpus (text documents) is in a folder named 'corpus' in the same directory as the script.

2. To build the index:

   ```
   python Main_Ans.py --build
   ```

3. To search:

   ```
   python Main_Ans.py --search
   ```

4. Follow the prompts to enter search queries.

### Using the Jupyter Notebook (.ipynb)

1. Start Jupyter Notebook:

   ```
   jupyter notebook
   ```

2. Open the `Main_Ans.ipynb` file.

3. Run the cells in order:
   - The first few cells will set up the environment and define the `EnhancedVSM` class.
   - Run the cell to build the index (only needed once).
   - Use the search cells to perform queries.

## Implementation Details

This implementation uses the Vector Space Model with the lnc.ltc weighting scheme:

- lnc for documents: log term frequency, no IDF, cosine normalization
- ltc for queries: log term frequency, IDF, cosine normalization

Key features:

- Lemmatization for term normalization
- Stopword removal
- Cosine similarity for ranking documents

## Calculations

1. Term Frequency (TF):

   ```
   TF = 1 + log10(raw frequency)
   ```

   This logarithmic scaling reduces the impact of high-frequency terms.

2. Inverse Document Frequency (IDF):

   ```
   IDF = log10(N / df)
   ```

   Where N is the total number of documents and df is the document frequency of the term.
   IDF gives higher weight to terms that appear in fewer documents.

3. Document Vector (lnc scheme):

   - Weight = TF (no IDF used for documents)
   - Normalization: Each term weight is divided by the document vector's magnitude:
     ```
     normalized_weight = weight / sqrt(sum(weight^2 for all terms))
     ```

4. Query Vector (ltc scheme):

   - Weight = TF \* IDF
   - Normalization: Same as document vector

5. Similarity Calculation:
   Cosine similarity between the normalized query vector and each normalized document vector:
   ```
   similarity = sum(query_weight * doc_weight for each term) / (query_magnitude * doc_magnitude)
   ```
   Since vectors are normalized, this simplifies to the dot product of the vectors.

## How It Works

1. Preprocessing:

   - Tokenization: Breaks text into individual words.
   - Lowercasing: Converts all text to lowercase.
   - Stopword Removal: Eliminates common words that don't carry much meaning.
   - Lemmatization: Reduces words to their base or dictionary form.

2. Indexing:

   - Creates an inverted index mapping terms to documents.
   - Calculates and stores term frequencies and document frequencies.
   - Computes document vector magnitudes for later normalization.

3. Querying:

   - Preprocesses the query text.
   - Constructs a query vector using the ltc scheme.
   - Computes similarities with all documents in the corpus.
   - Ranks documents based on similarity scores.

4. Result Presentation:
   - Returns the top 10 most relevant documents with their similarity scores.

## File Structure

```
IR_ASSIGNMENT2/
│
├── Main_Ans.py        # Python script version
├── Main_Ans.ipynb     # Jupyter Notebook version
├── README.md              # This file
└── corpus/                # Folder containing your text documents
    ├── doc1.txt
    ├── doc2.txt
    └── ...
```

## Notes

- The lnc.ltc scheme provides a balanced approach to term weighting, suitable for general text retrieval tasks.
- Lemmatization is used instead of stemming for potentially better semantic preservation.
- The model assumes that the relevance of a document increases with the number of query terms it contains, weighted by their importance.

#For any issues or suggestions, please open an issue in the GitHub repository.
