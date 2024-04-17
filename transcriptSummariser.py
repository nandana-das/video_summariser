import nltk
import numpy as np
import os
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def read_transcript(transcript_file_path):
    with open(transcript_file_path, "r") as f:
        return f.read()

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
        preprocessed_sentences.append(filtered_words)
    return preprocessed_sentences

def sentence_similarity(sent1, sent2):
    # Compute cosine similarity between two sentences
    vector1 = np.mean([word_embeddings.get(word, np.zeros((100,))) for word in sent1], axis=0)
    vector2 = np.mean([word_embeddings.get(word, np.zeros((100,))) for word in sent2], axis=0)
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):
    # Build a similarity matrix between sentences
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    return similarity_matrix

def generate_summary(text, top_n=3):
    # Preprocess the text
    preprocessed_sentences = preprocess_text(text)
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(preprocessed_sentences)
    # Rank sentences using PageRank algorithm
    scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))
    # Sort the sentences based on their scores
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(preprocessed_sentences)), reverse=True)
    # Select top sentences for summary
    summary = [sentence for _, sentence in ranked_sentences[:top_n]]
    return summary

# Load pre-trained word embeddings (e.g., Word2Vec)
# Replace this with your own word embeddings model
word_embeddings = {}  # Load your pre-trained word embeddings model here

# Read transcript
transcript = read_transcript(transcript_file_path)
# Generate summary
summary_sentences = generate_summary(transcript)

# Save summary to a text file
summary_file_path = r"C:\Users\user\OneDrive\Desktop\Nandana\btech\s6 ai\project\mini-project\summaries\summary4.txt"
with open(summary_file_path, "w") as f:
    for sentence in summary_sentences:
        f.write(' '.join(sentence) + '\n')

print("Summary saved successfully.")
