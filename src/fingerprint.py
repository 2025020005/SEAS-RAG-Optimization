import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class HybridFingerprint:
    """
    Core Module 1: Generates Hybrid Semantic Fingerprints.
    Combines dense embeddings (MiniLM) with sparse features (TF-IDF).
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading Encoder: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        # Max features limited to 64 for efficiency
        self.vectorizer = TfidfVectorizer(max_features=64, stop_words='english')
        self.is_fitted = False

    def generate(self, texts, alpha=0.6):
        """
        Generate hybrid vectors. 
        Alpha controls the weight of dense embeddings (0.0 to 1.0).
        """
        # 1. Dense Embeddings
        dense_vecs = self.encoder.encode(texts)
        
        # 2. Sparse Features (TF-IDF)
        if not self.is_fitted:
            sparse_vecs = self.vectorizer.fit_transform(texts).toarray()
            self.is_fitted = True
        else:
            sparse_vecs = self.vectorizer.transform(texts).toarray()
            
        # 3. L2 Normalization
        dense_norm = dense_vecs / np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        # Avoid division by zero for short texts
        sparse_norm = sparse_vecs / (np.linalg.norm(sparse_vecs, axis=1, keepdims=True) + 1e-9)
        
        # 4. Concatenation
        return np.hstack((alpha * dense_norm, (1-alpha) * sparse_norm))