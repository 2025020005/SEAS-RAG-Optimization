import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridFingerprint:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"🤖 [Model] Loading SBERT: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer(max_features=64, stop_words='english')
        self.is_fitted = False

    def generate(self, texts, alpha=0.5):
        """
        Args:
            alpha: 权重系数。
                   alpha=1.0 -> 仅 Dense (w/o Hybrid)
                   alpha=0.5 -> 混合 (Full Model)
        """
        # 1. Dense Semantic Features
        dense = self.encoder.encode(texts, batch_size=128, show_progress_bar=False)
        dense_norm = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)

        # 2. Sparse Lexical Features
        if not self.is_fitted:
            sparse = self.vectorizer.fit_transform(texts).toarray()
            self.is_fitted = True
        else:
            sparse = self.vectorizer.transform(texts).toarray()
        sparse_norm = sparse / (np.linalg.norm(sparse, axis=1, keepdims=True) + 1e-9)

        # 3. Weighted Fusion
        # 如果 alpha=1.0，稀疏部分系数为0，相当于没有
        return np.hstack((alpha * dense_norm, (1 - alpha) * sparse_norm))