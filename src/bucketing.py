import numpy as np

class PrefixBucketing:
    """
    Core Module 2: Prefix-based Bucketing using LSH logic.
    Reduces complexity from O(N^2) to O(N).
    """
    def __init__(self, input_dim=448, n_bits=8):
        # 384 (MiniLM) + 64 (TF-IDF) = 448 dimensions
        np.random.seed(42)
        self.projections = np.random.randn(input_dim, n_bits)

    def assign(self, vectors):
        """
        Project high-dim vectors to discrete bucket IDs.
        """
        # Dot product
        hashes = np.dot(vectors, self.projections) > 0
        # Convert boolean array to integer (Pack bits)
        bucket_ids = np.packbits(hashes, axis=1).flatten()
        return bucket_ids