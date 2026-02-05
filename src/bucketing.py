import numpy as np
import pandas as pd

class PrefixBucketing:
    """
    Implements O(N) Blocking strategy using LSH-style Random Projection.
    """
    def __init__(self, input_dim, num_buckets=8, seed=42):
        np.random.seed(seed)
        # Random projection matrix to project high-dim vectors to discrete buckets
        self.projections = np.random.randn(input_dim, num_buckets)

    def assign(self, vectors):
        # x . P > 0 -> Boolean Hash
        hashes = np.dot(vectors, self.projections) > 0
        # Convert bool array to integer bucket IDs
        bucket_ids = np.packbits(hashes, axis=1).flatten()
        return bucket_ids

    def group(self, indices, bucket_ids):
        """Groups original indices by bucket ID for processing."""
        df = pd.DataFrame({'idx': indices, 'bucket': bucket_ids})
        return df.groupby('bucket')