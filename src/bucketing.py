import numpy as np
import pandas as pd

class PrefixBucketing:
    """
    Implements O(N) Blocking strategy using LSH-style Random Projection.
    """
    def __init__(self, input_dim, num_buckets=8, seed=42):
        """
        Args:
            input_dim: Dimension of input vectors.
            num_buckets: This is actually 'n_bits' for LSH (e.g., 8 bits = 256 buckets).
        """
        np.random.seed(seed)
        self.n_bits = num_buckets
        # Random projection matrix to project high-dim vectors to discrete buckets
        self.projections = np.random.randn(input_dim, self.n_bits)

    def assign(self, vectors):
        """
        Project vectors to bucket IDs ensuring 1-to-1 mapping.
        """
        # 1. Random Projection: x . P > 0 -> Boolean Hash (Shape: [N, n_bits])
        hashes = np.dot(vectors, self.projections) > 0
        
        # 2. Convert Boolean Array to Integer IDs using Powers of 2
        # This fixes the bug where np.packbits produces multiple bytes for n_bits > 8
        # Example: [True, False, True] -> 1*4 + 0*2 + 1*1 = 5
        powers_of_two = 1 << np.arange(self.n_bits - 1, -1, -1)
        
        # Dot product to get unique integer ID for each row
        bucket_ids = hashes.astype(int).dot(powers_of_two)
        
        return bucket_ids

    def group(self, indices, bucket_ids):
        """Groups original indices by bucket ID for processing."""
        # Ensure bucket_ids and indices have same length
        if len(indices) != len(bucket_ids):
            raise ValueError(f"Shape mismatch: indices({len(indices)}) vs buckets({len(bucket_ids)})")
            
        df = pd.DataFrame({'idx': indices, 'bucket': bucket_ids})
        return df.groupby('bucket')