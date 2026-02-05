import numpy as np
import pandas as pd

class PrefixClustering:
    """
    Implements the Online Bucketing strategy (Section 2.2 of the paper).
    Uses LSH-like random projection to group semantically similar vectors 
    into coarse-grained buckets for O(N) complexity.
    """
    
    def __init__(self, input_dim=448, n_bits=8, seed=42):
        """
        Initialize the clustering module.
        
        Args:
            input_dim (int): Dimension of input vectors (Default: 384 MiniLM + 64 TF-IDF).
            n_bits (int): Number of bits for the hash (controls number of buckets).
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        # Random projection matrix for Locality Sensitive Hashing (LSH)
        self.projections = np.random.randn(input_dim, n_bits)

    def assign_buckets(self, vectors):
        """
        Project high-dimensional vectors to discrete bucket IDs.
        
        Args:
            vectors (np.ndarray): Input feature vectors.
            
        Returns:
            np.ndarray: Array of bucket IDs (integers).
        """
        # 1. Random Projection: x . P
        projections = np.dot(vectors, self.projections)
        
        # 2. Binarize: x > 0 -> 1, x <= 0 -> 0
        bool_hashes = projections > 0
        
        # 3. Pack bits to integer (e.g., [True, False, True] -> 5)
        # This converts the binary hash code into a unique Bucket ID
        bucket_ids = np.packbits(bool_hashes, axis=1).flatten()
        
        return bucket_ids

    def get_groups(self, dataset, bucket_ids):
        """
        Helper function to group the dataset by bucket IDs.
        """
        df = pd.DataFrame({
            'text': dataset, 
            'bucket': bucket_ids, 
            'original_idx': range(len(dataset))
        })
        return df.groupby('bucket')