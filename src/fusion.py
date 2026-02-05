import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AdaptiveFusion:
    """
    Core Module 3: Adaptive Thresholding for Deduplication.
    """
    def __init__(self, base_threshold=0.90):
        self.base_threshold = base_threshold

    def deduplicate(self, vectors, bucket_ids):
        """
        Perform deduplication within buckets.
        """
        n = len(vectors)
        keep_mask = np.ones(n, dtype=bool)
        
        # Iterate over unique buckets
        unique_buckets = np.unique(bucket_ids)
        
        for b_id in unique_buckets:
            # Get indices in this bucket
            indices = np.where(bucket_ids == b_id)[0]
            if len(indices) < 2:
                continue
                
            # Compute similarity matrix for this bucket
            block_vecs = vectors[indices]
            sim_matrix = cosine_similarity(block_vecs)
            
            # Simple greedy clustering
            # Mark items as removed if they are similar to a previous item
            for i in range(len(indices)):
                if not keep_mask[indices[i]]:
                    continue
                
                # Check similarity with subsequent items
                for j in range(i + 1, len(indices)):
                    if not keep_mask[indices[j]]:
                        continue
                        
                    sim = sim_matrix[i, j]
                    # Dynamic Threshold Logic (Simplified)
                    if sim > self.base_threshold:
                        keep_mask[indices[j]] = False # Drop redundant
                        
        return np.where(keep_mask)[0]