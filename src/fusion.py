import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AdaptiveFusion:
    """
    Implements Confidence-Aware Dynamic Thresholding.
    """
    def __init__(self, base_threshold=0.85):
        self.base_threshold = base_threshold

    def compute_threshold(self, doc_len_a, doc_len_b):
        """
        Dynamically adjusts threshold.
        Hypothesis: Longer documents contain more noise/detail, requiring stricter (higher) similarity.
        Short documents (snippets) can be merged more aggressively (lower threshold).
        """
        # Simple heuristic: Use length as a proxy for information density/confidence
        avg_len = (doc_len_a + doc_len_b) / 2
        
        # If texts are long (>300 chars), raise threshold to 0.90
        # If texts are short, lower to 0.82
        adaptive_offset = 0.05 if avg_len > 300 else -0.03
        
        return np.clip(self.base_threshold + adaptive_offset, 0.75, 0.98)

    def deduplicate_bucket(self, embeddings, indices, texts):
        """
        Runs O(B^2) deduplication only within a small bucket.
        """
        if len(indices) < 2:
            return indices.tolist()

        local_embs = embeddings[indices]
        sim_matrix = cosine_similarity(local_embs)
        
        keep_mask = np.ones(len(indices), dtype=bool)
        
        # Greedy clustering within bucket
        for i in range(len(indices)):
            if not keep_mask[i]: continue
            
            for j in range(i + 1, len(indices)):
                if not keep_mask[j]: continue
                
                sim = sim_matrix[i, j]
                # Dynamic Threshold Check
                thresh = self.compute_threshold(len(texts[indices[i]]), len(texts[indices[j]]))
                
                if sim > thresh:
                    keep_mask[j] = False # Mark as duplicate
        
        return indices[keep_mask].tolist()