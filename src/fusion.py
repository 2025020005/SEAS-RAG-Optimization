import numpy as np

class AdaptiveFusion:
    """
    Implements Confidence-Aware Dynamic Thresholding.
    Optimized with Vectorized Operations for extreme speed.
    """
    def __init__(self, base_threshold=0.60):
        self.base_threshold = base_threshold

    def deduplicate_bucket(self, embeddings, indices, texts, use_adaptive=True):
        if len(indices) < 2: 
            return indices.tolist()
            
        local_embs = embeddings[indices]
        sim_matrix = np.dot(local_embs, local_embs.T) 
        
        keep_mask = np.ones(len(indices), dtype=bool)
        lengths = np.array([len(texts[i]) for i in indices])
        
        for i in range(len(indices)):
            if not keep_mask[i]: continue
            
            j_idx = np.arange(i + 1, len(indices))
            valid_j = j_idx[keep_mask[j_idx]]
            
            if len(valid_j) == 0:
                continue
                
            sims = sim_matrix[i, valid_j]
            
            if use_adaptive:
                # 向量化计算动态阈值
                avg_lens = (lengths[i] + lengths[valid_j]) / 2
                adaptive_offsets = np.where(avg_lens > 300, 0.05, -0.03)
                thresh = np.clip(self.base_threshold + adaptive_offsets, 0.50, 0.98) 
            else:
                thresh = self.base_threshold
                
            # 瞬间找出并标记所有冗余项
            dupes = valid_j[sims > thresh]
            keep_mask[dupes] = False
            
        return indices[keep_mask].tolist()