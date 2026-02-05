import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AdaptiveFusion:
    def __init__(self, base_threshold=0.85):
        self.base_threshold = base_threshold

    def compute_threshold(self, doc_len_a, doc_len_b, use_adaptive=True):
        """
        Args:
            use_adaptive: 是否启用动态阈值。False 则返回固定值。
        """
        if not use_adaptive:
            return self.base_threshold

        # 动态逻辑
        avg_len = (doc_len_a + doc_len_b) / 2
        adaptive_offset = 0.05 if avg_len > 300 else -0.03
        return np.clip(self.base_threshold + adaptive_offset, 0.75, 0.98)

    def deduplicate_bucket(self, embeddings, indices, texts, use_adaptive=True):
        if len(indices) < 2: return indices.tolist()
        
        local_embs = embeddings[indices]
        sim_matrix = cosine_similarity(local_embs)
        keep_mask = np.ones(len(indices), dtype=bool)
        
        for i in range(len(indices)):
            if not keep_mask[i]: continue
            for j in range(i + 1, len(indices)):
                if not keep_mask[j]: continue
                
                sim = sim_matrix[i, j]
                # 传入消融开关
                thresh = self.compute_threshold(len(texts[indices[i]]), len(texts[indices[j]]), use_adaptive)
                
                if sim > thresh:
                    keep_mask[j] = False
        
        return indices[keep_mask].tolist()