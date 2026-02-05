import numpy as np
from sklearn.datasets import fetch_20newsgroups

class RealDataLoader:
    """
    Loads real-world data from 20 Newsgroups and constructs a semantic deduplication dataset.
    """
    def __init__(self, categories=['sci.space', 'comp.graphics', 'rec.autos']):
        print(f"📥 [Data] Downloading 20 Newsgroups dataset (Categories: {categories})...")
        self.raw_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
        self.documents = [doc for doc in self.raw_data.data if len(doc) > 100] # Filter very short texts
        print(f"✅ [Data] Loaded {len(self.documents)} unique base documents.")

    def get_dataset(self):
        """
        Constructs a dataset with ground-truth labels for deduplication.
        Simulates RAG retrieval noise:
        1. Originals (Label = ID)
        2. Semantic Fragments (Label = ID) - Simulating chunking overlap
        3. Quotes/Reposts (Label = ID) - Simulating forum replies
        """
        dataset = []
        labels = []
        
        print("🔄 [Data] Synthesizing semantic variants for stress testing...")
        for idx, doc in enumerate(self.documents):
            # 1. Original Document
            dataset.append(doc)
            labels.append(idx)
            
            # 2. Semantic Fragment (First 60% of text) -> Simulates overlapping chunks
            if len(doc) > 200:
                fragment = doc[:int(len(doc)*0.6)]
                dataset.append(fragment)
                labels.append(idx)
            
            # 3. Quoted/Contextual Variant -> Simulates distinct formulation of same content
            # (In a real scenario, this would be a paraphrase. Here we simulate structure variation)
            variant = f"Reference to previous post: {doc[:150]}... [End Quote]"
            dataset.append(variant)
            labels.append(idx)
            
        print(f"📊 [Data] Final Dataset Size: {len(dataset)} items.")
        return dataset, np.array(labels)