import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../src') # Ensure src is in path

from fingerprint import HybridFingerprint
from bucketing import PrefixBucketing
from fusion import AdaptiveFusion

# Dummy Data Generator
def generate_dummy_data(n=1000):
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",
        "Climate change requires immediate global action.",
        "Python is a great programming language for data science."
    ]
    data = []
    for _ in range(n):
        base = np.random.choice(base_sentences)
        if np.random.random() < 0.5:
            data.append(base) # Duplicate
        else:
            data.append(base + " " + str(np.random.randint(1000))) # Variant
    return data

def main():
    print("=== SEAS Benchmark Start ===")
    
    # 1. Load Data
    data = generate_dummy_data(n=2000)
    print(f"Loaded {len(data)} documents.")
    
    # 2. Init Modules
    fp = HybridFingerprint()
    bk = PrefixBucketing()
    fs = AdaptiveFusion()
    
    start_time = time.time()
    
    # 3. Pipeline Execution
    print("Step 1: Hybrid Fingerprinting...")
    vectors = fp.generate(data)
    
    print("Step 2: Assigning Buckets...")
    bucket_ids = bk.assign(vectors)
    
    print("Step 3: Deduplication...")
    kept_indices = fs.deduplicate(vectors, bucket_ids)
    
    total_time = (time.time() - start_time) * 1000
    
    # 4. Report
    rr = (1 - len(kept_indices)/len(data)) * 100
    print(f"\n=== Results ===")
    print(f"Total Time: {total_time:.2f} ms")
    print(f"Throughput: {len(data)/(total_time/1000):.0f} docs/sec")
    print(f"Redundancy Removal (RR): {rr:.2f}%")
    print("Benchmark Finished.")

if __name__ == "__main__":
    main()