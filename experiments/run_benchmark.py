import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans

# Ensure src is visible
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import RealDataLoader
from fingerprint import HybridFingerprint
from bucketing import PrefixBucketing
from fusion import AdaptiveFusion

# Setup Plotting Style for Paper
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'

def calc_irr(kept_indices, true_labels):
    """Calculates Information Retention Rate based on ground-truth labels."""
    kept_labels = set(true_labels[kept_indices])
    total_unique = set(true_labels)
    return len(kept_labels) / len(total_unique) * 100

def run_experiment():
    print("🚀 [Main] Starting SEAS-RAG-Optimization Benchmark...")
    
    # 1. Load Data
    loader = RealDataLoader()
    texts, labels = loader.get_dataset()
    
    # 2. Init Models
    fp = HybridFingerprint()
    bucketer = PrefixBucketing(input_dim=384+64) # MiniLM(384) + TFIDF(64)
    fuser = AdaptiveFusion()
    
    # --- BASELINE 1: SimHash (Approx) ---
    print("\n--- Running Baseline: SimHash ---")
    start = time.time()
    hashes = set()
    kept_sim = []
    for idx, text in enumerate(texts):
        # 3-gram SimHash Simulation
        tokens = [text[i:i+3] for i in range(len(text)-2)]
        if not tokens: tokens = [text]
        h = hash(tuple(sorted(tokens))[:50]) # Simple sorted n-gram hash
        if h not in hashes:
            hashes.add(h)
            kept_sim.append(idx)
    time_sim = (time.time() - start) * 1000
    rr_sim = (1 - len(kept_sim)/len(texts))*100
    irr_sim = calc_irr(kept_sim, labels)
    print(f"SimHash: Time={time_sim:.1f}ms, RR={rr_sim:.1f}%, IRR={irr_sim:.1f}%")

    # --- BASELINE 2: SBERT + KMeans ---
    print("\n--- Running Baseline: SBERT-KMeans ---")
    # Pre-compute embeddings for fair comparison of CLUSTERING time
    print("(Pre-computing embeddings for clustering baselines...)")
    all_vectors = fp.generate(texts) # Note: KMeans usually just uses Dense, but Hybrid is fine too
    
    start = time.time()
    # Assume we know roughly the number of unique events (Best case for KMeans)
    n_clusters = len(set(labels))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512, n_init='auto', random_state=42)
    kmeans.fit(all_vectors)
    # Pick one from each cluster
    _, idxs = np.unique(kmeans.labels_, return_index=True)
    kept_bert = idxs
    time_bert = (time.time() - start) * 1000
    rr_bert = (1 - len(kept_bert)/len(texts))*100
    irr_bert = calc_irr(kept_bert, labels)
    print(f"SBERT-KMeans: Time={time_bert:.1f}ms, RR={rr_bert:.1f}%, IRR={irr_bert:.1f}%")

    # --- OURS: SEAS ---
    print("\n--- Running Model: SEAS ---")
    start = time.time()
    kept_seas = []
    
    # Step A: Bucketing
    bucket_ids = bucketer.assign(all_vectors)
    groups = bucketer.group(np.arange(len(texts)), bucket_ids)
    
    # Step B: Adaptive Fusion
    for b_id, group in groups:
        indices = group['idx'].values
        # Call fusion module
        local_kept = fuser.deduplicate_bucket(all_vectors, indices, texts)
        kept_seas.extend(local_kept)
        
    time_seas = (time.time() - start) * 1000
    rr_seas = (1 - len(kept_seas)/len(texts))*100
    irr_seas = calc_irr(kept_seas, labels)
    print(f"SEAS: Time={time_seas:.1f}ms, RR={rr_seas:.1f}%, IRR={irr_seas:.1f}%")

    # --- Plotting ---
    plot_results(rr_sim, irr_sim, time_sim, 
                 rr_bert, irr_bert, time_bert, 
                 rr_seas, irr_seas, time_seas)

def plot_results(rr1, irr1, t1, rr2, irr2, t2, rr3, irr3, t3):
    print("\n🎨 [Plotting] Generating PDF charts...")
    os.makedirs('../assets', exist_ok=True)
    
    methods = ['SimHash', 'SBERT-KMeans', 'SEAS (Ours)']
    
    # Plot 1: Efficiency (Latency)
    plt.figure(figsize=(6, 4))
    times = [t1, t2, t3]
    plt.plot(methods, times, marker='o', color='#c44e52', linewidth=2)
    plt.ylabel('Latency (ms)')
    plt.title('Processing Efficiency (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(times):
        plt.text(i, v + max(times)*0.05, f"{v:.0f}ms", ha='center')
    plt.savefig('../assets/latency_plot.pdf', bbox_inches='tight')
    
    # Plot 2: Effectiveness (RR & IRR)
    plt.figure(figsize=(6, 4))
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, [rr1, rr2, rr3], width, label='Redundancy Removal', color='#4c72b0')
    plt.bar(x + width/2, [irr1, irr2, irr3], width, label='Info Retention', color='#55a868')
    plt.xticks(x, methods)
    plt.ylabel('Percentage (%)')
    plt.title('Deduplication Effectiveness')
    plt.legend(loc='lower right')
    plt.ylim(0, 110)
    plt.savefig('../assets/accuracy_plot.pdf', bbox_inches='tight')
    
    print("✅ Experiment Complete. Check 'assets/' folder.")

if __name__ == "__main__":
    run_experiment()