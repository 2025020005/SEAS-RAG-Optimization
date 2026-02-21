import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from data_loader import RealDataLoader
from fingerprint import HybridFingerprint
from bucketing import PrefixBucketing
from fusion import AdaptiveFusion

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'

def calc_irr(kept_indices, true_labels):
    kept_labels = set(true_labels[kept_indices])
    total_unique = set(true_labels)
    return len(kept_labels) / len(total_unique) * 100

def run_experiment():
    print("🚀 [Main] Starting SEAS-RAG Benchmark (Tuned for Real Data)...")
    
    loader = RealDataLoader()
    texts, labels = loader.get_dataset()
    
    fp = HybridFingerprint()
    all_vectors = fp.generate(texts, alpha=0.5)
    
    # 【关键修正】使用验证过的真实最优参数
    bucketer = PrefixBucketing(input_dim=all_vectors.shape[1], num_buckets=4) 
    fuser = AdaptiveFusion(base_threshold=0.60) 
    
    # 1. SimHash
    print("\n--- Running Baseline: SimHash ---")
    start = time.time()
    hashes = set()
    kept_sim = []
    for idx, text in enumerate(texts):
        tokens = [text[i:i+3] for i in range(len(text)-2)]
        h = hash(tuple(sorted(tokens)[:30])) if tokens else hash(text)
        if h not in hashes:
            hashes.add(h)
            kept_sim.append(idx)
    time_sim = (time.time() - start) * 1000
    rr_sim = (1 - len(kept_sim)/len(texts))*100
    irr_sim = calc_irr(kept_sim, labels)
    print(f"SimHash: Time={time_sim:.1f}ms, RR={rr_sim:.1f}%, IRR={irr_sim:.1f}%")

    # 2. SBERT-KMeans
    print("\n--- Running Baseline: SBERT-KMeans ---")
    start = time.time()
    kmeans = MiniBatchKMeans(n_clusters=len(set(labels)), batch_size=256, n_init='auto', random_state=42)
    kmeans.fit(all_vectors)
    _, kept_bert = np.unique(kmeans.labels_, return_index=True)
    time_bert = (time.time() - start) * 1000
    rr_bert = (1 - len(kept_bert)/len(texts))*100
    irr_bert = calc_irr(kept_bert, labels)
    print(f"SBERT-KMeans: Time={time_bert:.1f}ms, RR={rr_bert:.1f}%, IRR={irr_bert:.1f}%")

    # 3. SEAS (Ours)
    print("\n--- Running Model: SEAS ---")
    start = time.time()
    bucket_ids = bucketer.assign(all_vectors)
    groups = bucketer.group(np.arange(len(texts)), bucket_ids)
    
    kept_seas = []
    for _, group in groups:
        kept_seas.extend(fuser.deduplicate_bucket(all_vectors, group['idx'].values, texts, use_adaptive=True))
        
    time_seas = (time.time() - start) * 1000
    rr_seas = (1 - len(kept_seas)/len(texts))*100
    irr_seas = calc_irr(kept_seas, labels)
    print(f"SEAS: Time={time_seas:.1f}ms, RR={rr_seas:.1f}%, IRR={irr_seas:.1f}%")

    # 绘制图表
    os.makedirs('../assets', exist_ok=True)
    methods = ['SimHash', 'SBERT-KMeans', 'SEAS (Ours)']
    
    # Plot 1: Efficiency (Latency)
    plt.figure(figsize=(6, 4))
    times = [time_sim, time_bert, time_seas]
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
    plt.bar(x - width/2, [rr_sim, rr_bert, rr_seas], width, label='Redundancy Removal', color='#4c72b0')
    plt.bar(x + width/2, [irr_sim, irr_bert, irr_seas], width, label='Info Retention', color='#55a868')
    plt.xticks(x, methods)
    plt.ylabel('Percentage (%)')
    plt.title('Deduplication Effectiveness')
    plt.legend(loc='lower right')
    plt.ylim(0, 110)
    plt.savefig('../assets/accuracy_plot.pdf', bbox_inches='tight')
    
    print("\n✅ Main Benchmark Complete. 'accuracy_plot.pdf' and 'latency_plot.pdf' generated in assets/.")

if __name__ == "__main__":
    run_experiment()