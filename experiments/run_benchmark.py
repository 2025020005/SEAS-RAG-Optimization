import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    print("🚀 [Main] Starting SEAS-RAG Benchmark (Vectorized & Optimized)...")
    
    loader = RealDataLoader()
    texts, labels = loader.get_dataset()
    total_docs = len(texts)
    
    fp = HybridFingerprint()
    all_vectors = fp.generate(texts, alpha=0.5)
    dense_vectors = fp.generate(texts, alpha=1.0) 
    
    # 调优：使用 6-bits (64个桶)，极大降低桶内计算量
    bucketer = PrefixBucketing(input_dim=all_vectors.shape[1], num_buckets=6) 
    fuser = AdaptiveFusion(base_threshold=0.60) 
    
    # ---------------------------------------------------------
    # 1. SimHash
    # ---------------------------------------------------------
    print("\n--- Running Baseline: SimHash ---")
    start = time.time()
    hashes = set()
    kept_sim = []
    for idx, text in enumerate(texts):
        tokens = text.lower().split()
        h = hash(tuple(tokens[:10])) if tokens else hash(text)
        if h not in hashes:
            hashes.add(h)
            kept_sim.append(idx)
    time_sim = (time.time() - start) * 1000
    rr_sim = (1 - len(kept_sim)/total_docs)*100
    irr_sim = calc_irr(kept_sim, labels)

    # ---------------------------------------------------------
    # 2. SBERT-KMeans 
    # ---------------------------------------------------------
    print("--- Running Baseline: SBERT-KMeans ---")
    start = time.time()
    kmeans = MiniBatchKMeans(n_clusters=len(set(labels)), batch_size=256, n_init='auto', random_state=42)
    kmeans.fit(dense_vectors)
    _, kept_bert = np.unique(kmeans.labels_, return_index=True)
    time_bert = (time.time() - start) * 1000
    rr_bert = (1 - len(kept_bert)/total_docs)*100
    irr_bert = calc_irr(kept_bert, labels)

    # ---------------------------------------------------------
    # 3. SemDeDup 
    # ---------------------------------------------------------
    print("--- Running Baseline: SemDeDup ---")
    start = time.time()
    sim_matrix = np.dot(dense_vectors, dense_vectors.T)
    keep_semdedup = np.ones(total_docs, dtype=bool)
    semdedup_threshold = 0.85 
    
    for i in range(total_docs):
        if not keep_semdedup[i]: continue
        sims = sim_matrix[i, i+1:]
        dupes = np.where(sims > semdedup_threshold)[0] + i + 1
        keep_semdedup[dupes] = False
                
    kept_semdedup_idx = np.where(keep_semdedup)[0]
    time_semdedup = (time.time() - start) * 1000
    rr_semdedup = (1 - len(kept_semdedup_idx)/total_docs)*100
    irr_semdedup = calc_irr(kept_semdedup_idx, labels)

    # ---------------------------------------------------------
    # 4. SEAS (Ours) 
    # ---------------------------------------------------------
    print("--- Running Model: SEAS ---")
    start = time.time()
    bucket_ids = bucketer.assign(all_vectors)
    groups = bucketer.group(np.arange(total_docs), bucket_ids)
    
    kept_seas = []
    for _, group in groups:
        kept_seas.extend(fuser.deduplicate_bucket(all_vectors, group['idx'].values, texts, use_adaptive=True))
        
    time_seas = (time.time() - start) * 1000
    rr_seas = (1 - len(kept_seas)/total_docs)*100
    irr_seas = calc_irr(kept_seas, labels)


    # =========================================================
    # 生成论文表格数据
    # =========================================================
    print("\n" + "="*50)
    print("🏆 PERFORMANCE COMPARISON (For LaTeX Table I)")
    print("="*50)
    
    table_data = [
        ["SimHash", rr_sim, irr_sim, time_sim],
        ["SBERT-KMeans", rr_bert, irr_bert, time_bert],
        ["SemDeDup", rr_semdedup, irr_semdedup, time_semdedup],
        ["SEAS (Ours)", rr_seas, irr_seas, time_seas]
    ]
    
    df = pd.DataFrame(table_data, columns=["Method", "RR (%)", "IRR (%)", "Latency (ms)"])
    print(df.to_string(index=False, float_format="%.1f"))
    print("="*50 + "\n")

    # =========================================================
    # 绘制图表
    # =========================================================
    os.makedirs('../assets', exist_ok=True)
    methods = ['SimHash', 'SBERT-KMeans', 'SemDeDup', 'SEAS (Ours)']
    
    # Plot 1: Efficiency (Latency)
    plt.figure(figsize=(7, 4))
    times = [time_sim, time_bert, time_semdedup, time_seas]
    plt.plot(methods, times, marker='o', color='#c44e52', linewidth=2)
    plt.ylabel('Latency (ms)')
    plt.title('Processing Efficiency (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.ylim(0, max(times) * 1.25)
    
    for i, v in enumerate(times):
        plt.text(i, v + max(times)*0.05, f"{v:.0f}ms", ha='center')
    plt.savefig('../assets/latency_plot.pdf', bbox_inches='tight')
    
    # Plot 2: Effectiveness (RR & IRR)
    plt.figure(figsize=(7, 4))
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, [rr_sim, rr_bert, rr_semdedup, rr_seas], width, label='Redundancy Removal', color='#4c72b0')
    plt.bar(x + width/2, [irr_sim, irr_bert, irr_semdedup, irr_seas], width, label='Info Retention', color='#55a868')
    plt.xticks(x, methods)
    plt.ylabel('Percentage (%)')
    plt.title('Deduplication Effectiveness')
    plt.legend(loc='lower right')
    plt.ylim(0, 115)
    plt.savefig('../assets/accuracy_plot.pdf', bbox_inches='tight')
    
    print("✅ Experiment Complete. 'accuracy_plot.pdf' and 'latency_plot.pdf' updated in assets/.")

if __name__ == "__main__":
    run_experiment()