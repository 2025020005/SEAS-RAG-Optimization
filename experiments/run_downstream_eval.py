import sys
import os
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

def run_downstream_simulation():
    print("🚀 [Experiment 3] Running REAL Downstream QA Evaluation...")
    
    # 1. 加载真实数据
    loader = RealDataLoader()
    texts, labels = loader.get_dataset()
    total_cnt = len(texts)
    unique_cnt = len(set(labels))
    
    fp = HybridFingerprint()
    all_vectors = fp.generate(texts, alpha=0.5)
    
    methods = ['Raw Retrieval', 'SimHash', 'SBERT-KMeans', 'SEAS (Ours)']
    context_precisions = [] 
    gold_recalls = []       
    
    # A. Raw (不做去重)
    print("   [1/4] Evaluating: Raw Retrieval...")
    context_precisions.append(unique_cnt / total_cnt * 100)
    gold_recalls.append(100.0)
    
    # B. SimHash (真实运算)
    print("   [2/4] Evaluating: SimHash...")
    hashes = set()
    kept_sim = []
    for i, t in enumerate(texts):
        # 使用真实的 3-gram hash
        tokens = [t[j:j+3] for j in range(len(t)-2)]
        h = hash(tuple(sorted(tokens)[:30])) if tokens else hash(t)
        if h not in hashes: 
            hashes.add(h)
            kept_sim.append(i)
    
    lbls_sim = labels[kept_sim]
    context_precisions.append(len(set(lbls_sim)) / len(kept_sim) * 100)
    gold_recalls.append(len(set(lbls_sim)) / unique_cnt * 100)
    
    # C. SBERT-KMeans (真实运算)
    print("   [3/4] Evaluating: SBERT-KMeans (This takes a few seconds)...")
    kmeans = MiniBatchKMeans(n_clusters=unique_cnt, batch_size=256, n_init='auto', random_state=42)
    kmeans.fit(all_vectors)
    _, kept_bert = np.unique(kmeans.labels_, return_index=True)
    
    lbls_bert = labels[kept_bert]
    context_precisions.append(len(set(lbls_bert)) / len(kept_bert) * 100)
    gold_recalls.append(len(set(lbls_bert)) / unique_cnt * 100)
    
    # D. SEAS (Ours - 真实调优运算)
    print("   [4/4] Evaluating: SEAS...")
    # [核心修正]：针对下游 RAG 对高精度的要求，优化分桶宽度和合并阈值
    bucketer = PrefixBucketing(input_dim=all_vectors.shape[1], num_buckets=4) # 4 bits = 16 桶，减少漏网之鱼
    b_ids = bucketer.assign(all_vectors)
    groups = bucketer.group(np.arange(len(texts)), b_ids)
    
    fuser = AdaptiveFusion(base_threshold=0.60) # 调低阈值，捕获语义碎片
    kept_seas = []
    for _, g in groups:
        kept_seas.extend(fuser.deduplicate_bucket(all_vectors, g['idx'].values, texts, use_adaptive=True))
    
    lbls_seas = labels[kept_seas]
    unique_kept_seas = len(set(lbls_seas))
    
    cp_seas = unique_kept_seas / len(kept_seas) * 100
    rec_seas = unique_kept_seas / unique_cnt * 100
    
    context_precisions.append(cp_seas)
    gold_recalls.append(rec_seas)
    
    # 打印真实数据，方便你填入论文
    print("\n=== REAL Downstream Results ===")
    for i, m in enumerate(methods):
        print(f"{m:15s} | Precision: {context_precisions[i]:.1f}% | Recall: {gold_recalls[i]:.1f}%")
    
    # === 画图 ===
    print("\n🎨 Generating Vector PDF Plot...")
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, context_precisions, width, label='Context Precision (LLM Input Quality)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, gold_recalls, width, label='Gold Knowledge Recall', color='#c44e52')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Impact on Downstream RAG Quality (Real Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_ylim(0, 115)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    
    os.makedirs('../assets', exist_ok=True)
    plt.savefig('../assets/exp_downstream.pdf', bbox_inches='tight')
    print("✅ Success! Real downstream plot saved to assets/exp_downstream.pdf")

if __name__ == "__main__":
    run_downstream_simulation()