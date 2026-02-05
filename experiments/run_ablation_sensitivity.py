import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import RealDataLoader
from fingerprint import HybridFingerprint
from bucketing import PrefixBucketing
from fusion import AdaptiveFusion

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'

def calc_metrics(kept_indices, true_labels, total_len):
    rr = (1 - len(kept_indices) / total_len) * 100
    
    kept_labels = set(true_labels[kept_indices])
    total_unique = set(true_labels)
    irr = len(kept_labels) / len(total_unique) * 100
    return rr, irr

def run_ablation_study(texts, labels):
    print("\n🚀 [Experiment 1] Running Ablation Study...")
    
    # 初始化
    fp = HybridFingerprint()
    bucketer = PrefixBucketing(input_dim=384+64)
    fuser = AdaptiveFusion()
    
    results = []
    
    # 1. SEAS Full Model (Full Hybrid + Adaptive)
    print("   Running: SEAS Full Model...")
    vectors = fp.generate(texts, alpha=0.5) # 混合
    bucket_ids = bucketer.assign(vectors)
    groups = bucketer.group(np.arange(len(texts)), bucket_ids)
    
    kept = []
    for _, group in groups:
        kept.extend(fuser.deduplicate_bucket(vectors, group['idx'].values, texts, use_adaptive=True))
    rr, irr = calc_metrics(kept, labels, len(texts))
    results.append(['SEAS Full', rr, irr])

    # 2. w/o Hybrid (Dense Only)
    print("   Running: w/o Hybrid Fingerprint...")
    vectors_dense = fp.generate(texts, alpha=1.0) # 仅 Dense
    # 重新分桶（因为向量变了）
    bucketer_dense = PrefixBucketing(input_dim=384+64) 
    bucket_ids = bucketer_dense.assign(vectors_dense)
    groups = bucketer_dense.group(np.arange(len(texts)), bucket_ids)
    
    kept = []
    for _, group in groups:
        kept.extend(fuser.deduplicate_bucket(vectors_dense, group['idx'].values, texts, use_adaptive=True))
    rr, irr = calc_metrics(kept, labels, len(texts))
    results.append(['w/o Hybrid', rr, irr])

    # 3. w/o Adaptive Threshold (Fixed)
    print("   Running: w/o Adaptive Threshold...")
    # 使用混合向量，但关闭自适应
    vectors = fp.generate(texts, alpha=0.5)
    bucket_ids = bucketer.assign(vectors)
    groups = bucketer.group(np.arange(len(texts)), bucket_ids)
    
    kept = []
    for _, group in groups:
        kept.extend(fuser.deduplicate_bucket(vectors, group['idx'].values, texts, use_adaptive=False))
    rr, irr = calc_metrics(kept, labels, len(texts))
    results.append(['w/o Adaptive', rr, irr])
    
    # 输出表格数据
    df_res = pd.DataFrame(results, columns=['Variant', 'RR (%)', 'IRR (%)'])
    print("\n=== Ablation Results (for Table 2) ===")
    print(df_res)
    return df_res

def run_sensitivity_analysis(texts, labels):
    print("\n🚀 [Experiment 2] Running Hyperparameter Sensitivity (Bucketing Bits)...")
    
    fp = HybridFingerprint()
    vectors = fp.generate(texts, alpha=0.5)
    fuser = AdaptiveFusion()
    
    # 改变分桶的比特数 (n_bits)，观察 Latency 和 RR 的变化
    # n_bits 越多 -> 桶越多 -> 桶内数据越少 -> 速度越快 -> 但可能切断语义 -> RR降低
    bit_settings = [4, 6, 8, 10, 12]
    
    sensitivity_data = []
    
    for n_bits in bit_settings:
        print(f"   Testing n_bits = {n_bits}...")
        # 重新初始化分桶器
        bucketer = PrefixBucketing(input_dim=384+64, num_buckets=n_bits) 
        
        start = time.time()
        bucket_ids = bucketer.assign(vectors)
        groups = bucketer.group(np.arange(len(texts)), bucket_ids)
        
        kept = []
        for _, group in groups:
            kept.extend(fuser.deduplicate_bucket(vectors, group['idx'].values, texts, use_adaptive=True))
        
        duration = (time.time() - start) * 1000
        rr, irr = calc_metrics(kept, labels, len(texts))
        
        sensitivity_data.append({
            'n_bits': n_bits,
            'Latency': duration,
            'RR': rr,
            'IRR': irr
        })
    
    df_sens = pd.DataFrame(sensitivity_data)
    print("\n=== Sensitivity Results ===")
    print(df_sens)
    
    # 画图
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color = '#4c72b0'
    ax1.set_xlabel('Number of LSH Bits (Bucketing Granularity)')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.plot(df_sens['n_bits'], df_sens['Latency'], marker='o', color=color, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = '#55a868'
    ax2.set_ylabel('Redundancy Removal (%)', color=color)  
    ax2.plot(df_sens['n_bits'], df_sens['RR'], marker='s', linestyle='--', color=color, linewidth=2, label='RR')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Hyperparameter Sensitivity: Impact of Bucketing Bits')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存矢量图
    os.makedirs('../assets', exist_ok=True)
    plt.savefig('../assets/exp_sensitivity.pdf', bbox_inches='tight')
    print("✅ Sensitivity plot saved to assets/exp_sensitivity.pdf")

if __name__ == "__main__":
    # 加载真实数据
    loader = RealDataLoader()
    texts, labels = loader.get_dataset()
    
    # 运行消融实验
    run_ablation_study(texts, labels)
    
    # 运行敏感性分析
    run_sensitivity_analysis(texts, labels)