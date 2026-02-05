import matplotlib.pyplot as plt
import numpy as np

# 设置学术风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif' # 使用衬线字体 (Times New Roman风格)

def generate_pdf_plot():
    methods = ['SimHash', 'SBERT-Cluster', 'SEAS (Ours)']
    rr_scores = [58.2, 74.5, 75.1]
    irr_scores = [82.1, 91.5, 95.2]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, rr_scores, width, label='Redundancy Removal (RR)', color='#4c72b0', alpha=0.9)
    rects2 = ax.bar(x + width/2, irr_scores, width, label='Info Retention (IRR)', color='#55a868', alpha=0.9)
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Performance Comparison on OpenWeb-QA', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', frameon=True)
    
    # 自动标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    # 关键：保存为 PDF
    plt.savefig('Figure1_Performance.pdf', format='pdf')
    print("Success: Figure1_Performance.pdf generated!")

if __name__ == "__main__":
    generate_pdf_plot()