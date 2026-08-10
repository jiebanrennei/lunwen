import numpy as np
import matplotlib.pyplot as plt

# 全局字体设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 38,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,  # 输出TrueType字体到PDF
    'ps.fonttype': 42,
})

# 数据整理
data = {
    'CLUHCS': {
        'ACM': {'Precision': 0.649, 'Recall': 0.937, 'F1': 0.753, 'Jaccard': 0.632},
        'DBLP': {'Precision': 0.823, 'Recall': 0.895, 'F1': 0.852, 'Jaccard': 0.778},
        'IMDB': {'Precision': 0.442, 'Recall': 0.638, 'F1': 0.520, 'Jaccard': 0.356}
    },
    'FCS-HGNN-I': {
        'ACM': {'Precision': 0.01, 'Recall': 0.006, 'F1': 0.007, 'Jaccard': 0.004},
        'DBLP': {'Precision': 0.231, 'Recall': 1.0, 'F1': 0.373, 'Jaccard': 0.231},
        'IMDB': {'Precision': 0.359, 'Recall': 0.85, 'F1': 0.43, 'Jaccard': 0.281}
    },
    'FCS-HGNN-T': {
        'ACM': {'Precision': 0.341, 'Recall': 0.995, 'F1': 0.493, 'Jaccard': 0.337},
        'DBLP': {'Precision': 0.255, 'Recall': 1.0, 'F1': 0.404, 'Jaccard': 0.255},
        'IMDB': {'Precision': 0.375, 'Recall': 0.86, 'F1': 0.459, 'Jaccard': 0.305}
    },
    'FCS-HGNN-H': {
        'ACM': {'Precision': 0.327, 'Recall': 0.474, 'F1': 0.379, 'Jaccard': 0.306},
        'DBLP': {'Precision': 0.25, 'Recall': 0.717, 'F1': 0.369, 'Jaccard': 0.23},
        'IMDB': {'Precision': 0.376, 'Recall': 0.622, 'F1': 0.413, 'Jaccard': 0.265}
    },
    'WC-index': {
        'ACM': {'Precision': 0.785, 'Recall': 0.025, 'F1': 0.048, 'Jaccard': 0.025},
        'DBLP': {'Precision': 0.291, 'Recall': 0.052, 'F1': 0.085, 'Jaccard': 0.045},
        'IMDB': {'Precision': 0.459, 'Recall': 0.036, 'F1': 0.057, 'Jaccard': 0.031}
    },
    'TransZero': {
        'ACM': {'Precision': 0.489, 'Recall': 0.608, 'F1': 0.532, 'Jaccard': 0.39},
        'DBLP': {'Precision': 0.409, 'Recall': 0.636, 'F1': 0.486, 'Jaccard': 0.333},
        'IMDB': {'Precision': 0.428, 'Recall': 0.343, 'F1': 0.366, 'Jaccard': 0.228}
    }
}

datasets = ['ACM', 'DBLP', 'IMDB']
methods = ['CLUHCS', 'FCS-HGNN-I', 'FCS-HGNN-T', 'FCS-HGNN-H', 'WC-index', 'TransZero']
patterns = ['////', '||||','\\\\\\\\', '..', '++', 'xx']

gray_levels = [
    '#FFFFFF',  # 白色（最亮，第1级）
    '#D3D3D3',  # 浅灰（银灰，第2级）
    '#B0B0B0',  # 中浅灰（第3级）
    '#808080',  # 中灰（标准灰，第4级）
    '#505050',  # 深灰（第5级）
    '#202020',  # 近黑灰（最暗，第6级，避免纯黑#000000过于刺眼）
]
# 边框全部使用黑色，确保图案清晰
border_levels = [
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
]

metrics = ['Precision', 'Recall', 'F1', 'Jaccard']

# 通用绘图参数
bar_width = 0.13
x = np.arange(len(datasets))

prop_Value = {'size': 38}

# 分别绘制四个指标
for metric in metrics:
    #plt.figure(figsize=(11, 4))
    plt.figure(figsize=(11, 3.8))
    ax = plt.gca()
    # 存储每个方法的绘图对象，用于后续筛选图例
    bar_handles = []
    bar_labels = []

    # 绘制每个方法，并收集句柄和标签
    for j, method in enumerate(methods):
        values = [data[method][d].get(metric, 0) for d in datasets]
        positions = x + j * bar_width

        bar = plt.bar(positions, values, bar_width,
                color=gray_levels[j],
                edgecolor=border_levels[j],
                hatch=patterns[j],
                linewidth=1.0,  # 增加边框线宽
                alpha=1.0,  # 使用完全不透明
                #linewidth=0.5,
                #alpha=0.8,
                label=method)
        bar_handles.append(bar)
        bar_labels.append(method)

    # 坐标轴设置
    plt.xticks(x + 2.5 * bar_width, datasets,fontsize=48)
    plt.ylabel(metric, fontsize=48)

    # 网格线和边框
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)


    # 根据不同指标设置图例显示规则（按需求修改核心逻辑）
    if metric == 'Precision':
        # 第一张图：显示前3个图例（CLUHCS, FCS-HGNN-I, FCS-HGNN-T）
        #plt.ylim(0, 1.05)
        plt.ylim(0, 1.19)

        plt.subplots_adjust(top=1)
        # 筛选前3个句柄和标签（索引0,1,2）
        selected_handles = bar_handles[:3]
        selected_labels = bar_labels[:3]
        plt.legend(handles=selected_handles, labels=selected_labels,
                   loc='upper right',
                   ncol=3,
                   handletextpad=0.1,
                   columnspacing=0.2,
                   labelspacing=0.1,
                   frameon=False,
                   handlelength=1.5,  # 减小图形块宽度，默认是2.0
                   handleheight=0.6,
                   #prop={'size': 27},
                   prop=prop_Value,
                   #bbox_to_anchor=(1.05, 1.225))
                   bbox_to_anchor=(1.05, 1.30))
    elif metric == 'Recall':
        # 第二张图：显示后3个图例（FCS-HGNN-H, WC-index, TransZero）
        plt.ylim(0, 1.19)
        plt.subplots_adjust(top=1)
        # 筛选后3个句柄和标签（索引3,4,5）
        selected_handles = bar_handles[3:6]
        selected_labels = bar_labels[3:6]
        plt.legend(handles=selected_handles, labels=selected_labels,
                   loc='upper left',
                   ncol=3,
                   handletextpad=0.1,
                   columnspacing=0.2,
                   labelspacing=0.1,
                   frameon=False,
                   handlelength=1.5,  # 减小图形块宽度，默认是2.0
                   handleheight=0.6,
                   #prop={'size': 27},
                   prop=prop_Value,
                   #bbox_to_anchor=(-0.21, 1.22))
                   bbox_to_anchor=(-0.21, 1.265))
    elif metric == 'F1':
        # 第三张图：不显示图例
        plt.ylim(0, 1.05)
        plt.subplots_adjust(top=1)
    elif metric == 'Jaccard':
        # 第四张图：不显示图例
        plt.ylim(0, 1.05)
        plt.subplots_adjust(top=1)

    #plt.yticks(fontsize=45)
    plt.yticks([0, 0.5, 1.0], fontsize=45)

    plt.subplots_adjust(top=0.90, bottom=0.15, left=0.10, right=0.95)

    # 保存为PDF和PNG
    plt.savefig(f'fig_perf_{metric}.pdf',
                bbox_inches='tight',
                pad_inches=0.03,  # 减小内边距
                dpi=1200,
                transparent=True)
    plt.savefig(f'fig_perf_{metric}.png',
                bbox_inches='tight',
                dpi=1200,
                transparent=True)
    plt.close()

print("四个图表已保存：")
print("1. Precision图（显示前3个方法图例）：fig_perf_Precision.pdf/png")
print("2. Recall图（显示后3个方法图例）：fig_perf_Recall.pdf/png")
print("3. F1图（不显示图例）：fig_perf_F1.pdf/png")
print("4. Jaccard图（不显示图例）：fig_perf_Jaccard.pdf/png")