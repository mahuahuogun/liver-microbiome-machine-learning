import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns

# 设置图形风格 - 使用矢量友好的设置
plt.style.use('default')
sns.set_palette("husl")

# 设置字体和矢量输出参数
plt.rcParams['pdf.fonttype'] = 42  # 确保字体在PDF中可编辑
plt.rcParams['ps.fonttype'] = 42  # 确保字体在PostScript中可编辑
plt.rcParams['font.family'] = 'Arial'  # 使用Arial字体，在AI中兼容性好
plt.rcParams['svg.fonttype'] = 'none'  # 确保SVG中的文字可编辑

# 读取数据
file_path = '/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/菌群分析/10项菌群.xlsx'
df = pd.read_excel(file_path)

print("数据列名:", df.columns.tolist())
print("数据形状:", df.shape)

# 计算每个分组的中位数
group_medians = df.groupby('分组').median()

# 菌群名称映射
bacteria_dict = {
    '肠杆菌对数': 'Enterobacteria',
    '肠球菌对数': 'Enterococcus',
    '乳酸菌最早数值对数': 'Lactobacillus',
    '双歧杆菌最早数值对数': 'Bifidobacterium',
    '类杆菌最早数值对数': 'Bacteroides',
    '奇异菌最早数值对数': 'Atopobium cluster',
    '丁酸梭菌最早数值对数': 'Clostridium butyricum',
    '柔嫩梭菌最早数值对数': 'Clostridium leptum',
    '直肠真杆菌最早数值对数': 'Eubacterium rectale',
    '普拉梭菌最早数值对数': 'Faecalibacterium prausnitzii'
}

# 所有菌群列表（用于计算总菌群数）
all_bacteria = list(bacteria_dict.keys())

# 疾病阶段标签
disease_stages = {
    1: 'Chronic\nLiver\nDisease',
    2: 'Compensated\nCirrhosis',
    3: 'Decompensated\nCirrhosis',
    4: 'ACLF'
}

disease_stages_short = {
    1: 'CLD', 2: 'CC', 3: 'DC', 4: 'ACLF'
}

# 菌群分类
pro_inflammatory_bacteria = ['肠杆菌对数', '肠球菌对数']  # 促炎菌
anti_inflammatory_bacteria = ['乳酸菌最早数值对数', '双歧杆菌最早数值对数',
                              '丁酸梭菌最早数值对数', '柔嫩梭菌最早数值对数',
                              '直肠真杆菌最早数值对数', '普拉梭菌最早数值对数']  # 抗炎菌

butyrate_producing_bacteria = ['丁酸梭菌最早数值对数', '柔嫩梭菌最早数值对数',
                               '直肠真杆菌最早数值对数', '普拉梭菌最早数值对数']  # 丁酸盐产生菌


# 1. 计算总菌群数（包含所有10种菌群）
def calculate_total_bacteria():
    """计算总菌群数（包含所有10种菌群）"""
    df_linear = 10 ** df[all_bacteria]  # 将对数值转换为线性值
    df['总菌群数'] = np.log10(df_linear.sum(axis=1))  # 计算总菌群数并取对数
    return df.groupby('分组')['总菌群数'].median()


# 2. 计算促炎/抗炎菌比例
def calculate_inflammatory_ratio():
    """计算促炎菌与抗炎菌比例"""
    df_linear = 10 ** df[all_bacteria]
    pro_inflammatory_total = df_linear[pro_inflammatory_bacteria].sum(axis=1)
    anti_inflammatory_total = df_linear[anti_inflammatory_bacteria].sum(axis=1)
    df['促炎抗炎比例'] = pro_inflammatory_total / anti_inflammatory_total
    return df.groupby('分组')['促炎抗炎比例'].median()


# 3. 计算丁酸盐产生菌比例
def calculate_butyrate_ratio():
    """计算丁酸盐产生菌在总细菌中的比例"""
    df_linear = 10 ** df[all_bacteria]
    butyrate_total = df_linear[butyrate_producing_bacteria].sum(axis=1)
    total_bacteria_linear = df_linear.sum(axis=1)
    df['丁酸盐菌比例'] = butyrate_total / total_bacteria_linear
    return df.groupby('分组')['丁酸盐菌比例'].median()


# 4. 计算肠球菌/直肠真杆菌比值
def calculate_enterococcus_rectale_ratio():
    """计算肠球菌与直肠真杆菌的比值"""
    df_linear = 10 ** df[all_bacteria]
    enterococcus = df_linear['肠球菌对数']
    eubacterium_rectale = df_linear['直肠真杆菌最早数值对数']
    df['肠球菌直肠真杆菌比值'] = enterococcus / eubacterium_rectale
    return df.groupby('分组')['肠球菌直肠真杆菌比值'].median()


# 执行计算
print("计算核心指标...")
total_bacteria = calculate_total_bacteria()
inflammatory_ratio = calculate_inflammatory_ratio()
butyrate_ratio = calculate_butyrate_ratio()
enterococcus_rectale_ratio = calculate_enterococcus_rectale_ratio()


# 多种图表展示 - 只保留三个维度
def create_diverse_plots():
    """创建多种类型的图表展示三个核心指标"""

    # 创建大图
    fig = plt.figure(figsize=(18, 12))

    # 使用GridSpec进行更灵活的布局
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])

    # 颜色设置
    colors = ['#A23B72', '#F18F01', '#8FBC94']

    # 1. 柱状图 - 促炎/抗炎比例
    ax1 = fig.add_subplot(gs[0, 0])
    stages = range(1, 5)
    bars = ax1.bar(stages, inflammatory_ratio, color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Pro-/Anti-inflammatory Ratio', fontweight='bold', fontsize=12)
    ax1.set_title('A. Inflammation Balance\n(Pro vs Anti-inflammatory)', fontweight='bold', fontsize=14, pad=20)
    ax1.set_xticks(stages)
    ax1.set_xticklabels([disease_stages_short[i] for i in stages])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 在柱子上添加数值
    for bar, value in zip(bars, inflammatory_ratio):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. 面积图 - 丁酸盐菌比例
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(stages, butyrate_ratio * 100, alpha=0.6, color=colors[1], label='Butyrate-producing Bacteria')
    ax2.plot(stages, butyrate_ratio * 100, marker='o', linewidth=3, markersize=8,
             color=colors[1], markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Percentage of Total Bacteria (%)', fontweight='bold', fontsize=12)
    ax2.set_title('B. Butyrate-producing Bacteria\n(4 Species)', fontweight='bold', fontsize=14, pad=20)
    ax2.set_xticks(stages)
    ax2.set_xticklabels([disease_stages_short[i] for i in stages])
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend()

    # 3. 折线图 - 肠球菌/直肠真杆菌比值
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(stages, enterococcus_rectale_ratio, marker='s', linewidth=3, markersize=8,
             color=colors[2], markerfacecolor='white', markeredgewidth=2)
    ax3.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Enterococcus/Eubacterium rectale Ratio', fontweight='bold', fontsize=12)
    ax3.set_title('C. Enterococcus/Eubacterium rectale\nRatio', fontweight='bold', fontsize=14, pad=20)
    ax3.set_xticks(stages)
    ax3.set_xticklabels([disease_stages_short[i] for i in stages])
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 4. 热力图风格 - 三个指标对比
    ax4 = fig.add_subplot(gs[1, 0])
    # 标准化数据用于热力图
    normalized_data = pd.DataFrame({
        'Inflammatory Ratio': (inflammatory_ratio - inflammatory_ratio.min()) / (
                inflammatory_ratio.max() - inflammatory_ratio.min()),
        'Butyrate Ratio': (butyrate_ratio - butyrate_ratio.min()) / (butyrate_ratio.max() - butyrate_ratio.min()),
        'Entero/E.rectale': (enterococcus_rectale_ratio - enterococcus_rectale_ratio.min()) / (
                enterococcus_rectale_ratio.max() - enterococcus_rectale_ratio.min())
    })

    im = ax4.imshow(normalized_data.T.values, cmap='RdYlBu_r', aspect='auto', alpha=0.8)

    ax4.set_xticks(range(len(stages)))
    ax4.set_xticklabels([disease_stages_short[i] for i in stages])
    ax4.set_yticks(range(3))
    ax4.set_yticklabels(['Inflammatory\nRatio', 'Butyrate\nRatio', 'Entero/\nE.rectale'])
    ax4.set_title('D. Normalized Indicators Heatmap', fontweight='bold', fontsize=14, pad=20)

    # 添加数值文本
    for i in range(len(stages)):
        for j in range(3):
            text = ax4.text(i, j, f'{normalized_data.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black", fontweight='bold')

    # 5. 雷达图 - 综合展示三个指标（修改为疾病越严重面积越小）
    ax5 = fig.add_subplot(gs[1, 1], polar=True)

    # 准备雷达图数据（标准化）
    # 使用 Butyrate Ratio, 1-Inflammatory Ratio, 1-Entero/E.rectale
    # 这样疾病越严重，雷达图面积越小
    categories = ['Butyrate Ratio', '1-Inflammatory\nRatio', '1-Entero/\nE.rectale']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 标准化数据
    butyrate_norm = (butyrate_ratio - butyrate_ratio.min()) / (butyrate_ratio.max() - butyrate_ratio.min())
    inflammatory_norm = (inflammatory_ratio - inflammatory_ratio.min()) / (
                inflammatory_ratio.max() - inflammatory_ratio.min())
    entero_norm = (enterococcus_rectale_ratio - enterococcus_rectale_ratio.min()) / (
                enterococcus_rectale_ratio.max() - enterococcus_rectale_ratio.min())

    # 使用转换后的指标：Butyrate Ratio, 1-Inflammatory Ratio, 1-Entero/E.rectale
    # 这样疾病越严重，后两个指标的值越小
    for i, stage in enumerate(stages):
        values = [
            butyrate_norm.iloc[i],  # Butyrate Ratio (越大越好)
            1 - inflammatory_norm.iloc[i],  # 1-Inflammatory Ratio (越大越好)
            1 - entero_norm.iloc[i]  # 1-Entero/E.rectale (越大越好)
        ]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=disease_stages_short[stage], markersize=8)
        ax5.fill(angles, values, alpha=0.1)

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_title('E. Microbiota Health Radar\n(Higher values indicate better health)', fontweight='bold', fontsize=14,
                  pad=20)
    ax5.legend(bbox_to_anchor=(1.2, 1), loc='upper left')

    # 设置雷达图的y轴范围，确保所有数据都能显示
    ax5.set_ylim(0, 1)

    # 6. 气泡图 - 三指标关系
    ax6 = fig.add_subplot(gs[1, 2])

    bubble_sizes = butyrate_ratio * 1000  # 气泡大小基于丁酸盐比例

    scatter = ax6.scatter(inflammatory_ratio, enterococcus_rectale_ratio, s=bubble_sizes,
                          c=butyrate_ratio * 100, cmap='viridis', alpha=0.7,
                          edgecolors='black', linewidth=0.5)

    # 添加阶段标签
    for i, stage in enumerate(stages):
        ax6.annotate(disease_stages_short[stage],
                     (inflammatory_ratio.iloc[i], enterococcus_rectale_ratio.iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontweight='bold')

    ax6.set_xlabel('Pro-/Anti-inflammatory Ratio', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Enterococcus/Eubacterium rectale Ratio', fontweight='bold', fontsize=12)
    ax6.set_title('F. Three Indicators Relationship\n(Size: Butyrate Ratio, Color: Butyrate Ratio)',
                  fontweight='bold', fontsize=14, pad=20)
    ax6.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Butyrate Ratio (%)', fontweight='bold')

    # 7. 三个核心指标的对比折线图
    ax7 = fig.add_subplot(gs[2, 0:3])

    # 标准化三个指标以便在同一图上比较
    normalized_inflammatory = (inflammatory_ratio - inflammatory_ratio.min()) / (
                inflammatory_ratio.max() - inflammatory_ratio.min())
    normalized_butyrate = (butyrate_ratio - butyrate_ratio.min()) / (butyrate_ratio.max() - butyrate_ratio.min())
    normalized_entero = (enterococcus_rectale_ratio - enterococcus_rectale_ratio.min()) / (
                enterococcus_rectale_ratio.max() - enterococcus_rectale_ratio.min())

    ax7.plot(stages, normalized_inflammatory, marker='s', linewidth=3, label='Inflammatory Ratio', color=colors[0])
    ax7.plot(stages, normalized_butyrate, marker='^', linewidth=3, label='Butyrate Ratio', color=colors[1])
    ax7.plot(stages, normalized_entero, marker='D', linewidth=3, label='Entero/E.rectale Ratio', color=colors[2])

    ax7.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Normalized Value', fontweight='bold', fontsize=12)
    ax7.set_title('G. Three Core Indicators Comparison\n(Normalized for Comparison)', fontweight='bold', fontsize=14,
                  pad=20)
    ax7.set_xticks(stages)
    ax7.set_xticklabels([disease_stages[i] for i in stages])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    plt.suptitle(
        'Comprehensive Analysis of Gut Microbiota in Liver Disease Progression',
        fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    # 保存为可编辑的矢量格式
    plt.savefig('/Users/daimengting/Desktop/comprehensive_microbiota_three_ratios.pdf',
                bbox_inches='tight', format='pdf')
    plt.savefig('/Users/daimengting/Desktop/comprehensive_microbiota_three_ratios.eps',
                bbox_inches='tight', format='eps')

    plt.show()


# 创建单个指标详细分析图
def create_individual_plots():
    """为每个指标创建单独的详细分析图"""

    # 1. 促炎/抗炎比例详细分析
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 6))

    # 左侧：分组柱状图
    stages = range(1, 5)
    bar_width = 0.35
    x_pos = np.arange(len(stages))

    # 计算促炎菌和抗炎菌的中位数
    pro_medians = df.groupby('分组')[pro_inflammatory_bacteria].median().mean(axis=1)
    anti_medians = df.groupby('分组')[anti_inflammatory_bacteria].median().mean(axis=1)

    bars1 = ax1a.bar(x_pos - bar_width / 2, pro_medians, bar_width, label='Pro-inflammatory (2)', color='#E74C3C',
                     alpha=0.8)
    bars2 = ax1a.bar(x_pos + bar_width / 2, anti_medians, bar_width, label='Anti-inflammatory (6)', color='#2ECC71',
                     alpha=0.8)

    ax1a.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax1a.set_ylabel('Median Bacterial Count (log)', fontweight='bold', fontsize=12)
    ax1a.set_title('Pro-inflammatory vs Anti-inflammatory Bacteria', fontweight='bold', fontsize=14)
    ax1a.set_xticks(x_pos)
    ax1a.set_xticklabels([disease_stages_short[i] for i in stages])
    ax1a.legend()
    ax1a.grid(True, alpha=0.3, axis='y')

    # 右侧：比例变化面积图
    ax1b.fill_between(stages, inflammatory_ratio, alpha=0.6, color='#A23B72')
    ax1b.plot(stages, inflammatory_ratio, marker='s', linewidth=3, markersize=8,
              color='#A23B72', markerfacecolor='white', markeredgewidth=2)
    ax1b.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax1b.set_ylabel('Pro-inflammatory / Anti-inflammatory Ratio', fontweight='bold', fontsize=12)
    ax1b.set_title('Inflammatory Balance Ratio\n(2 Pro vs 6 Anti species)', fontweight='bold', fontsize=14)
    ax1b.set_xticks(stages)
    ax1b.set_xticklabels([disease_stages_short[i] for i in stages])
    ax1b.grid(True, alpha=0.3)

    plt.tight_layout()
    # 保存为可编辑的矢量格式
    plt.savefig('/Users/daimengting/Desktop/inflammatory_ratio_detailed.pdf',
                bbox_inches='tight', format='pdf')
    plt.savefig('/Users/daimengting/Desktop/inflammatory_ratio_detailed.eps',
                bbox_inches='tight', format='eps')
    plt.show()

    # 2. 丁酸盐产生菌详细分析
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))

    # 左侧：饼图展示最终比例
    ax2a.pie([butyrate_ratio.iloc[-1] * 100, (1 - butyrate_ratio.iloc[-1]) * 100],
             labels=['Butyrate-producing', 'Other Bacteria'],
             colors=['#F39C12', '#BDC3C7'], autopct='%1.1f%%', startangle=90)
    ax2a.set_title('Butyrate-producing Bacteria in ACLF\n(Final Stage, 4 Species)', fontweight='bold', fontsize=14)

    # 右侧：各丁酸盐菌变化
    butyrate_individual = df.groupby('分组')[butyrate_producing_bacteria].median()

    for bacteria in butyrate_producing_bacteria:
        ax2b.plot(stages, butyrate_individual[bacteria], marker='^', linewidth=2,
                  markersize=6, label=bacteria_dict[bacteria])

    ax2b.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax2b.set_ylabel('Bacterial Count (log)', fontweight='bold', fontsize=12)
    ax2b.set_title('Individual Butyrate-producing Bacteria\n(4 Species)', fontweight='bold', fontsize=14)
    ax2b.set_xticks(stages)
    ax2b.set_xticklabels([disease_stages_short[i] for i in stages])
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)

    plt.tight_layout()
    # 保存为可编辑的矢量格式
    plt.savefig('/Users/daimengting/Desktop/butyrate_bacteria_detailed.pdf',
                bbox_inches='tight', format='pdf')
    plt.savefig('/Users/daimengting/Desktop/butyrate_bacteria_detailed.eps',
                bbox_inches='tight', format='eps')
    plt.show()

    # 3. 肠球菌/直肠真杆菌比值详细分析
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6))

    # 左侧：比值变化折线图
    ax3a.plot(stages, enterococcus_rectale_ratio, marker='D', linewidth=3, markersize=10,
              color='#8FBC94', markerfacecolor='white', markeredgewidth=3)

    # 添加个体数据点
    for stage in stages:
        stage_data = df[df['分组'] == stage]['肠球菌直肠真杆菌比值']
        jitter = np.random.normal(0, 0.02, len(stage_data))
        ax3a.scatter([stage + j for j in jitter], stage_data, alpha=0.4, color='#8FBC94', s=30)

    ax3a.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax3a.set_ylabel('Enterococcus/Eubacterium rectale Ratio', fontweight='bold', fontsize=12)
    ax3a.set_title('Enterococcus/Eubacterium rectale Ratio\nwith Individual Data Points', fontweight='bold',
                   fontsize=14)
    ax3a.set_xticks(stages)
    ax3a.set_xticklabels([disease_stages[i] for i in stages])
    ax3a.grid(True, alpha=0.3)

    # 右侧：两个菌种的对比图
    enterococcus_data = df.groupby('分组')['肠球菌对数'].median()
    eubacterium_data = df.groupby('分组')['直肠真杆菌最早数值对数'].median()

    x_pos = np.arange(len(stages))
    bar_width = 0.35

    bars1 = ax3b.bar(x_pos - bar_width / 2, enterococcus_data, bar_width,
                     label='Enterococcus', color='#E74C3C', alpha=0.8)
    bars2 = ax3b.bar(x_pos + bar_width / 2, eubacterium_data, bar_width,
                     label='Eubacterium rectale', color='#2ECC71', alpha=0.8)

    ax3b.set_xlabel('Disease Stage', fontweight='bold', fontsize=12)
    ax3b.set_ylabel('Bacterial Count (log)', fontweight='bold', fontsize=12)
    ax3b.set_title('Enterococcus vs Eubacterium rectale\n(Individual Counts)', fontweight='bold', fontsize=14)
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels([disease_stages_short[i] for i in stages])
    ax3b.legend()
    ax3b.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # 保存为可编辑的矢量格式
    plt.savefig('/Users/daimengting/Desktop/enterococcus_rectale_ratio_detailed.pdf',
                bbox_inches='tight', format='pdf')
    plt.savefig('/Users/daimengting/Desktop/enterococcus_rectale_ratio_detailed.eps',
                bbox_inches='tight', format='eps')
    plt.show()


# 执行绘图
print("创建多种图表展示...")
create_diverse_plots()

print("创建详细分析图...")
create_individual_plots()

# 统计分析报告
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS SUMMARY")
print("=" * 80)

# 计算变化百分比
inflammatory_change = ((inflammatory_ratio.iloc[-1] - inflammatory_ratio.iloc[0]) / inflammatory_ratio.iloc[0]) * 100
butyrate_change = ((butyrate_ratio.iloc[-1] - butyrate_ratio.iloc[0]) / butyrate_ratio.iloc[0]) * 100
entero_rectale_change = ((enterococcus_rectale_ratio.iloc[-1] - enterococcus_rectale_ratio.iloc[0]) /
                         enterococcus_rectale_ratio.iloc[0]) * 100

print(f"\n1. Pro-inflammatory/Anti-inflammatory Ratio:")
print(f"   {disease_stages_short[1]}: {inflammatory_ratio.iloc[0]:.3f}")
print(f"   {disease_stages_short[4]}: {inflammatory_ratio.iloc[-1]:.3f}")
print(f"   Change: {inflammatory_change:+.1f}%")

print(f"\n2. Butyrate-producing Bacteria Ratio (4 species):")
print(f"   {disease_stages_short[1]}: {butyrate_ratio.iloc[0] * 100:.2f}%")
print(f"   {disease_stages_short[4]}: {butyrate_ratio.iloc[-1] * 100:.2f}%")
print(f"   Change: {butyrate_change:+.1f}%")

print(f"\n3. Enterococcus/Eubacterium rectale Ratio:")
print(f"   {disease_stages_short[1]}: {enterococcus_rectale_ratio.iloc[0]:.3f}")
print(f"   {disease_stages_short[4]}: {enterococcus_rectale_ratio.iloc[-1]:.3f}")
print(f"   Change: {entero_rectale_change:+.1f}%")

# 保存分析结果
output_path = '/Users/daimengting/Desktop/microbiota_core_analysis.xlsx'
with pd.ExcelWriter(output_path) as writer:
    pd.DataFrame({'Inflammatory_Ratio': inflammatory_ratio}).to_excel(writer, sheet_name='Inflammatory_Ratio')
    pd.DataFrame({'Butyrate_Ratio': butyrate_ratio}).to_excel(writer, sheet_name='Butyrate_Ratio')
    pd.DataFrame({'Enterococcus_Rectale_Ratio': enterococcus_rectale_ratio}).to_excel(writer,
                                                                                      sheet_name='Entero_Rectale_Ratio')

    # 保存所有菌群的中位数数据
    group_medians[all_bacteria].to_excel(writer, sheet_name='All_Bacteria_Medians')

print(f"\n分析完成！所有图表和数据已保存。")
print(f"\n主要发现:")
print(f"- 促炎/抗炎菌比例{'增加' if inflammatory_change > 0 else '下降'}了{abs(inflammatory_change):.1f}%")
print(f"- 丁酸盐产生菌比例（4种）{'增加' if butyrate_change > 0 else '下降'}了{abs(butyrate_change):.1f}%")
print(f"- 肠球菌/直肠真杆菌比值{'增加' if entero_rectale_change > 0 else '下降'}了{abs(entero_rectale_change):.1f}%")