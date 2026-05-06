import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置图形风格
plt.style.use('default')

# 解决中文字体问题
try:
    # 尝试不同的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("中文字体设置成功")
except:
    print("中文字体设置失败，将使用英文标签")

# 读取数据
file_path = '/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/菌群分析/10项菌群.xlsx'
df = pd.read_excel(file_path)

# 菌群名称
bacteria_names = df.columns[1:].tolist()

# 计算每个分组中每种菌群的中位数
group_medians = df.groupby('分组')[bacteria_names].median()

# 菌群中英文对照
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

# 疾病阶段标签（使用英文避免字体问题）
disease_stages = {
    1: 'Chronic\nLiver\nDisease',
    2: 'Compensated\nCirrhosis',
    3: 'Decompensated\nCirrhosis',
    4: 'ACLF'
}

# 分析每个菌群的趋势模式
def analyze_trend_pattern(medians):
    """分析菌群的变化趋势模式"""
    # 计算相邻组间的变化
    changes = [medians.iloc[i + 1] - medians.iloc[i] for i in range(len(medians) - 1)]

    # 计算总体变化
    total_change = medians.iloc[-1] - medians.iloc[0]

    # 判断上升/下降的段数
    rising_segments = sum(1 for change in changes if change > 0)
    falling_segments = sum(1 for change in changes if change < 0)

    # 基于变化模式分类（使用英文）
    if rising_segments == 3:
        return "Consistently Increasing"
    elif falling_segments == 3:
        return "Consistently Decreasing"
    elif rising_segments == 2 and falling_segments == 1:
        if changes[2] < 0:
            return "Increase then Decrease"
        elif changes[1] < 0:
            return "Fluctuating Increase"
        else:
            return "Decrease then Increase"
    elif rising_segments == 1 and falling_segments == 2:
        if changes[2] > 0:
            return "Decrease then Increase"
        elif changes[1] > 0:
            return "Fluctuating Decrease"
        else:
            return "Increase then Decrease"
    else:
        if total_change > 0.5:
            return "Overall Increase"
        elif total_change < -0.5:
            return "Overall Decrease"
        else:
            return "Relatively Stable"

# 对每个菌群进行趋势分类
trend_groups = {}
for bacteria in bacteria_names:
    medians = group_medians[bacteria]
    trend = analyze_trend_pattern(medians)

    if trend not in trend_groups:
        trend_groups[trend] = []
    trend_groups[trend].append(bacteria)

# 打印分类结果
print("Bacterial Trend Classification Results:")
for trend, bacteria_list in trend_groups.items():
    print(f"\n{trend} Group ({len(bacteria_list)} bacterial species):")
    for bacteria in bacteria_list:
        en_name = bacteria_dict[bacteria]
        medians = group_medians[bacteria]
        total_change = medians.iloc[-1] - medians.iloc[0]
        print(f"  - {bacteria} ({en_name}), Total Change: {total_change:.3f}")

# 为每个趋势组创建图表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 创建每个趋势组的图表
for trend, bacteria_list in trend_groups.items():
    plt.figure(figsize=(10, 6))

    for i, bacteria in enumerate(bacteria_list):
        medians = group_medians[bacteria]
        en_name = bacteria_dict[bacteria]

        # 绘制折线
        plt.plot(range(1, 5), medians,
                 marker='o', linewidth=2.5, markersize=8,
                 color=colors[i % len(colors)],
                 label=f'{en_name}')

        # 在最后一个点添加数值标签
        plt.annotate(f'{medians.iloc[-1]:.2f}',
                     xy=(4, medians.iloc[-1]),
                     xytext=(4.1, medians.iloc[-1]),
                     fontsize=9, ha='left', va='center')

    plt.xlabel('Disease Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Median Bacterial Count (log)', fontsize=12, fontweight='bold')
    plt.title(f'Bacterial Trends: {trend} Pattern\n({len(bacteria_list)} species)',
              fontsize=14, fontweight='bold')
    plt.xticks([1, 2, 3, 4], [disease_stages[1], disease_stages[2],
                              disease_stages[3], disease_stages[4]])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 调整布局并保存为EPS格式
    plt.tight_layout()
    filename = f'/Users/daimengting/Desktop/trend_group_{trend.replace(" ", "_")}.eps'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='eps')
    plt.show()

# 创建趋势汇总表格
trend_summary = []
for bacteria in bacteria_names:
    medians = group_medians[bacteria]
    trend = analyze_trend_pattern(medians)
    change_g1_g4 = medians.iloc[-1] - medians.iloc[0]
    percent_change = (change_g1_g4 / medians.iloc[0]) * 100

    trend_summary.append({
        'Bacteria_CN': bacteria,
        'Bacteria_EN': bacteria_dict[bacteria],
        'Trend_Pattern': trend,
        'Group1_Median': medians.iloc[0],
        'Group4_Median': medians.iloc[-1],
        'Absolute_Change': change_g1_g4,
        'Percent_Change': percent_change
    })

trend_df = pd.DataFrame(trend_summary)

# 按趋势模式排序（自定义顺序）
trend_order = ["Consistently Increasing", "Overall Increase", "Fluctuating Increase",
               "Increase then Decrease", "Decrease then Increase", "Fluctuating Decrease",
               "Overall Decrease", "Consistently Decreasing", "Relatively Stable"]
trend_df['Trend_Order'] = trend_df['Trend_Pattern'].apply(
    lambda x: trend_order.index(x) if x in trend_order else len(trend_order))
trend_df = trend_df.sort_values('Trend_Order')

print("\n" + "=" * 80)
print("TREND ANALYSIS SUMMARY")
print("=" * 80)
print(trend_df[['Bacteria_EN', 'Trend_Pattern', 'Group1_Median', 'Group4_Median', 'Percent_Change']].to_string(
    index=False))

# 保存趋势分析结果
output_path = '/Users/daimengting/Desktop/bacteria_trend_analysis.xlsx'
with pd.ExcelWriter(output_path) as writer:
    trend_df.to_excel(writer, sheet_name='Trend_Analysis_Summary', index=False)

    # 为每个趋势组创建单独的工作表
    for trend in trend_groups.keys():
        trend_bacteria = trend_groups[trend]
        trend_data = group_medians[trend_bacteria].T
        trend_data.to_excel(writer, sheet_name=f'{trend.replace(" ", "_")}_Trend')

print(f"\nAnalysis completed! Trend grouping charts and detailed data saved to: {output_path}")

# 创建趋势分布饼图
plt.figure(figsize=(10, 8))
trend_counts = {trend: len(bacteria_list) for trend, bacteria_list in trend_groups.items()}
colors_pie = ['#2ca02c', '#98df8a', '#ff7f0e', '#ffbb78', '#d62728',
              '#ff9896', '#1f77b4', '#aec7e8', '#c5b0d5']

plt.pie(trend_counts.values(), labels=trend_counts.keys(), autopct='%1.1f%%',
        colors=colors_pie[:len(trend_counts)], startangle=90)
plt.title('Distribution of Bacterial Trend Patterns', fontsize=14, fontweight='bold')
plt.axis('equal')

plt.tight_layout()
plt.savefig('/Users/daimengting/Desktop/trend_distribution_pie.eps', dpi=300, bbox_inches='tight', format='eps')
plt.show()

# 创建所有趋势组的汇总图（小图形式）
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 为每个趋势组创建小图
trend_list = list(trend_groups.keys())
for i, trend in enumerate(trend_list[:6]):  # 最多显示6个趋势组
    ax = axes[i]
    bacteria_list = trend_groups[trend]

    for j, bacteria in enumerate(bacteria_list):
        medians = group_medians[bacteria]
        en_name = bacteria_dict[bacteria]

        ax.plot(range(1, 5), medians,
                marker='o', linewidth=2, markersize=4,
                color=colors[j % len(colors)],
                label=en_name)

    ax.set_title(f'{trend}\n({len(bacteria_list)} species)', fontsize=10)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['CLD', 'CC', 'DC', 'ACLF'])
    ax.grid(True, alpha=0.3)

    # 只在第一个图添加y轴标签
    if i % 3 == 0:
        ax.set_ylabel('Median Value (log)', fontsize=9)

    # 简化图例（如果菌群太多）
    if len(bacteria_list) <= 4:
        ax.legend(fontsize=7)

# 隐藏多余的子图
for i in range(len(trend_list), 6):
    axes[i].set_visible(False)

plt.suptitle('Summary of All Bacterial Trend Patterns', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/daimengting/Desktop/trend_summary_all.eps', dpi=300, bbox_inches='tight', format='eps')
plt.show()

# 创建所有菌群的汇总图
plt.figure(figsize=(14, 8))

for i, bacteria in enumerate(bacteria_names):
    medians = group_medians[bacteria]
    en_name = bacteria_dict[bacteria]

    plt.plot(range(1, 5), medians,
             marker='o', linewidth=2, markersize=6,
             color=colors[i % len(colors)],
             label=en_name)

plt.xlabel('Disease Stage', fontsize=12, fontweight='bold')
plt.ylabel('Median Bacterial Count (log)', fontsize=12, fontweight='bold')
plt.title('Changes in Gut Microbiota During Liver Disease Progression',
          fontsize=16, fontweight='bold')
plt.xticks([1, 2, 3, 4], [disease_stages[1], disease_stages[2],
                          disease_stages[3], disease_stages[4]])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/daimengting/Desktop/all_bacteria_trends.eps',
            dpi=300, bbox_inches='tight', format='eps')
plt.show()

print("所有图表生成完成！EPS格式文件已保存。")