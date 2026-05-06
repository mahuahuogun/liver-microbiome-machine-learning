import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

# ==================== 配置 ====================
INPUT_FILE = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组/汇总947.xlsx"
OUTPUT_DIR = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组"

GROUP_MAP = {1: 'CLD', 2: 'CC', 3: 'DC', 4: 'ACLF'}

RAW_COLS = {
    'Enterococcus': '肠球菌最早数值',
    'Eubacterium rectale': '直肠真杆菌最早数值',
    'Enterobacterium': '肠杆菌最早数值',
    'Lactobacillus': '乳酸菌最早数值',
    'Bifidobacterium': '双歧杆菌最早数值',
    'Clostridium butyricum': '丁酸梭菌最早数值',
    'Clostridium leptum': '柔嫩梭菌最早数值',
    'Faecalibacterium prausnitzii': '普拉梭菌最早数值',
}
TOTAL_COL = '菌群总数'


def prepare_radar_data(df):
    """计算三个归一化指数，使用分指标 Min‑Max 归一化"""
    # 计算三个原始比值的组中位数
    groups = [1, 2, 3, 4]
    medians = {g: {} for g in groups}
    for g in groups:
        sub = df[df['分组'] == g]
        e = sub['肠球菌最早数值']
        er = sub['直肠真杆菌最早数值']
        eb = sub['肠杆菌最早数值']
        lac = sub['乳酸菌最早数值']
        bif = sub['双歧杆菌最早数值']
        cb = sub['丁酸梭菌最早数值']
        cl = sub['柔嫩梭菌最早数值']
        fp = sub['普拉梭菌最早数值']
        total = sub[TOTAL_COL]

        pro = e + eb
        anti = lac + bif + cb + cl + er + fp
        buty = er + fp + cl + cb

        medians[g]['Butyrate'] = (buty / (total + 1e-10)).median()
        medians[g]['Pro_Anti'] = (pro / (anti + 1e-10)).median()
        medians[g]['E_Er'] = (e / (er + 1e-10)).median()

    # 分指标独立 Min‑Max 归一化
    # Butyrate 越高越好 → 直接 Min‑Max
    buty_vals = np.array([medians[g]['Butyrate'] for g in groups])
    buty_norm = (buty_vals - buty_vals.min()) / (buty_vals.max() - buty_vals.min())

    # Pro/Anti 越高越差 → 1 - Min‑Max
    pro_vals = np.array([medians[g]['Pro_Anti'] for g in groups])
    pro_norm = 1 - (pro_vals - pro_vals.min()) / (pro_vals.max() - pro_vals.min())

    # E/Er 越高越差 → 1 - Min‑Max
    eer_vals = np.array([medians[g]['E_Er'] for g in groups])
    eer_norm = 1 - (eer_vals - eer_vals.min()) / (eer_vals.max() - eer_vals.min())

    # 组装结果
    radar_scores = {}
    group_labels = {1: 'CLD', 2: 'CC', 3: 'DC', 4: 'ACLF'}
    for i, g in enumerate(groups):
        radar_scores[group_labels[g]] = [
            buty_norm[i],   # Normalized Butyrate Ratio
            pro_norm[i],    # Normalized Anti-inflammatory Balance (1−Pro/Anti)
            eer_norm[i],    # Normalized Eubiosis Index (1−E/Er)
        ]

    # 打印各组的原始中位数和归一化后的值，方便核对
    print("\n原始中位数:")
    for g in groups:
        print(f"  {group_labels[g]}: Butyrate={medians[g]['Butyrate']:.4f}, "
              f"Pro/Anti={medians[g]['Pro_Anti']:.4f}, E/Er={medians[g]['E_Er']:.4f}")
    print("\n归一化后:")
    for g_name, vals in radar_scores.items():
        print(f"  {g_name}: Buty={vals[0]:.3f}, Anti-infl={vals[1]:.3f}, Eubiosis={vals[2]:.3f}")

    return radar_scores


def plot_radar_chart(radar_scores, output_dir):
    """绘制雷达图并保存"""
    categories = [
        'Normalized\nButyrate Ratio',
        'Normalized Anti-inflammatory\nBalance (1−Pro/Anti)',
        'Normalized Eubiosis\nIndex (1−E/Er)'
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    group_names = ['CLD', 'CC', 'DC', 'ACLF']

    for idx, group in enumerate(group_names):
        values = radar_scores[group]
        values += values[:1]  # 闭合多边形
        ax.plot(angles, values, 'o-', linewidth=2, label=group, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Multi-dimensional Microbiome Indices', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    out_png = os.path.join(output_dir, 'Figure_Radar_Chart.png')
    out_pdf = os.path.join(output_dir, 'Figure_Radar_Chart.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n雷达图已保存: {out_png} 和 {out_pdf}")


def main():
    df = pd.read_excel(INPUT_FILE)
    radar_scores = prepare_radar_data(df)
    plot_radar_chart(radar_scores, OUTPUT_DIR)


if __name__ == "__main__":
    main()