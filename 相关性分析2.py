import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
FILE_PATH = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组/汇总947.xlsx"
OUTPUT_DIR = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组/Correlation_Filtered_English"

# 10种菌群的Excel列名（对数转换值）
BACTERIA_COLS = {
    'Enterobacteria': '肠杆菌对数',
    'Enterococcus': '肠球菌对数',
    'Lactobacillus': '乳酸菌最早数值对数',
    'Bifidobacterium': '双歧杆菌最早数值对数',
    'Bacteroides': '类杆菌最早数值对数',
    'Atopobium cluster': '奇异菌最早数值对数',
    'Clostridium butyricum': '丁酸梭菌最早数值对数',
    'Clostridium leptum': '柔嫩梭菌最早数值对数',
    'Eubacterium rectale': '直肠真杆菌最早数值对数',
    'Faecalibacterium prausnitzii': '普拉梭菌最早数值对数'
}

# 临床指标中文列名 → 英文缩写映射（包含新增的SII、NLR、PLR、NPAR）
CLINICAL_EN_MAP = {
    # Demographics
    '年龄数值': 'Age',
    '性别': 'Male',
    # Liver Function
    '总蛋白（正）TPgL最早数值': 'TP',
    '白蛋白（正）AlbgL最早数值': 'Alb',
    '球蛋白（正）GLBgL最早数值': 'GLB',
    '白球蛋白比（正）AG最早数值': 'Alb/GLB',
    '丙氨酸氨基转移酶ALTUL最早数值': 'ALT',
    '天门冬氨基转移酶ASTUL最早数值': 'AST',
    'ASTALT最早数值': 'AST/ALT',
    'γ谷氨酰基转移酶GGTUL最早数值': 'GGT',
    '碱性磷酸酶测定ALPUL最早数值': 'ALP',
    '总胆红素TBilumolL.1最早数值': 'TBil',
    '直接胆红素DBilumolL最早数值': 'DBil',
    '间接胆红素IBilumolL最早数值': 'IBil',
    '胆碱酯酶ChEUL最早数值': 'ChE',
    '总胆汁酸TBAumolL最早数值': 'TBA',
    '甘脯二肽氨基肽酶（GPDA）UL最早数值': 'GPDA',
    'αL岩藻糖苷酶AFUUL最早数值': 'AFU',
    '腺苷脱氨酶ADA血清UL最早数值': 'ADA',
    # Coagulation
    '凝血酶原时间血秒最早数值': 'PT',
    '国际标准化血最早数值': 'INR',
    '活化部分凝血活酶时间血秒最早数值': 'APTT',
    '凝血酶时间血秒最早数值': 'TT',
    '纤维蛋白原血gL最早数值': 'FIB',
    'D二聚体血ugLDDU最早数值': 'DD',
    # Inflammation & Hematology
    '白细胞计数血10E9L最早数值': 'WBC',
    '中性粒细胞血最早数值': 'Neutrophil%',
    '淋巴细胞血最早数值': 'Lymphocyte%',
    '单核细胞血最早数值': 'Monocyte%',
    '嗜酸性粒细胞血最早数值': 'Eosinophil%',
    '嗜碱性粒细胞血最早数值': 'Basophils%',
    '中性粒细胞血10E9L最早数值': 'Neutrophil',
    '淋巴细胞血10E9L最早数值': 'Lymphocyte',
    '单核细胞血10E9L最早数值': 'Monocyte',
    '嗜酸性粒细胞血10E9L最早数值': 'Eosinophil',
    '嗜碱性粒细胞血10E9L最早数值': 'Basophils',
    '红细胞计数血10E12L最早数值': 'RBC',
    '血红蛋白血gL最早数值': 'Hb',
    '红细胞压积血最早数值': 'HCT',
    '平均红细胞体积血fl最早数值': 'MCV',
    '平均血红蛋白含量血pg最早数值': 'MCH',
    '平均血红蛋白浓度血gL最早数值': 'MCHC',
    '红细胞分布宽度血最早数值': 'RDW',
    '血小板计数血10E9L最早数值': 'PLT',
    '血小板压积血最早数值': 'PCT',
    '血小板平均体积血fl最早数值': 'MPV',
    '血小板分布宽度血fl最早数值': 'PDW',
    'C反应蛋白血mgL最早数值': 'CRP',
    '超敏C反应蛋白血mgL最早数值': 'hs-CRP',
    # Composite Inflammatory Indices (新增)
    'SII': 'SII',
    'NLR': 'NLR',
    'PLR': 'PLR',
    'NPAR': 'NPAR',
    # Metabolic & Renal
    '尿素UreammolL最早数值': 'Urea',
    '肌酐CrumolL最早数值': 'Cr',
    '尿酸UAumolL最早数值': 'UA',
    '肾小球滤过率Crmlmin最早数值': 'GFR',
    '总胆固醇TCmmolL最早数值': 'TC',
    '甘油三酯TGmmolL最早数值': 'TG',
    '高密度脂蛋白胆固醇HDLcmmolL最早数值': 'HDL',
    '低密度脂蛋白胆固醇LDLcmmolL最早数值': 'LDL',
    '载脂蛋白AⅠapoAⅠgL最早数值': 'apoA1',
    '载脂蛋白BapoBgL最早数值': 'apoB',
    '肌酸激酶测定CKUL最早数值': 'CK',
    '乳酸脱氢酶LDHUL最早数值': 'LDH',
    'α羟丁酸脱氢酶（HBDH）UL最早数值': 'HBDH',
    # Electrolytes
    '钾（K）mmolL最早数值': 'K',
    '钠（Na）mmolL最早数值': 'Na',
    '氯（Cl）mmolL最早数值': 'Cl',
    '总钙测定CammolL最早数值': 'Ca',
    '无机磷PmmolL最早数值': 'P',
    # Thyroid
    '三碘甲状原氨酸T3nmolL最早数值': 'T3',
    '甲状腺素T4nmolL最早数值': 'T4',
    '游离三碘甲状原氨酸FT3pmolL最早数值': 'FT3',
    '游离甲状腺素FT4pmolL最早数值': 'FT4',
    '促甲状腺激素TSHμIUmL最早数值': 'TSH',
    # Urinalysis
    '比重尿最早数值': 'USG',
    'pH值尿最早数值': 'Urine pH',
}

# 明确排除的列（衍生比值和无效列）
EXCLUDE_EXACT = [
    '丁酸4菌比总菌群',
    '丁酸4菌比2致病菌',
    '丁酸2菌比2致病菌',
    '促炎菌2比抗炎菌6',
]

# 排除包含这些关键词的列
EXCLUDE_PATTERNS = [
    '分组', 'target', '姓名', '科室', '入院', '出院', '病历号', '诊断',
    '肠杆菌', '肠球菌', '乳酸菌', '双歧杆菌', '类杆菌', '奇异菌', '丁酸梭菌',
    '柔嫩梭菌', '直肠真杆菌', '普拉梭菌', '菌群总数', '对数',
    '身高', '体重', 'BE值', '病因分类',
    'Enterococcus_Eubacterium_ratio', '颜色', '性状', '吞噬', '油滴', '夏雷登',
    '脓细胞', '霉菌', '不消化', '隐血试验', '胆红素尿', '酮体', '蛋白质尿',
    '隐血尿', '尿胆原', '亚硝酸盐', '红细胞尿', '白细胞尿', '细菌尿',
    '抗甲状腺', '凝血酶原时间正常对照', 'Unnamed',
]

# 绘图参数
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 10,
                     'figure.dpi': 300})


def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def get_clinical_columns(df):
    """提取临床指标列，排除指定列和衍生比值"""
    clinical_cols = []
    for col in df.columns:
        # 精确排除
        if col in EXCLUDE_EXACT:
            continue
        # 模式排除
        if any(pat in col for pat in EXCLUDE_PATTERNS):
            continue
        # 必须是在映射字典中定义的指标
        if col not in CLINICAL_EN_MAP:
            continue
        clinical_cols.append(col)
    return clinical_cols


def apply_fdr_correction(pvalue_df):
    pvals = pvalue_df.values.flatten()
    mask = ~np.isnan(pvals)
    if mask.sum() == 0:
        return pvalue_df.copy()
    _, p_corr, _, _ = multipletests(pvals[mask], method='fdr_bh')
    out = np.full_like(pvals, np.nan)
    out[mask] = p_corr
    return pd.DataFrame(out.reshape(pvalue_df.shape),
                        index=pvalue_df.index, columns=pvalue_df.columns)


def calculate_correlations(df, group_name=None):
    if group_name:
        data = df[df['分组'] == group_name].copy()
        print(f"\n=== {group_name} (n={len(data)}) ===")
    else:
        data = df.copy()
        group_name = 'Overall'
        print(f"\n=== Overall (n={len(data)}) ===")

    # 提取菌群数据
    bacteria_data = {}
    for eng_name, cn_col in BACTERIA_COLS.items():
        if cn_col in data.columns:
            bacteria_data[eng_name] = pd.to_numeric(data[cn_col], errors='coerce')

    # 提取临床指标
    clinical_cn = get_clinical_columns(data)
    clinical_en = [CLINICAL_EN_MAP[cn] for cn in clinical_cn]
    print(f"Identified {len(clinical_cn)} clinical parameters")

    corr = pd.DataFrame(index=list(bacteria_data.keys()), columns=clinical_en, dtype=float)
    p_raw = pd.DataFrame(index=list(bacteria_data.keys()), columns=clinical_en, dtype=float)

    for bac_name, bac_vals in bacteria_data.items():
        for cn_col, en_col in zip(clinical_cn, clinical_en):
            clin_vals = pd.to_numeric(data[cn_col], errors='coerce')
            valid = ~(bac_vals.isna() | clin_vals.isna())
            if valid.sum() >= 5:
                r, p = spearmanr(bac_vals[valid], clin_vals[valid])
                corr.loc[bac_name, en_col] = r
                p_raw.loc[bac_name, en_col] = p
            else:
                corr.loc[bac_name, en_col] = np.nan
                p_raw.loc[bac_name, en_col] = np.nan

    p_fdr = apply_fdr_correction(p_raw)
    return corr, p_raw, p_fdr, clinical_en


def filter_significant(corr, p_fdr, p_thresh=0.05):
    sig = []
    for col in corr.columns:
        if (p_fdr[col] < p_thresh).any():
            sig.append(col)
    print(f"  → {len(sig)} indicators with FDR < {p_thresh}")
    return sig


def plot_filtered_heatmap(corr, p_fdr, group_name, output_dir, figsize=(14, 12)):
    sig_inds = filter_significant(corr, p_fdr)
    if not sig_inds:
        print(f"  No significant indicators for {group_name}")
        return None

    corr_filt = corr[sig_inds]
    p_filt = p_fdr[sig_inds]

    annot = corr_filt.copy().astype(object)
    for i in range(corr_filt.shape[0]):
        for j in range(corr_filt.shape[1]):
            p = p_filt.iloc[i, j]
            if pd.isna(p):
                annot.iloc[i, j] = ""
            elif p < 0.001:
                annot.iloc[i, j] = "***"
            elif p < 0.01:
                annot.iloc[i, j] = "**"
            elif p < 0.05:
                annot.iloc[i, j] = "*"
            else:
                annot.iloc[i, j] = ""

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_filt.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    for i in range(corr_filt.shape[0]):
        for j in range(corr_filt.shape[1]):
            if annot.iloc[i, j]:
                ax.text(j, i, annot.iloc[i, j], ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')

    ax.set_xticks(range(len(sig_inds)))
    ax.set_xticklabels(sig_inds, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(corr_filt.index)))
    ax.set_yticklabels(corr_filt.index, fontsize=10)
    ax.set_title(f"{group_name} (FDR < 0.05, {len(sig_inds)} indicators)", fontsize=14, pad=15)
    ax.set_xlabel("Clinical Indicators", fontsize=12, labelpad=10)
    ax.set_ylabel("Gut Microbiota", fontsize=12, labelpad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ', size=11)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"Heatmap_{group_name}_filtered.png")
    out_eps = os.path.join(output_dir, f"Heatmap_{group_name}_filtered.eps")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_eps, format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Heatmap saved: {out_png}")
    return corr_filt, p_filt


def save_results(all_results, output_dir):
    with pd.ExcelWriter(os.path.join(output_dir, 'Supplementary_Table_S2_Correlation_Matrices.xlsx')) as writer:
        for group, (corr, p_raw, p_fdr, _) in all_results.items():
            sheet_prefix = group.replace(' ', '_')
            corr.to_excel(writer, sheet_name=f'{sheet_prefix}_Spearman_r')
            p_fdr.to_excel(writer, sheet_name=f'{sheet_prefix}_FDR_P')


def main():
    output_dir = create_output_dir()
    print(f"Output: {output_dir}")

    df = pd.read_excel(FILE_PATH)
    print(f"Data: {len(df)} rows")

    # 处理性别为数值（1→男，2→女）
    if '性别' in df.columns:
        df['性别'] = df['性别'].apply(lambda x: 1 if str(x).strip() == '1' else 0)

    groups = ['Overall'] + [g for g in df['分组'].unique() if pd.notna(g)]
    all_results = {}

    for group in groups:
        if group == 'Overall':
            corr, p_raw, p_fdr, en_cols = calculate_correlations(df)
        else:
            corr, p_raw, p_fdr, en_cols = calculate_correlations(df, group)
        all_results[group] = (corr, p_raw, p_fdr, en_cols)
        plot_filtered_heatmap(corr, p_fdr, group, output_dir)

    save_results(all_results, output_dir)

    # 单独输出Overall筛选后数据
    overall_corr, overall_p_fdr = all_results['Overall'][0], all_results['Overall'][2]
    sig = filter_significant(overall_corr, overall_p_fdr)
    overall_corr[sig].to_excel(os.path.join(output_dir, 'Figure2_Spearman_r_filtered.xlsx'))
    overall_p_fdr[sig].to_excel(os.path.join(output_dir, 'Figure2_FDR_P_filtered.xlsx'))

    print(f"\nDone! Overall retained {len(sig)} indicators.")


if __name__ == "__main__":
    main()