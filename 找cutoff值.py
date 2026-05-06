import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import warnings
import os

warnings.filterwarnings('ignore')

# 设置matplotlib为非交互模式
plt.ioff()

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置EPS兼容模式 - 禁用透明度
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# 文件路径
file_path = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组/汇总947.xlsx"
output_dir = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组"

print("正在读取数据...")
data = pd.read_excel(file_path)
print(f"数据读取成功! 形状: {data.shape}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)


class DiseaseStageRatioAnalyzer:
    """分析疾病阶段的肠球菌/直肠真杆菌比值"""

    def __init__(self):
        self.stage_data = {}
        self.ratio_stats = {}
        self.cutoff_results = {}

    def rename_columns(self, df):
        """重命名菌群列为英文"""
        df_renamed = df.copy()

        # 菌群重命名映射
        bacteria_rename_map = {
            '肠球菌对数': 'Enterococcus',
            '直肠真杆菌最早数值对数': 'Eubacterium rectale'
        }

        for old_name, new_name in bacteria_rename_map.items():
            if old_name in df_renamed.columns:
                df_renamed.rename(columns={old_name: new_name}, inplace=True)
                print(f"重命名: {old_name} → {new_name}")

        return df_renamed

    def preprocess_data(self, df):
        """数据预处理"""
        df_clean = df.copy()

        # 重命名列
        df_clean = self.rename_columns(df_clean)

        # 处理目标变量
        if '分组' in df_clean.columns:
            initial_len = len(df_clean)
            df_clean = df_clean.dropna(subset=['分组'])
            print(f"删除目标变量缺失的样本: {initial_len - len(df_clean)} 个")

        # 创建肠球菌/直肠真杆菌比值
        df_clean = self.create_enterococcus_eubacterium_ratio(df_clean)

        # 处理数值型缺失值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)

        print(f"预处理后数据形状: {df_clean.shape}")
        return df_clean

    def create_enterococcus_eubacterium_ratio(self, df):
        """创建肠球菌/直肠真杆菌比值"""
        df_ratio = df.copy()

        # 检查肠球菌和直肠真杆菌是否存在
        if 'Enterococcus' in df_ratio.columns and 'Eubacterium rectale' in df_ratio.columns:
            # 避免除零错误
            df_ratio['Enterococcus_Eubacterium_ratio'] = df_ratio['Enterococcus'] / (
                    df_ratio['Eubacterium rectale'].replace(0, np.nan) + 1e-10)
            # 删除无穷大的值
            df_ratio = df_ratio[~df_ratio['Enterococcus_Eubacterium_ratio'].isin([np.inf, -np.inf])]
            print("已创建比值: Enterococcus/E.rectale ratio")
        else:
            print("警告: 未找到肠球菌或直肠真杆菌数据，无法创建比值")

        return df_ratio

    def get_stage_names(self, df):
        """获取疾病阶段名称映射"""
        # 假设分组编码为: 2=肝硬化代偿期, 3=肝硬化失代偿期, 4=ACLF
        stage_mapping = {
            2: "Compensated_Cirrhosis",
            3: "Decompensated_Cirrhosis",
            4: "ACLF"
        }

        # 检查实际存在的分组
        actual_groups = sorted(df['分组'].unique())
        print(f"数据中存在的分组: {actual_groups}")

        return stage_mapping

    def calculate_stage_ratios(self, df):
        """计算各疾病阶段的比值统计"""
        df_processed = self.preprocess_data(df)
        stage_mapping = self.get_stage_names(df_processed)

        print(f"\n{'=' * 60}")
        print("各疾病阶段肠球菌/直肠真杆菌比值分析")
        print(f"{'=' * 60}")

        all_stats = []

        for stage_num, stage_name in stage_mapping.items():
            if stage_num not in df_processed['分组'].unique():
                print(f"警告: 分组 {stage_num} ({stage_name}) 在数据中不存在")
                continue

            # 筛选该阶段的样本
            stage_df = df_processed[df_processed['分组'] == stage_num]

            if 'Enterococcus_Eubacterium_ratio' not in stage_df.columns:
                print(f"警告: 分组 {stage_num} 中未找到比值数据")
                continue

            ratio_data = stage_df['Enterococcus_Eubacterium_ratio'].dropna()

            if len(ratio_data) == 0:
                print(f"警告: 分组 {stage_num} 中无有效比值数据")
                continue

            # 计算统计量
            stats_dict = {
                'Stage_Number': stage_num,
                'Stage_Name': stage_name,
                'Sample_Size': len(ratio_data),
                'Mean_Ratio': ratio_data.mean(),
                'Median_Ratio': ratio_data.median(),
                'Std_Ratio': ratio_data.std(),
                'Min_Ratio': ratio_data.min(),
                'Max_Ratio': ratio_data.max(),
                'Q1_Ratio': ratio_data.quantile(0.25),
                'Q3_Ratio': ratio_data.quantile(0.75),
                'IQR_Ratio': ratio_data.quantile(0.75) - ratio_data.quantile(0.25)
            }

            all_stats.append(stats_dict)

            # 保存原始数据用于绘图
            self.stage_data[stage_name] = ratio_data

            print(f"\n{stage_name} (分组 {stage_num}):")
            print(f"  样本数: {len(ratio_data)}")
            print(f"  均值: {ratio_data.mean():.4f}")
            print(f"  中位数: {ratio_data.median():.4f}")
            print(f"  标准差: {ratio_data.std():.4f}")
            print(f"  范围: [{ratio_data.min():.4f}, {ratio_data.max():.4f}]")
            print(f"  四分位距: {stats_dict['IQR_Ratio']:.4f}")

        # 创建统计表格
        stats_df = pd.DataFrame(all_stats)
        self.ratio_stats = stats_df

        return stats_df

    def plot_stage_comparison(self):
        """绘制各疾病阶段的比值比较图"""
        if not self.stage_data:
            print("没有可用的阶段数据进行绘图")
            return

        # 创建综合图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enterococcus/Eubacterium rectale Ratio Across Disease Stages',
                     fontsize=16, fontweight='bold')

        # 准备数据
        stage_names = list(self.stage_data.keys())
        ratio_data = list(self.stage_data.values())

        # EPS兼容的颜色 - 纯色，不透明
        colors = ['#ADD8E6', '#90EE90', '#F0E68C']  # lightblue, lightgreen, lightcoral替代色

        # 1. 箱线图
        box_plot = axes[0, 0].boxplot(ratio_data, labels=stage_names, patch_artist=True)
        # 设置箱形图颜色
        for patch, color in zip(box_plot['boxes'], colors[:len(stage_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)  # EPS兼容，完全不透明
        axes[0, 0].set_ylabel('Enterococcus/E.rectale Ratio')
        axes[0, 0].set_title('Box Plot by Disease Stage', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 小提琴图 - EPS不兼容，改为使用箱线图替代
        # 为了EPS兼容，我们不用violinplot，用增强的箱线图
        for i, data in enumerate(ratio_data):
            # 添加散点表示数据分布
            x_pos = np.random.normal(i + 1, 0.04, size=len(data))
            axes[0, 1].scatter(x_pos, data, alpha=0.6, s=20, color=colors[i])

        # 仍绘制箱线图表示统计
        bp = axes[0, 1].boxplot(ratio_data, positions=range(1, len(ratio_data) + 1), widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(stage_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)

        axes[0, 1].set_xticks(range(1, len(stage_names) + 1))
        axes[0, 1].set_xticklabels(stage_names)
        axes[0, 1].set_ylabel('Enterococcus/E.rectale Ratio')
        axes[0, 1].set_title('Distribution by Disease Stage', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 均值柱状图
        means = [data.mean() for data in ratio_data]
        stds = [data.std() for data in ratio_data]
        x_pos = np.arange(len(stage_names))

        bars = axes[0, 2].bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(stage_names)],
                              alpha=1.0, edgecolor='black', linewidth=0.5)
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(stage_names)
        axes[0, 2].set_ylabel('Mean Ratio')
        axes[0, 2].set_title('Mean Ratio with Standard Deviation', fontsize=14, fontweight='bold')
        axes[0, 2].grid(axis='y', alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值标签
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            axes[0, 2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + stds[i] + 0.1,
                            f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. 中位数柱状图
        medians = [data.median() for data in ratio_data]
        bars_median = axes[1, 0].bar(x_pos, medians, color=colors[:len(stage_names)],
                                     alpha=1.0, edgecolor='black', linewidth=0.5)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(stage_names)
        axes[1, 0].set_ylabel('Median Ratio')
        axes[1, 0].set_title('Median Ratio by Disease Stage', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值标签
        for i, (bar, median_val) in enumerate(zip(bars_median, medians)):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                            f'{median_val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 5. 样本数量图
        sample_sizes = [len(data) for data in ratio_data]
        bars_samples = axes[1, 1].bar(x_pos, sample_sizes, color=colors[:len(stage_names)],
                                      alpha=1.0, edgecolor='black', linewidth=0.5)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(stage_names)
        axes[1, 1].set_ylabel('Sample Size')
        axes[1, 1].set_title('Sample Size by Disease Stage', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值标签
        for i, (bar, size) in enumerate(zip(bars_samples, sample_sizes)):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                            f'{size}', ha='center', va='bottom', fontweight='bold')

        # 6. 趋势图（按疾病严重程度）
        if len(means) >= 2:
            # 假设阶段顺序代表疾病严重程度递增
            axes[1, 2].plot(range(len(means)), means, 'o-', linewidth=3, markersize=8,
                            color='darkred', label='Mean Ratio')
            axes[1, 2].plot(range(len(medians)), medians, 's-', linewidth=3, markersize=8,
                            color='darkblue', label='Median Ratio')
            axes[1, 2].set_xticks(range(len(stage_names)))
            axes[1, 2].set_xticklabels(stage_names)
            axes[1, 2].set_ylabel('Ratio Value')
            axes[1, 2].set_title('Ratio Trend Across Disease Progression', fontsize=14, fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].grid(alpha=0.3)
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            # 如果没有足够的数据点，显示统计摘要
            axes[1, 2].axis('off')
            summary_text = "Statistical Summary:\n\n"
            for stage_name, data in self.stage_data.items():
                summary_text += f"{stage_name}:\n"
                summary_text += f"  n={len(data)}, mean={data.mean():.3f}\n"
                summary_text += f"  median={data.median():.3f}, std={data.std():.3f}\n\n"
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        # 保存图片 - 使用纯色以兼容EPS
        plt.savefig(f'{output_dir}/disease_stage_ratio_comparison.png',
                    dpi=300, bbox_inches='tight')
        # 保存EPS格式 - 禁用透明度
        plt.savefig(f'{output_dir}/disease_stage_ratio_comparison.eps',
                    dpi=300, bbox_inches='tight', format='eps',
                    facecolor='white', edgecolor='none',
                    transparent=False)
        print(f"已保存图形到: {output_dir}/disease_stage_ratio_comparison.eps")
        plt.close(fig)

    def perform_statistical_tests(self):
        """执行组间统计检验"""
        if len(self.stage_data) < 2:
            print("需要至少2个组才能进行统计检验")
            return

        print(f"\n{'=' * 60}")
        print("组间统计检验")
        print(f"{'=' * 60}")

        stage_names = list(self.stage_data.keys())
        ratio_data = list(self.stage_data.values())

        # 正态性检验
        print("\n正态性检验 (Shapiro-Wilk test):")
        for i, (stage_name, data) in enumerate(zip(stage_names, ratio_data)):
            if len(data) >= 3 and len(data) <= 5000:  # Shapiro-Wilk test的样本量限制
                stat, p_value = stats.shapiro(data)
                print(f"  {stage_name}: W={stat:.4f}, p={p_value:.4f} {'(正态)' if p_value > 0.05 else '**(非正态)**'}")
            else:
                print(f"  {stage_name}: 样本量不适合Shapiro-Wilk检验")

        # 方差齐性检验
        print("\n方差齐性检验 (Levene test):")
        if len(ratio_data) >= 2:
            stat, p_value = stats.levene(*ratio_data)
            print(f"  所有组: W={stat:.4f}, p={p_value:.4f} {'(方差齐)' if p_value > 0.05 else '**(方差不齐)**'}")

        # Mann-Whitney U检验（两个组比较）
        print("\nMann-Whitney U检验 (两两比较):")
        if len(ratio_data) >= 2:
            # 只比较相邻的两个组
            for i in range(len(ratio_data) - 1):
                stat, p_value = stats.mannwhitneyu(ratio_data[i], ratio_data[i + 1])
                print(
                    f"  {stage_names[i]} vs {stage_names[i + 1]}: U={stat:.4f}, p={p_value:.4f} {'(无显著差异)' if p_value > 0.05 else '**(有显著差异)**'}")

    def calculate_auc_ci(self, y_true, y_score, n_bootstrap=1000, alpha=0.95):
        """计算AUC的置信区间（使用bootstrap方法）"""
        auc_values = []
        n = len(y_true)

        # 设置随机种子以确保可重复性
        np.random.seed(42)

        for _ in range(n_bootstrap):
            # Bootstrap重采样
            indices = np.random.choice(range(n), n, replace=True)
            y_true_boot = y_true[indices]
            y_score_boot = y_score[indices]

            # 计算AUC
            try:
                auc_boot = roc_auc_score(y_true_boot, y_score_boot)
                auc_values.append(auc_boot)
            except:
                continue

        if auc_values:
            auc_values = np.array(auc_values)
            # 计算百分位数置信区间
            lower_percentile = (1 - alpha) / 2 * 100
            upper_percentile = (alpha + (1 - alpha) / 2) * 100

            ci_lower = np.percentile(auc_values, lower_percentile)
            ci_upper = np.percentile(auc_values, upper_percentile)

            return ci_lower, ci_upper
        else:
            return None, None

    def analyze_cutoff_values(self, df):
        """分析两个模型的cutoff值: CC-DC, DC-ACLF"""
        print(f"\n{'=' * 60}")
        print("寻找两个模型的cutoff值")
        print(f"{'=' * 60}")

        # 定义两个模型
        models = {
            'CC_DC': {
                'case': 3,
                'control': 2,
                'name': 'Compensated_Cirrhosis vs Decompensated_Cirrhosis',
                'display_name': 'CC vs DC'
            },
            'DC_ACLF': {
                'case': 4,
                'control': 3,
                'name': 'Decompensated_Cirrhosis vs ACLF',
                'display_name': 'DC vs ACLF'
            }
        }

        cutoff_results = {}

        for model_key, model_config in models.items():
            print(f"\n分析模型: {model_config['name']}")

            # 筛选相关样本
            case_value = model_config['case']
            control_value = model_config['control']

            model_data = df[df['分组'].isin([case_value, control_value])].copy()

            if len(model_data) < 20:
                print(f"  样本量不足 ({len(model_data)})，跳过模型: {model_config['name']}")
                continue

            # 检查比值是否存在
            if 'Enterococcus_Eubacterium_ratio' not in model_data.columns:
                print(f"  未找到Enterococcus_Eubacterium_ratio特征，跳过模型: {model_config['name']}")
                continue

            # 创建二分类标签
            model_data['target'] = (model_data['分组'] == case_value).astype(int)
            y = model_data['target']

            print(f"  病例组(分组{case_value}): {y.sum()}, 对照组(分组{control_value}): {len(y) - y.sum()}")

            # 获取比值数据
            ratio_data = model_data['Enterococcus_Eubacterium_ratio'].values
            y_true = y.values

            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(y_true, ratio_data)

            # 计算Youden指数
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]
            best_tpr = tpr[best_idx]
            best_fpr = fpr[best_idx]

            # 计算AUC及其置信区间
            auc_score = roc_auc_score(y_true, ratio_data)
            auc_ci_lower, auc_ci_upper = self.calculate_auc_ci(y_true, ratio_data)

            # 计算cutoff的敏感性、特异性
            # 计算基于该cutoff的分类性能
            y_pred = (ratio_data >= best_threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)

            # 计算混淆矩阵
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 计算敏感性、特异性的95%置信区间（精确二项置信区间）
            sens_ci_lower, sens_ci_upper = self.calculate_binomial_ci(tp, tp + fn)
            spec_ci_lower, spec_ci_upper = self.calculate_binomial_ci(tn, tn + fp)

            result = {
                'cutoff': best_threshold,
                'auc': auc_score,
                'auc_ci_lower': auc_ci_lower,
                'auc_ci_upper': auc_ci_upper,
                'sensitivity': sensitivity,
                'sensitivity_ci_lower': sens_ci_lower,
                'sensitivity_ci_upper': sens_ci_upper,
                'specificity': specificity,
                'specificity_ci_lower': spec_ci_lower,
                'specificity_ci_upper': spec_ci_upper,
                'accuracy': accuracy,
                'youden_index': youden_index[best_idx],
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'case_samples': y.sum(),
                'control_samples': len(y) - y.sum(),
                'total_samples': len(y),
                'display_name': model_config['display_name'],
                'fpr': fpr,
                'tpr': tpr
            }

            cutoff_results[model_key] = result

            print(f"  最佳cutoff值: {best_threshold:.4f}")
            print(f"  AUC: {auc_score:.4f} (95% CI: {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")
            print(f"  敏感性: {sensitivity:.4f} (95% CI: {sens_ci_lower:.4f}-{sens_ci_upper:.4f})")
            print(f"  特异性: {specificity:.4f} (95% CI: {spec_ci_lower:.4f}-{spec_ci_upper:.4f})")
            print(f"  准确率: {accuracy:.4f}")
            print(f"  Youden指数: {youden_index[best_idx]:.4f}")
            print(f"  混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

            # 绘制ROC曲线
            self.plot_model_roc_curve(model_config['name'], model_key, result)

        self.cutoff_results = cutoff_results
        return cutoff_results

    def calculate_binomial_ci(self, k, n, alpha=0.95):
        """计算二项分布比例的置信区间（Wilson score interval）"""
        if n == 0:
            return 0, 0

        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - alpha) / 2)  # 对于95% CI，z=1.96
        p = k / n
        denominator = 1 + z ** 2 / n
        centre_adjusted_probability = p + z ** 2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)

        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

        return max(0, lower_bound), min(1, upper_bound)

    def plot_model_roc_curve(self, model_name, model_key, result):
        """绘制单个模型的ROC曲线"""
        # 设置图形大小和布局
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])

        ax1 = fig.add_subplot(gs[0, 0])  # ROC曲线
        ax2 = fig.add_subplot(gs[0, 1])  # 统计信息

        # 主图：ROC曲线
        fpr, tpr = result['fpr'], result['tpr']
        auc_score = result['auc']
        auc_ci_lower, auc_ci_upper = result['auc_ci_lower'], result['auc_ci_upper']
        cutoff = result['cutoff']
        sensitivity = result['sensitivity']
        specificity = result['specificity']
        sens_ci_lower, sens_ci_upper = result['sensitivity_ci_lower'], result['sensitivity_ci_upper']
        spec_ci_lower, spec_ci_upper = result['specificity_ci_lower'], result['specificity_ci_upper']

        # 绘制ROC曲线 - 使用纯色，不透明
        ax1.plot(fpr, tpr, color='darkorange', lw=3, solid_capstyle='round', solid_joinstyle='round')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

        # 标记最佳cutoff点
        best_fpr = 1 - specificity
        ax1.plot(best_fpr, sensitivity, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1)

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax1.set_title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='-', linewidth=0.5)

        # 添加AUC和cutoff信息到主图
        auc_text = f'AUC = {auc_score:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f})'
        cutoff_text = f'Cutoff = {cutoff:.3f}'
        sens_text = f'Sensitivity = {sensitivity:.3f} (95% CI: {sens_ci_lower:.3f}-{sens_ci_upper:.3f})'
        spec_text = f'Specificity = {specificity:.3f} (95% CI: {spec_ci_lower:.3f}-{spec_ci_upper:.3f})'

        info_text = f"{auc_text}\n{cutoff_text}\n{sens_text}\n{spec_text}"

        # 创建文本框
        from matplotlib.patches import Rectangle
        text_box = Rectangle((0.05, 0.95), 0.4, 0.15, transform=ax1.transAxes,
                             facecolor='white', edgecolor='black', linewidth=0.5)
        ax1.add_patch(text_box)

        ax1.text(0.07, 0.97, info_text, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace')

        # 添加图例
        ax1.plot([], [], color='darkorange', lw=3, label='ROC curve')
        ax1.plot([], [], 'ro', markersize=10, markeredgecolor='black',
                 markeredgewidth=1, label='Optimal cutoff point')
        ax1.legend(loc="lower right", fontsize=10, frameon=True, framealpha=1.0, edgecolor='black')

        # 副图：cutoff值的详细信息
        ax2.axis('off')
        summary_text = f"Model: {model_name}\n"
        summary_text += "=" * 40 + "\n\n"
        summary_text += f"Optimal Cutoff: {cutoff:.4f}\n\n"
        summary_text += f"AUC: {auc_score:.4f}\n"
        summary_text += f"95% CI: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]\n\n"
        summary_text += f"Sensitivity: {sensitivity:.4f}\n"
        summary_text += f"95% CI: [{sens_ci_lower:.4f}, {sens_ci_upper:.4f}]\n\n"
        summary_text += f"Specificity: {specificity:.4f}\n"
        summary_text += f"95% CI: [{spec_ci_lower:.4f}, {spec_ci_upper:.4f}]\n\n"
        summary_text += f"Youden Index: {result['youden_index']:.4f}\n\n"
        summary_text += "Confusion Matrix:\n"
        summary_text += f"  TP: {result['tp']}, TN: {result['tn']}\n"
        summary_text += f"  FP: {result['fp']}, FN: {result['fn']}\n\n"
        summary_text += f"Sample Size:\n"
        summary_text += f"  Case: {result['case_samples']}\n"
        summary_text += f"  Control: {result['control_samples']}\n"
        summary_text += f"  Total: {result['total_samples']}\n"

        # 在文本框内显示文本
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top',
                 fontfamily='monospace', linespacing=1.5)

        plt.suptitle(f'ROC Analysis: {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片 - EPS兼容格式
        plt.savefig(f'{output_dir}/{model_key}_ROC_curve_with_CI.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{output_dir}/{model_key}_ROC_curve_with_CI.eps',
                    dpi=300, bbox_inches='tight', format='eps',
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"已保存ROC曲线图到: {output_dir}/{model_key}_ROC_curve_with_CI.eps")
        plt.close(fig)

    def plot_cutoff_comparison(self):
        """绘制两个模型cutoff值的比较图"""
        if not self.cutoff_results:
            print("没有cutoff结果可用于绘图")
            return

        # 创建综合图表 - EPS兼容
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparison of Cutoff Values for CC-DC and DC-ACLF Models',
                     fontsize=16, fontweight='bold')

        models = list(self.cutoff_results.keys())

        # 设置显示名称
        display_names = [self.cutoff_results[model]['display_name'] for model in models]
        x_pos = np.arange(len(models))

        # EPS兼容的颜色 - 纯色，不透明
        colors = ['#90EE90', '#F08080']  # lightgreen, lightcoral

        # 1. Cutoff值比较
        cutoff_values = [self.cutoff_results[model]['cutoff'] for model in models]
        bars1 = axes[0, 0].bar(x_pos, cutoff_values, color=colors, width=0.6,
                               edgecolor='black', linewidth=0.5)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(display_names, fontsize=12)
        axes[0, 0].set_ylabel('Cutoff Value', fontsize=12)
        axes[0, 0].set_title('Cutoff Values by Model', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

        # 添加数值标签
        for i, bar in enumerate(bars1):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(cutoff_values) * 0.01,
                            f'{cutoff_values[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 2. AUC值比较
        auc_values = [self.cutoff_results[model]['auc'] for model in models]
        auc_ci_lower = [self.cutoff_results[model]['auc_ci_lower'] for model in models]
        auc_ci_upper = [self.cutoff_results[model]['auc_ci_upper'] for model in models]

        # 计算误差条
        auc_errors = [[auc_values[i] - auc_ci_lower[i], auc_ci_upper[i] - auc_values[i]]
                      for i in range(len(auc_values))]

        bars2 = axes[0, 1].bar(x_pos, auc_values, yerr=np.array(auc_errors).T, capsize=5,
                               color=colors, width=0.6, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(display_names, fontsize=12)
        axes[0, 1].set_ylabel('AUC', fontsize=12)
        axes[0, 1].set_title('AUC Values with 95% Confidence Intervals', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        axes[0, 1].set_ylim(0, 1.0)

        # 添加数值标签
        for i, bar in enumerate(bars2):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                            f'{auc_values[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            ci_text = f'[{auc_ci_lower[i]:.3f}-{auc_ci_upper[i]:.3f}]'
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                            ci_text, ha='center', va='bottom', fontsize=8)

        # 3. 敏感性和特异性比较
        sensitivity_values = [self.cutoff_results[model]['sensitivity'] for model in models]
        specificity_values = [self.cutoff_results[model]['specificity'] for model in models]

        width = 0.35
        bars_sens = axes[1, 0].bar(x_pos - width / 2, sensitivity_values, width, label='Sensitivity',
                                   color='#ADD8E6', edgecolor='black', linewidth=0.5)  # lightblue
        bars_spec = axes[1, 0].bar(x_pos + width / 2, specificity_values, width, label='Specificity',
                                   color='#FFA07A', edgecolor='black', linewidth=0.5)  # lightsalmon

        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(display_names, fontsize=12)
        axes[1, 0].set_ylabel('Value', fontsize=12)
        axes[1, 0].set_title('Sensitivity and Specificity', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10, frameon=True, framealpha=1.0, edgecolor='black')
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        axes[1, 0].set_ylim(0, 1.0)

        # 添加数值标签
        for i, (sens, spec) in enumerate(zip(sensitivity_values, specificity_values)):
            axes[1, 0].text(x_pos[i] - width / 2, sens + 0.02, f'{sens:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            axes[1, 0].text(x_pos[i] + width / 2, spec + 0.02, f'{spec:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 4. Youden指数比较
        youden_values = [self.cutoff_results[model]['youden_index'] for model in models]
        bars4 = axes[1, 1].bar(x_pos, youden_values, color=colors, width=0.6,
                               edgecolor='black', linewidth=0.5)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(display_names, fontsize=12)
        axes[1, 1].set_ylabel('Youden Index', fontsize=12)
        axes[1, 1].set_title('Youden Index (Sensitivity + Specificity - 1)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        axes[1, 1].set_ylim(0, 1.0)

        # 添加数值标签
        for i, bar in enumerate(bars4):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                            f'{youden_values[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        # 保存EPS格式 - 禁用透明度
        plt.savefig(f'{output_dir}/cutoff_comparison_two_models.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{output_dir}/cutoff_comparison_two_models.eps',
                    dpi=300, bbox_inches='tight', format='eps',
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"已保存比较图到: {output_dir}/cutoff_comparison_two_models.eps")
        plt.close(fig)

    def save_results(self):
        """保存结果到Excel文件"""
        if self.ratio_stats is not None and not self.ratio_stats.empty:
            with pd.ExcelWriter(f'{output_dir}/disease_stage_ratio_analysis.xlsx') as writer:
                # 保存基本统计
                self.ratio_stats.to_excel(writer, sheet_name='Basic_Statistics', index=False)

                # 保存cutoff结果
                if self.cutoff_results:
                    cutoff_data = []
                    for model_key, result in self.cutoff_results.items():
                        # 创建简化的结果字典，不包含numpy数组
                        simple_result = {}
                        for key, value in result.items():
                            if key not in ['fpr', 'tpr']:
                                simple_result[key] = value
                        cutoff_data.append(simple_result)

                    cutoff_df = pd.DataFrame(cutoff_data)
                    cutoff_df.to_excel(writer, sheet_name='Cutoff_Analysis', index=False)

                print(f"\n结果已保存:")
                print(f"  基本统计和cutoff分析: {output_dir}/disease_stage_ratio_analysis.xlsx")

        # 保存每个阶段的详细数据
        if self.stage_data:
            with pd.ExcelWriter(f'{output_dir}/disease_stage_ratio_detailed_data.xlsx') as detailed_writer:
                for stage_name, ratio_data in self.stage_data.items():
                    stage_df = pd.DataFrame({
                        'Enterococcus_Eubacterium_ratio': ratio_data
                    })
                    stage_df.to_excel(detailed_writer, sheet_name=stage_name, index=False)
            print(f"  详细数据: {output_dir}/disease_stage_ratio_detailed_data.xlsx")

    def run_complete_analysis(self, data):
        """运行完整分析"""
        print("正在计算各疾病阶段的肠球菌/直肠真杆菌比值...")

        # 计算各阶段比值统计
        stats_df = self.calculate_stage_ratios(data)

        if stats_df.empty:
            print("未能计算任何阶段的比值统计")
            return None

        # 绘制比较图
        self.plot_stage_comparison()

        # 执行统计检验
        self.perform_statistical_tests()

        # 分析两个模型的cutoff值
        df_processed = self.preprocess_data(data)
        self.analyze_cutoff_values(df_processed)

        # 绘制cutoff比较图
        self.plot_cutoff_comparison()

        # 保存结果
        self.save_results()

        return stats_df


# 主执行函数
def main():
    print("正在读取数据...")
    data = pd.read_excel(file_path)
    print(f"数据读取成功! 形状: {data.shape}")

    # 初始化分析器
    analyzer = DiseaseStageRatioAnalyzer()

    # 运行完整分析
    results = analyzer.run_complete_analysis(data)

    if results is not None:
        print(f"\n{'=' * 60}")
        print("分析完成!")
        print(f"{'=' * 60}")

        # 打印最终汇总
        print("\n各疾病阶段肠球菌/直肠真杆菌比值汇总:")
        for _, row in results.iterrows():
            print(f"\n{row['Stage_Name']} (分组 {row['Stage_Number']}):")
            print(f"  样本数: {row['Sample_Size']}")
            print(f"  均值: {row['Mean_Ratio']:.4f}")
            print(f"  中位数: {row['Median_Ratio']:.4f}")
            print(f"  标准差: {row['Std_Ratio']:.4f}")
            print(f"  范围: [{row['Min_Ratio']:.4f}, {row['Max_Ratio']:.4f}]")

        # 打印cutoff值汇总
        if analyzer.cutoff_results:
            print(f"\n两个模型的cutoff值汇总:")
            for model, result in analyzer.cutoff_results.items():
                print(f"\n{result['display_name']}:")
                print(f"  最佳cutoff值: {result['cutoff']:.4f}")
                print(f"  AUC: {result['auc']:.4f} (95% CI: {result['auc_ci_lower']:.4f}-{result['auc_ci_upper']:.4f})")
                print(
                    f"  敏感性: {result['sensitivity']:.4f} (95% CI: {result['sensitivity_ci_lower']:.4f}-{result['sensitivity_ci_upper']:.4f})")
                print(
                    f"  特异性: {result['specificity']:.4f} (95% CI: {result['specificity_ci_lower']:.4f}-{result['specificity_ci_upper']:.4f})")
                print(f"  准确率: {result['accuracy']:.4f}")
                print(f"  Youden指数: {result['youden_index']:.4f}")
                print(f"  混淆矩阵: TP={result['tp']}, TN={result['tn']}, FP={result['fp']}, FN={result['fn']}")
    else:
        print("分析失败，请检查数据格式")


if __name__ == "__main__":
    main()