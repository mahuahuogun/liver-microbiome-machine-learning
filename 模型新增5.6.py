import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

file_path = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组/汇总947.xlsx"
output_dir = "/Users/daimengting/Desktop/课题/重症肝病课题/师兄文章/python分组"

print("正在读取数据...")
data = pd.read_excel(file_path)
print(f"数据读取成功! 形状: {data.shape}")

os.makedirs(output_dir, exist_ok=True)


class OptimizedClinicalMicrobiomeModel:
    def __init__(self):
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.selected_features = []
        self.best_combination = {}
        self.clinical_only_accuracy = 0
        self.combined_accuracy = 0

    # ---------- 数据预处理 ----------
    def rename_columns(self, df):
        df_renamed = df.copy()
        clinical_rename_map = {
            '总胆红素最大值': 'TBil',
            '白蛋白（正）AlbgL最早数值': 'Alb',
            '凝血酶原时间血秒最早数值': 'PT'
        }
        bacteria_rename_map = {
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
        rename_map = {**clinical_rename_map, **bacteria_rename_map}
        for old_name, new_name in rename_map.items():
            if old_name in df_renamed.columns:
                df_renamed.rename(columns={old_name: new_name}, inplace=True)
                print(f"重命名: {old_name} → {new_name}")
        return df_renamed

    def preprocess_data(self, df):
        df_clean = df.copy()
        df_clean = self.rename_columns(df_clean)
        if '分组' in df_clean.columns:
            initial_len = len(df_clean)
            df_clean = df_clean.dropna(subset=['分组'])
            print(f"删除目标变量缺失的样本: {initial_len - len(df_clean)} 个")
        df_clean = self.create_bacteria_ratios(df_clean)
        df_clean = self.handle_non_numeric_data(df_clean)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        print(f"预处理后数据形状: {df_clean.shape}")
        return df_clean

    def handle_non_numeric_data(self, df):
        df_processed = df.copy()
        non_numeric_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        non_numeric_cols = [col for col in non_numeric_cols if col != '分组']
        for col in non_numeric_cols:
            if df_processed[col].nunique() <= 10:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            else:
                df_processed = df_processed.drop(columns=[col])
        return df_processed

    def create_bacteria_ratios(self, df):
        df_ratios = df.copy()
        pro_inflammatory = ['Enterococcus', 'Enterobacteria']
        anti_inflammatory = ['Bifidobacterium', 'Lactobacillus', 'Clostridium butyricum',
                             'Clostridium leptum', 'Eubacterium rectale', 'Faecalibacterium prausnitzii']
        butyrate_bacteria = ['Eubacterium rectale', 'Faecalibacterium prausnitzii',
                             'Clostridium leptum', 'Clostridium butyricum']
        specific_butyrate = ['Eubacterium rectale', 'Faecalibacterium prausnitzii']
        all_bacteria = pro_inflammatory + anti_inflammatory + ['Bacteroides', 'Atopobium cluster']

        available_pro = [f for f in pro_inflammatory if f in df_ratios.columns]
        available_anti = [f for f in anti_inflammatory if f in df_ratios.columns]
        available_butyrate = [f for f in butyrate_bacteria if f in df_ratios.columns]
        available_specific = [f for f in specific_butyrate if f in df_ratios.columns]
        available_all = [f for f in all_bacteria if f in df_ratios.columns]

        # 1. Pro/Anti-inflammatory
        if available_pro and available_anti:
            pro_sum = df_ratios[available_pro].sum(axis=1)
            anti_sum = df_ratios[available_anti].sum(axis=1)
            df_ratios['Pro_Anti_ratio'] = pro_sum / (anti_sum + 1e-10)
        # 2. Butyrate/Pro
        if available_butyrate and available_pro:
            buty_sum = df_ratios[available_butyrate].sum(axis=1)
            pro_sum = df_ratios[available_pro].sum(axis=1)
            df_ratios['Butyrate_Pro_ratio'] = buty_sum / (pro_sum + 1e-10)
        # 3. Butyrate/Total
        if available_butyrate and len(available_all) >= 3:
            buty_sum = df_ratios[available_butyrate].sum(axis=1)
            total_sum = df_ratios[available_all].sum(axis=1)
            df_ratios['Butyrate_Total_ratio'] = buty_sum / (total_sum + 1e-10)
        # 4. Specific-Butyrate/Total
        if available_specific and len(available_all) >= 3:
            spec_sum = df_ratios[available_specific].sum(axis=1)
            total_sum = df_ratios[available_all].sum(axis=1)
            df_ratios['Specific_Butyrate_Total_ratio'] = spec_sum / (total_sum + 1e-10)
        # 5. Specific-Butyrate/Pro
        if available_specific and available_pro:
            spec_sum = df_ratios[available_specific].sum(axis=1)
            pro_sum = df_ratios[available_pro].sum(axis=1)
            df_ratios['Specific_Butyrate_Pro_ratio'] = spec_sum / (pro_sum + 1e-10)
        # 6. E/Er ratio
        if 'Enterococcus' in df_ratios.columns and 'Eubacterium rectale' in df_ratios.columns:
            df_ratios['Enterococcus_Eubacterium_ratio'] = df_ratios['Enterococcus'] / (
                    df_ratios['Eubacterium rectale'] + 1e-10)

        # 删除中间计算列（若有残留）
        cols_to_drop = [c for c in df_ratios.columns if any(k in c for k in [
            'Pro_inflammatory_total', 'Anti_inflammatory_total', 'Butyrate_total',
            'Specific_butyrate_total', 'Total_bacteria'
        ])]
        df_ratios.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return df_ratios

    # ---------- 特征获取 ----------
    def get_core_clinical_features(self, df):
        clinical_features = []
        for variant in ['TBil', '总胆红素最大值', '总胆红素TBilumolL.1最早数值', '总胆红素']:
            if variant in df.columns:
                if variant != 'TBil':
                    df.rename(columns={variant: 'TBil'}, inplace=True)
                clinical_features.append('TBil')
                break
        for variant in ['Alb', '白蛋白（正）AlbgL最早数值', '白蛋白']:
            if variant in df.columns:
                if variant != 'Alb':
                    df.rename(columns={variant: 'Alb'}, inplace=True)
                clinical_features.append('Alb')
                break
        for variant in ['PT', '凝血酶原时间血秒最早数值', '凝血酶原时间']:
            if variant in df.columns:
                if variant != 'PT':
                    df.rename(columns={variant: 'PT'}, inplace=True)
                clinical_features.append('PT')
                break
        return clinical_features

    def get_bacteria_features(self, df):
        ratio_features = [
            'Pro_Anti_ratio', 'Butyrate_Pro_ratio', 'Butyrate_Total_ratio',
            'Specific_Butyrate_Total_ratio', 'Specific_Butyrate_Pro_ratio',
            'Enterococcus_Eubacterium_ratio'
        ]
        return [f for f in ratio_features if f in df.columns]

    def evaluate_features_with_accuracy(self, X, y, n_splits=5):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        auc = cross_val_score(rf, X, y, cv=StratifiedKFold(n_splits), scoring='roc_auc')
        acc = cross_val_score(rf, X, y, cv=StratifiedKFold(n_splits), scoring='accuracy')
        return {'auc': auc.mean(), 'accuracy': acc.mean(),
                'auc_std': auc.std(), 'accuracy_std': acc.std()}

    def find_optimal_feature_combination(self, X, y, clinical_features, bacteria_features):
        print(f"\n寻找最优特征组合...")
        best_auc = 0
        best_features = clinical_features.copy()
        auc_history = []
        accuracy_history = []

        X_clinical = X[clinical_features]
        clin_perf = self.evaluate_features_with_accuracy(X_clinical, y)
        auc_history.append({'num_bacteria': 0, 'auc': clin_perf['auc'], 'features': clinical_features})
        accuracy_history.append({'num_bacteria': 0, 'accuracy': clin_perf['accuracy'], 'features': clinical_features})
        print(f"Clinical only AUC: {clin_perf['auc']:.4f}, Accuracy: {clin_perf['accuracy']:.4f}")

        X_all = X[clinical_features + bacteria_features]
        rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_sel.fit(X_all, y)
        feat_imp = pd.DataFrame({
            'feature': clinical_features + bacteria_features,
            'importance': rf_sel.feature_importances_
        })
        bact_imp = feat_imp[feat_imp['feature'].isin(bacteria_features)].sort_values('importance', ascending=False)

        sorted_bact = bact_imp['feature'].tolist()
        for num in range(1, len(sorted_bact) + 1):
            selected = sorted_bact[:num]
            current_features = clinical_features + selected
            X_cur = X[current_features]
            perf = self.evaluate_features_with_accuracy(X_cur, y)
            auc_history.append({
                'num_bacteria': num, 'auc': perf['auc'],
                'features': current_features, 'bacteria_features': selected
            })
            accuracy_history.append({
                'num_bacteria': num, 'accuracy': perf['accuracy'],
                'features': current_features, 'bacteria_features': selected
            })
            if perf['auc'] > best_auc:
                best_auc = perf['auc']
                best_features = current_features
        best_comb = max(auc_history, key=lambda x: x['auc'])
        best_acc = max(accuracy_history, key=lambda x: x['accuracy'])
        print(f"最优组合: AUC={best_comb['auc']:.4f}, Accuracy={best_acc['accuracy']:.4f}")
        return best_comb, auc_history, accuracy_history, clin_perf['accuracy']

    # ---------- 模型构建 ----------
    def build_optimized_model(self, df, task_name, case_value, control_value):
        print(f"\n{'='*60}\n构建任务: {task_name}\n{'='*60}")
        df_proc = self.preprocess_data(df)
        task_data = df_proc[df_proc['分组'].isin([case_value, control_value])].copy()
        if len(task_data) < 50:
            print("样本量不足")
            return None
        task_data['target'] = (task_data['分组'] == case_value).astype(int)
        y = task_data['target']
        clinical_features = self.get_core_clinical_features(task_data)
        bacteria_features = self.get_bacteria_features(task_data)
        if not clinical_features or not bacteria_features:
            return None

        X = task_data[clinical_features + bacteria_features]
        best_comb, auc_hist, acc_hist, clin_acc = self.find_optimal_feature_combination(
            X, y, clinical_features, bacteria_features)

        best_features = best_comb['features']
        X_best = task_data[best_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_best, y, test_size=0.3, random_state=42, stratify=y)
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # ---------- 组合模型 ----------
        rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, class_weight='balanced')
        rf.fit(X_train_s, y_train)
        y_prob_rf = rf.predict_proba(X_test_s)[:, 1]
        y_pred_rf = rf.predict(X_test_s)

        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', penalty='l1', solver='liblinear')
        lr.fit(X_train_s, y_train)
        y_prob_lr = lr.predict_proba(X_test_s)[:, 1]
        y_pred_lr = lr.predict(X_test_s)
        self.rf_model = rf  # rf 是训练好的 RandomForestClassifier 对象
        self.lr_model = lr  # lr 是训练好的 LogisticRegression 对象

        # ---------- 临床-only ----------
        X_clin = task_data[clinical_features]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_clin, y, test_size=0.3, random_state=42, stratify=y)
        scaler_c = StandardScaler()
        X_train_c_s = scaler_c.fit_transform(X_train_c)
        X_test_c_s = scaler_c.transform(X_test_c)
        rf_clin = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, class_weight='balanced')
        rf_clin.fit(X_train_c_s, y_train_c)
        y_prob_c = rf_clin.predict_proba(X_test_c_s)[:, 1]
        y_pred_c = rf_clin.predict(X_test_c_s)

        # ---------- 微生物-only ----------
        X_mic = task_data[bacteria_features]
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            X_mic, y, test_size=0.3, random_state=42, stratify=y)
        scaler_m = StandardScaler()
        X_train_m_s = scaler_m.fit_transform(X_train_m)
        X_test_m_s = scaler_m.transform(X_test_m)
        rf_mic = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, class_weight='balanced')
        rf_mic.fit(X_train_m_s, y_train_m)
        y_prob_m = rf_mic.predict_proba(X_test_m_s)[:, 1]
        y_pred_m = rf_mic.predict(X_test_m_s)

        # 指标计算
        def calc_metrics(y_t, y_p, y_pr):
            return {
                'AUC': roc_auc_score(y_t, y_pr),
                'Accuracy': accuracy_score(y_t, y_p),
                'Precision': precision_score(y_t, y_p),
                'Recall': recall_score(y_t, y_p),
                'F1': f1_score(y_t, y_p)
            }

        metrics_rf = calc_metrics(y_test, y_pred_rf, y_prob_rf)
        metrics_lr = calc_metrics(y_test, y_pred_lr, y_prob_lr)
        metrics_clin = calc_metrics(y_test_c, y_pred_c, y_prob_c)
        metrics_mic = calc_metrics(y_test_m, y_pred_m, y_prob_m)

        comparison = pd.DataFrame([
            {'Model': 'Clinical-only', **metrics_clin},
            {'Model': 'Microbiome-only', **metrics_mic},
            {'Model': 'Combined (RF)', **metrics_rf},
            {'Model': 'Combined (LR)', **metrics_lr}
        ])
        print("\n模型比较:")
        print(comparison.to_string(index=False))

        # 交叉验证
        cv_auc = cross_val_score(rf, X_train_s, y_train, cv=StratifiedKFold(5), scoring='roc_auc')
        cv_acc = cross_val_score(rf, X_train_s, y_train, cv=StratifiedKFold(5), scoring='accuracy')
        print(f"组合RF 5折CV AUC: {cv_auc.mean():.4f}±{cv_auc.std():.4f}")

        self.selected_features = best_features
        self.best_combination = best_comb
        self.clinical_only_accuracy = clin_acc
        self.combined_accuracy = cv_acc.mean()

        results = {
            'task_name': task_name,
            'auc_rf': metrics_rf['AUC'], 'accuracy_rf': metrics_rf['Accuracy'],
            'precision_rf': metrics_rf['Precision'], 'recall_rf': metrics_rf['Recall'],
            'f1_rf': metrics_rf['F1'],
            'auc_lr': metrics_lr['AUC'], 'accuracy_lr': metrics_lr['Accuracy'],
            'precision_lr': metrics_lr['Precision'], 'recall_lr': metrics_lr['Recall'],
            'f1_lr': metrics_lr['F1'],
            'cv_auc_rf_mean': cv_auc.mean(), 'cv_auc_rf_std': cv_auc.std(),
            'cv_accuracy_rf_mean': cv_acc.mean(), 'cv_accuracy_rf_std': cv_acc.std(),
            'selected_features': best_features,
            'clinical_features': clinical_features,
            'bacteria_features': best_comb.get('bacteria_features', []),
            'auc_history': auc_hist,
            'accuracy_history': acc_hist,
            'clinical_only_accuracy': clin_acc,
            'combined_accuracy': cv_acc.mean(),
            'X_train': X_train_s, 'X_test': X_test_s,
            'y_train': y_train, 'y_test': y_test,
            'y_prob_rf': y_prob_rf, 'y_prob_lr': y_prob_lr,
            'feature_names': best_features,
            'comparison_df': comparison
        }
        return results

    # ---------- 绘图函数 ----------
    def save_plot(self, filename_base, dpi=300, bbox_inches='tight'):
        """保存 PNG 和 PDF 格式"""
        plt.savefig(f'{output_dir}/{filename_base}.png', dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(f'{output_dir}/{filename_base}.pdf', dpi=dpi, bbox_inches=bbox_inches)
        print(f"图表已保存: {filename_base}.png 和 {filename_base}.pdf")

    def plot_optimization_results(self, results):
        task_name = results['task_name']
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(f'{task_name} - Optimized Clinical-Microbiome Model Analysis', fontsize=16, fontweight='bold')

        # ROC
        fpr_rf, tpr_rf, _ = roc_curve(results['y_test'], results['y_prob_rf'])
        axes[0].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'RF AUC={results["auc_rf"]:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.05])
        axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve (Random Forest)'); axes[0].legend(); axes[0].grid(alpha=0.3)

        # Feature optimization
        auc_hist = results['auc_history']
        nums = [x['num_bacteria'] for x in auc_hist]
        aucs = [x['auc'] for x in auc_hist]
        axes[1].plot(nums, aucs, 'o-', color='green', lw=2, markersize=8)
        axes[1].set_xlabel('Number of Microbiome Features'); axes[1].set_ylabel('Cross-Validation AUC')
        axes[1].set_title('Feature Optimization Process'); axes[1].grid(alpha=0.3)
        best_idx = np.argmax(aucs)
        axes[1].plot(nums[best_idx], aucs[best_idx], 'ro', markersize=10)
        axes[1].annotate(f'Best: {aucs[best_idx]:.3f}', (nums[best_idx], aucs[best_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Feature importance
        imp = pd.DataFrame({
            'feature': results['feature_names'],
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        colors = ['red' if f in results['clinical_features'] else 'blue' for f in imp['feature']]
        axes[2].barh(np.arange(len(imp)), imp['importance'], color=colors)
        axes[2].set_yticks(np.arange(len(imp)))
        axes[2].set_yticklabels(imp['feature'])
        axes[2].set_xlabel('Importance'); axes[2].set_title('Feature Importance (Red:Clinical)')
        axes[2].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        self.save_plot(f'{task_name}_optimized_clinical_microbiome_model')
        plt.show()

        self.plot_accuracy_pie_charts(results)
        self.plot_calibration_curve(results)
        self.plot_decision_curve(results)
        return imp

    def plot_accuracy_pie_charts(self, results):
        task_name = results['task_name']
        clin_acc = results['clinical_only_accuracy']
        comb_acc = results['cv_accuracy_rf_mean']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'{task_name} - Model Diagnostic Accuracy Comparison', fontsize=16, fontweight='bold')

        sizes_clin = [clin_acc, 1 - clin_acc]
        labels_clin = [f'Correct\n{clin_acc:.1%}', f'Incorrect\n{1-clin_acc:.1%}']
        ax1.pie(sizes_clin, labels=labels_clin, colors=['#66c2a5', '#fc8d62'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Clinical Features Only\n(Cross-Validation Accuracy)', fontsize=14)

        sizes_comb = [comb_acc, 1 - comb_acc]
        labels_comb = [f'Correct\n{comb_acc:.1%}', f'Incorrect\n{1-comb_acc:.1%}']
        ax2.pie(sizes_comb, labels=labels_comb, colors=['#66c2a5', '#fc8d62'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Clinical + Microbiome Features\n(Cross-Validation Accuracy)', fontsize=14)

        improvement = comb_acc - clin_acc
        fig.text(0.5, 0.02, f'Improvement: {improvement:.3f} ({improvement:.1%})', ha='center', fontsize=12,
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        plt.tight_layout()
        self.save_plot(f'{task_name}_diagnostic_accuracy_pie_charts')
        plt.show()

    def plot_calibration_curve(self, results):
        task_name = results['task_name']
        y_test = results['y_test']
        y_prob = results['y_prob_rf']
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, 's-', color='darkorange', label='Random Forest')
        plt.xlabel('Predicted Probability'); plt.ylabel('Actual Probability')
        plt.title(f'{task_name} - Calibration Curve (RF)')
        plt.legend(); plt.grid(alpha=0.3)
        brier = np.mean((y_prob - y_test) ** 2)
        plt.text(0.05, 0.95, f'Brier Score: {brier:.4f}', transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        self.save_plot(f'{task_name}_calibration_curve')
        plt.show()

    def plot_decision_curve(self, results):
        task_name = results['task_name']
        y_test = results['y_test']
        y_prob = results['y_prob_rf']
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefit_rf = self.calculate_net_benefit(y_test, y_prob, thresholds)
        net_benefit_all = self.calculate_net_benefit_all(y_test, thresholds)
        net_benefit_none = np.zeros_like(thresholds)

        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, net_benefit_rf, color='darkorange', lw=2, label='Random Forest')
        plt.plot(thresholds, net_benefit_all, 'k--', label='Treat All')
        plt.plot(thresholds, net_benefit_none, 'k:', label='Treat None')
        plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit')
        plt.title(f'{task_name} - Decision Curve (RF)')
        plt.legend(); plt.grid(alpha=0.3); plt.xlim(0, 1)
        plt.tight_layout()
        self.save_plot(f'{task_name}_decision_curve')
        plt.show()

    @staticmethod
    def calculate_net_benefit(y_true, y_prob, thresholds):
        n = len(y_true)
        net = []
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tp = np.sum((pred == 1) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            if t == 0 or t == 1:
                net.append(0)
            else:
                net.append(tp/n - (fp/n) * (t/(1-t)))
        return np.array(net)

    @staticmethod
    def calculate_net_benefit_all(y_true, thresholds):
        prevalence = np.mean(y_true)
        net = []
        for t in thresholds:
            if t == 0 or t == 1:
                net.append(0)
            else:
                net.append(prevalence - (1-prevalence) * (t/(1-t)))
        return np.array(net)

    def plot_feature_composition_pie_charts(self, all_results):
        print("\n绘制特征占比饼图...")
        tasks = ['Early_Progression', 'Compensated_to_Decompensated', 'Liver_Failure']
        task_names = ['CLD → Compensated LC', 'Compensated → Decompensated LC', 'Decompensated LC → ACLF']
        clin_pcts, mic_pcts = [], []
        for t in tasks:
            if t in all_results:
                r = all_results[t]
                total = len(r['clinical_features']) + len(r['bacteria_features'])
                clin_pcts.append(len(r['clinical_features'])/total*100)
                mic_pcts.append(len(r['bacteria_features'])/total*100)
            else:
                clin_pcts.append(100); mic_pcts.append(0)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Clinical vs Microbiome Feature Importance', fontsize=16, fontweight='bold')
        colors = ['#ff6b6b', '#4d94ff']
        for i, (name, cp, mp) in enumerate(zip(task_names, clin_pcts, mic_pcts)):
            sizes = [cp, mp]
            labels = [f'Clinical\n{cp:.1f}%', f'Microbiome\n{mp:.1f}%']
            axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                        textprops={'fontsize': 10, 'fontweight': 'bold'})
            axes[i].set_title(name, fontsize=14, fontweight='bold')
        fig.legend(['Clinical Features', 'Microbiome Features'], loc='lower center', ncol=2, fontsize=12)
        plt.tight_layout()
        self.save_plot('disease_progression_feature_composition_pie_charts')
        plt.show()
        self.plot_feature_composition_trend(task_names, clin_pcts, mic_pcts)

    def plot_feature_composition_trend(self, task_names, clin_pcts, mic_pcts):
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(task_names))
        width = 0.35
        ax.bar(x - width/2, clin_pcts, width, color='#ff6b6b', label='Clinical')
        ax.bar(x + width/2, mic_pcts, width, color='#4d94ff', label='Microbiome')
        ax.plot(x, clin_pcts, 'r--', marker='o', markersize=10)
        ax.plot(x, mic_pcts, 'b--', marker='s', markersize=10)
        ax.set_xticks(x); ax.set_xticklabels(task_names, fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)'); ax.set_title('Feature Composition Trend', fontsize=16)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self.save_plot('feature_composition_trend')
        plt.show()

    def plot_detailed_feature_composition(self, all_results):
        print("\n绘制详细特征构成...")
        tasks = ['Early_Progression', 'Compensated_to_Decompensated', 'Liver_Failure']
        task_names = ['CLD → CC', 'CC → DC', 'DC → ACLF']
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Detailed Feature Composition', fontsize=18, fontweight='bold')
        for i, (t, name) in enumerate(zip(tasks, task_names)):
            if t in all_results:
                r = all_results[t]
                imp = pd.DataFrame({
                    'feature': r['feature_names'],
                    'importance': self.rf_model.feature_importances_
                }).sort_values('importance', ascending=True)
                colors = ['red' if f in r['clinical_features'] else 'blue' for f in imp['feature']]
                axes[i].barh(np.arange(len(imp)), imp['importance'], color=colors)
                axes[i].set_yticks(np.arange(len(imp)))
                axes[i].set_yticklabels(imp['feature'], fontsize=10)
                axes[i].set_xlabel('Importance'); axes[i].set_title(name, fontsize=14)
                axes[i].grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self.save_plot('detailed_feature_composition')
        plt.show()


# ---------- 主程序 ----------
def main():
    print("正在读取数据...")
    data = pd.read_excel(file_path)
    print(f"数据读取成功! 形状: {data.shape}")

    progression_tasks = {
        'Early_Progression': {'case': 2, 'control': 1},
        'Compensated_to_Decompensated': {'case': 3, 'control': 2},
        'Liver_Failure': {'case': 4, 'control': 3}
    }

    all_results = {}
    for task_name, config in progression_tasks.items():
        print(f"\n处理任务: {task_name}")
        model = OptimizedClinicalMicrobiomeModel()
        try:
            results = model.build_optimized_model(data, task_name, config['case'], config['control'])
            if results is None:
                continue
            _ = model.plot_optimization_results(results)

            # 保存Excel
            with pd.ExcelWriter(f'{output_dir}/{task_name}_detailed.xlsx') as writer:
                pd.DataFrame(results['auc_history']).to_excel(writer, sheet_name='Feature_Optimization', index=False)
                results['comparison_df'].to_excel(writer, sheet_name='Model_Comparison', index=False)
                perf_df = pd.DataFrame([{
                    'Task': task_name,
                    'RF_AUC': results['auc_rf'], 'RF_Accuracy': results['accuracy_rf'],
                    'RF_Precision': results['precision_rf'], 'RF_Recall': results['recall_rf'],
                    'RF_F1': results['f1_rf'],
                    'LR_AUC': results['auc_lr'], 'LR_Accuracy': results['accuracy_lr'],
                    'LR_Precision': results['precision_lr'], 'LR_Recall': results['recall_lr'],
                    'LR_F1': results['f1_lr'],
                    'CV_AUC_mean': results['cv_auc_rf_mean'], 'CV_AUC_std': results['cv_auc_rf_std'],
                    'CV_Acc_mean': results['cv_accuracy_rf_mean'], 'CV_Acc_std': results['cv_accuracy_rf_std'],
                    'Clinical_Only_Acc': results['clinical_only_accuracy'],
                    'Combined_Acc': results['combined_accuracy']
                }])
                perf_df.to_excel(writer, sheet_name='Performance', index=False)
                features_df = pd.DataFrame({
                    'Type': ['Clinical']*len(results['clinical_features']) + ['Microbiome']*len(results['bacteria_features']),
                    'Feature': results['clinical_features'] + results['bacteria_features']
                })
                features_df.to_excel(writer, sheet_name='Feature_List', index=False)
                prob_df = pd.DataFrame({
                    'Actual': results['y_test'],
                    'RF_Prob': results['y_prob_rf'],
                    'LR_Prob': results['y_prob_lr']
                })
                prob_df.to_excel(writer, sheet_name='Prediction_Probabilities', index=False)

            all_results[task_name] = results
        except Exception as e:
            print(f"Error: {task_name} -> {e}")
            import traceback; traceback.print_exc()

    if all_results:
        # 创建一个模型实例用于绘图
        plot_model = OptimizedClinicalMicrobiomeModel()
        # 将最后一个训练好的 rf_model 赋值给它（以便 feature_importance 能工作）
        last_results = all_results[list(all_results.keys())[-1]]
        plot_model.rf_model = RandomForestClassifier()  # 临时占位
        # 由于饼图和详细特征图不依赖 rf_model 的具体权重，直接调用即可
        # 但我们需要保留功能完整的模型对象
        dummy_model = OptimizedClinicalMicrobiomeModel()
        create_summary_report(all_results, dummy_model)
    else:
        print("无成功任务")


def create_summary_report(all_results, model_instance):
    print("\n=== 汇总报告 ===")
    summary = []
    for task, res in all_results.items():
        comp = res['comparison_df']
        summary.append({
            'Task': task,
            'Clinical_AUC': comp[comp['Model']=='Clinical-only']['AUC'].values[0],
            'Microbiome_AUC': comp[comp['Model']=='Microbiome-only']['AUC'].values[0],
            'Combined_RF_AUC': comp[comp['Model']=='Combined (RF)']['AUC'].values[0],
            'Combined_RF_F1': comp[comp['Model']=='Combined (RF)']['F1'].values[0],
            'Best_Bacteria_Features': ', '.join(res['bacteria_features'][:3])
        })
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_excel(f'{output_dir}/summary_comparison.xlsx', index=False)

    # 准确率比较图
    plt.figure(figsize=(12, 8))
    tasks = summary_df['Task']
    x = np.arange(len(tasks))
    width = 0.35
    clin_accs = [all_results[t]['clinical_only_accuracy'] for t in tasks]
    comb_accs = [all_results[t]['combined_accuracy'] for t in tasks]
    plt.bar(x - width/2, clin_accs, width, label='Clinical Only', color='red', alpha=0.8)
    plt.bar(x + width/2, comb_accs, width, label='Clinical+Microbiome', color='blue', alpha=0.8)
    plt.xticks(x, tasks); plt.ylabel('CV Accuracy'); plt.title('Model Accuracy Comparison')
    plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_accuracy_comparison.png', dpi=300)
    plt.savefig(f'{output_dir}/model_accuracy_comparison.pdf', dpi=300)
    plt.show()

    # 特征构成饼图和详细图使用最后一轮模型对象
    model_instance.plot_feature_composition_pie_charts(all_results)
    model_instance.plot_detailed_feature_composition(all_results)


if __name__ == "__main__":
    main()