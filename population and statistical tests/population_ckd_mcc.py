import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pingouin import ancova, multivariate_normality
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

# === Load Data ===
columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]

try:
    df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=columns, na_values=["?", "\t?"],
                     on_bad_lines='skip')
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'Chronic_Kidney_Disease.csv' was not found. Please ensure the file is in the correct directory.")

# === Data Cleaning ===
# Verify column names
expected_columns = columns
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in dataset: {missing_cols}. Available columns: {list(df.columns)}")

# Convert numerical columns to numeric, coercing errors to NaN
num_cols = ['Age', 'Blood Pressure', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine',
            'Sodium', 'Potassium', 'Hemoglobin', 'Packed Cell Volume',
            'White Blood Cell Count', 'Red Blood Cell Count']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"Imputed missing values in {col} with median: {median_val:.2f}")

# Impute missing values for categorical/nominal columns with mode
cat_cols = ['Specific Gravity', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell',
            'Pus Cell clumps', 'Bacteria', 'Hypertension', 'Diabetes Mellitus',
            'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia', 'Class']
for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"Imputed missing values in {col} with mode: {mode_val}")

# Convert categorical variables to appropriate formats
df['Specific Gravity'] = df['Specific Gravity'].astype(str)
df['Albumin'] = df['Albumin'].astype(int)
df['Sugar'] = df['Sugar'].astype(int)
df['Red Blood Cells'] = df['Red Blood Cells'].map({'normal': 1, 'abnormal': 0})
df['Pus Cell'] = df['Pus Cell'].map({'normal': 1, 'abnormal': 0})
df['Pus Cell clumps'] = df['Pus Cell clumps'].map({'present': 1, 'notpresent': 0})
df['Bacteria'] = df['Bacteria'].map({'present': 1, 'notpresent': 0})
df['Hypertension'] = df['Hypertension'].map({'yes': 1, 'no': 0})
df['Diabetes Mellitus'] = df['Diabetes Mellitus'].map({'yes': 1, 'no': 0})
df['Coronary Artery Disease'] = df['Coronary Artery Disease'].map({'yes': 1, 'no': 0})
df['Appetite'] = df['Appetite'].map({'good': 1, 'poor': 0})
df['Pedal Edema'] = df['Pedal Edema'].map({'yes': 1, 'no': 0})
df['Anemia'] = df['Anemia'].map({'yes': 1, 'no': 0})
df['Class'] = df['Class'].map({'ckd': 1, 'notckd': 0})

# Create plots directory
if not os.path.exists('ckd_plots'):
    os.makedirs('ckd_plots')


# === Normality Tests ===
def normality_tests(data, columns):
    print("\n=== Normality Tests ===")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 4, i)
        stats.probplot(data[col].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {col}')
    plt.tight_layout()
    plt.savefig('ckd_plots/ckd_qq_plots.png')
    plt.close()

    p_values = []
    for col in columns:
        stat, p = stats.shapiro(data[col].dropna())
        print(f'Shapiro-Wilk Test for {col}: Statistic={stat:.3f}, p-value={p:.3f}')
        print(f"Interpretation: {'Non-normal' if p < 0.05 else 'Normal'} distribution.")
        p_values.append(p)

    # Multiple comparisons correction for Shapiro-Wilk tests
    print("\n=== Multiple Comparisons Correction for Normality Tests ===")
    methods = ['bonferroni', 'holm', 'fdr_bh']  # Bonferroni, Holm, Benjamini-Hochberg
    for method in methods:
        corrected_p = multipletests(p_values, alpha=0.05, method=method)[1]
        print(f"\n{method.capitalize()} corrected p-values for Shapiro-Wilk Tests:")
        for col, p_corr in zip(columns, corrected_p):
            print(f"{col}: Corrected p-value={p_corr:.3f}, {'Non-normal' if p_corr < 0.05 else 'Normal'} distribution")


# === Levene's Test ===
def levene_test(data, dv, group):
    groups = [data[dv][data[group] == g] for g in data[group].unique() if not data[dv][data[group] == g].isna().all()]
    if len(groups) > 1:
        stat, p = stats.levene(*groups)
        print(f"\nLevene's Test for {dv} across {group}: Statistic={stat:.3f}, p-value={p:.3f}")
        print(f"Interpretation: {'Unequal' if p < 0.05 else 'Equal'} variances.")
        return p
    else:
        print(f"\nLevene's Test for {dv} across {group}: Not enough groups with valid data.")
        return np.nan


# === Mann-Whitney U Test ===
def mann_whitney_test(data, dv, group):
    group0 = data[dv][data[group] == 0].dropna()
    group1 = data[dv][data[group] == 1].dropna()
    if len(group0) > 0 and len(group1) > 0:
        stat, p = mannwhitneyu(group0, group1)
        print(f"\nMann-Whitney U Test for {dv} by {group}: Statistic={stat:.3f}, p-value={p:.3f}")
        print(f"Interpretation: {'Significant' if p < 0.05 else 'Non-significant'} difference.")
        # Cohen's d
        mean_diff = group0.mean() - group1.mean()
        pooled_std = np.sqrt(((len(group0) - 1) * group0.std() ** 2 + (len(group1) - 1) * group1.std() ** 2) / (
                    len(group0) + len(group1) - 2))
        cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
        print(
            f"Cohen's d: {cohen_d:.3f} (Effect size: {'>0.8: Large' if abs(cohen_d) > 0.8 else '0.5-0.8: Medium' if abs(cohen_d) > 0.5 else '0.2-0.5: Small' if abs(cohen_d) > 0.2 else 'Negligible'})")
        return p
    else:
        print(f"\nMann-Whitney U Test for {dv} by {group}: Insufficient data.")
        return np.nan


# === Chi-square Test ===
def chi_square_test(data, var1, var2):
    contingency_table = pd.crosstab(data[var1], data[var2])
    if contingency_table.size > 0 and contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"\nChi-square Test between {var1} and {var2}: Chi2={chi2:.3f}, p-value={p:.3f}")
        print(f"Interpretation: {'Significant' if p < 0.05 else 'Non-significant'} association.")
        return p
    else:
        print(f"\nChi-square Test between {var1} and {var2}: Insufficient data for test.")
        return np.nan


# === Cramér’s V ===
def cramers_v(data, var1, var2):
    contingency_table = pd.crosstab(data[var1], data[var2])
    if contingency_table.size > 0 and contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    return np.nan


def categorical_correlation_matrix(data, cat_vars):
    print("\n=== Categorical Correlation Analysis (Cramér’s V) ===")
    corr_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars)
    for var1 in cat_vars:
        for var2 in cat_vars:
            if var1 <= var2:
                corr_matrix.loc[var1, var2] = cramers_v(data, var1, var2)
                corr_matrix.loc[var2, var1] = corr_matrix.loc[var1, var2]

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.astype(float), annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.title('Cramér’s V Correlation Matrix for Categorical Variables')
    plt.savefig('ckd_plots/ckd_cramers_v_correlation.png')
    plt.close()


# === Correlation Analysis ===
def correlation_analysis(data, columns):
    print("\n=== Correlation Analysis ===")
    pearson_corr = data[columns].corr(method='pearson')
    spearman_corr = data[columns].corr(method='spearman')

    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix')
    plt.savefig('ckd_plots/ckd_pearson_correlation.png')
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
    plt.title('Spearman Correlation Matrix')
    plt.savefig('ckd_plots/ckd_spearman_correlation.png')
    plt.close()

    p_values_pearson = []
    p_values_spearman = []
    pairs = []
    for col1 in columns:
        for col2 in columns:
            if col1 < col2:
                valid_data = data[[col1, col2]].dropna()
                if len(valid_data) > 1:
                    pearson_r, pearson_p = stats.pearsonr(valid_data[col1], valid_data[col2])
                    spearman_r, spearman_p = stats.spearmanr(valid_data[col1], valid_data[col2])
                    print(f'\nPearson {col1} vs {col2}: r={pearson_r:.3f}, p={pearson_p:.3f}')
                    print(f'Spearman {col1} vs {col2}: r={spearman_r:.3f}, p={spearman_p:.3f}')
                    print(
                        f"Interpretation (Pearson): {'Significant' if pearson_p < 0.05 else 'Non-significant'} linear correlation.")
                    print(
                        f"Interpretation (Spearman): {'Significant' if spearman_p < 0.05 else 'Non-significant'} monotonic correlation.")
                    p_values_pearson.append(pearson_p)
                    p_values_spearman.append(spearman_p)
                    pairs.append(f"{col1} vs {col2}")

    # Multiple comparisons correction for correlation tests
    print("\n=== Multiple Comparisons Correction for Correlation Tests ===")
    methods = ['bonferroni', 'holm', 'fdr_bh']  # Bonferroni, Holm, Benjamini-Hochberg
    for method in methods:
        print(f"\n{method.capitalize()} corrected p-values for Pearson correlations:")
        corrected_p_pearson = multipletests(p_values_pearson, alpha=0.05, method=method)[1]
        for pair, p_corr in zip(pairs, corrected_p_pearson):
            print(f"{pair}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'} correlation")

        print(f"\n{method.capitalize()} corrected p-values for Spearman correlations:")
        corrected_p_spearman = multipletests(p_values_spearman, alpha=0.05, method=method)[1]
        for pair, p_corr in zip(pairs, corrected_p_spearman):
            print(f"{pair}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'} correlation")


# === MANOVA ===
def run_manova(data, dvs, iv):
    print(f"\n=== MANOVA for {iv} ===")
    df_clean = data[dvs + [iv]].dropna()
    if len(df_clean) > len(dvs):
        # Use Q() to handle column names with spaces
        formula = ' + '.join([f'Q("{dv}")' if ' ' in dv else dv for dv in dvs]) + f' ~ {iv}'
        manova = MANOVA.from_formula(formula, data=df_clean)
        result = manova.mv_test()
        print(result)
        p_values = []
        if hasattr(result, 'results'):
            for test in result.results:
                p_values.append(result.results[test]['stat']['Pr > F'].iloc[0])
            print("\n=== Multiple Comparisons Correction for MANOVA ===")
            methods = ['bonferroni', 'holm', 'fdr_bh']
            for method in methods:
                corrected_p = multipletests(p_values, alpha=0.05, method=method)[1]
                print(f"\n{method.capitalize()} corrected p-values for MANOVA tests:")
                for test, p_corr in zip(result.results.keys(), corrected_p):
                    print(f"{test}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'}")
        print(f"Interpretation: Tests if multiple dependent variables differ across {iv} groups.")
    else:
        print(f"MANOVA for {iv}: Insufficient data after dropping NaNs.")


# === MANCOVA ===
def run_mancova(data, dvs, iv, covariates):
    print(f"\n=== MANCOVA for {iv} with covariates {covariates} ===")
    df_clean = data[dvs + [iv] + covariates].dropna()
    if len(df_clean) > len(dvs) + len(covariates):
        try:
            # Use Q() to handle column names with spaces
            dvs_str = ' + '.join([f'Q("{dv}")' if ' ' in dv else dv for dv in dvs])
            cov_str = ' + '.join([f'Q("{cov}")' if ' ' in cov else cov for cov in covariates])
            formula = f'{dvs_str} ~ {iv} + {cov_str}'
            manova = MANOVA.from_formula(formula, data=df_clean)
            result = manova.mv_test()
            print(result)
            p_values = []
            if hasattr(result, 'results'):
                for test in result.results:
                    p_values.append(result.results[test]['stat']['Pr > F'].iloc[0])
                print("\n=== Multiple Comparisons Correction for MANCOVA ===")
                methods = ['bonferroni', 'holm', 'fdr_bh']
                for method in methods:
                    corrected_p = multipletests(p_values, alpha=0.05, method=method)[1]
                    print(f"\n{method.capitalize()} corrected p-values for MANCOVA tests:")
                    for test, p_corr in zip(result.results.keys(), corrected_p):
                        print(f"{test}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'}")
            print(
                f"Interpretation: Tests group differences in multiple dependent variables while controlling for covariates.")
        except ValueError as e:
            print(f"MANCOVA for {iv}: Failed due to {str(e)}.")
    else:
        print(f"MANCOVA for {iv}: Insufficient data after dropping NaNs.")


# === ANCOVA ===
def run_ancova(data, dv, iv, covariates):
    print(f"\n=== ANCOVA for {dv} with {iv} ===")
    df_clean = data[[dv, iv] + covariates].dropna()
    if len(df_clean) > len(covariates) + 1:
        try:
            # Use Q() to handle column names with spaces
            dv_str = f'Q("{dv}")' if ' ' in dv else dv
            cov_str = ' + '.join([f'Q("{cov}")' if ' ' in cov else cov for cov in covariates])
            formula = f'{dv_str} ~ {iv} + {cov_str}'
            model = ols(formula, data=df_clean).fit()
            anova_results = anova_lm(model)
            print(anova_results)
            p_value = anova_results.loc[iv, 'PR(>F)'] if iv in anova_results.index else np.nan
            if not np.isnan(p_value):
                print("\n=== Multiple Comparisons Correction for ANCOVA ===")
                methods = ['bonferroni', 'holm', 'fdr_bh']
                for method in methods:
                    corrected_p = multipletests([p_value], alpha=0.05, method=method)[1][0]
                    print(f"{method.capitalize()} corrected p-value: {corrected_p:.3f}, {'Significant' if corrected_p < 0.05 else 'Non-significant'}")
            print(f"Interpretation: Tests if {iv} groups differ on {dv} while controlling for covariates.")
        except ValueError as e:
            print(f"ANCOVA for {dv}: Failed due to {str(e)}.")
    else:
        print(f"ANCOVA for {dv}: Insufficient data after dropping NaNs.")


# === Logistic Regression ===
def logistic_regression_analysis(data, predictors, target):
    print("\n=== Logistic Regression Analysis ===")
    df_clean = data[predictors + [target]].dropna()
    if len(df_clean) > 0:
        X = df_clean[predictors]
        y = df_clean[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        coef = model.coef_[0]
        odds_ratios = np.exp(coef)
        print("Logistic Regression Coefficients and Odds Ratios:")
        for pred, c, odds in zip(predictors, coef, odds_ratios):
            print(f"{pred}: Coefficient={c:.3f}, Odds Ratio={odds:.3f}")
            print(f"Interpretation: Odds ratio > 1 indicates increased CKD risk with higher {pred}.")
    else:
        print("Logistic Regression: Insufficient data after dropping NaNs.")


# === Population Analysis ===
def population_analysis(data):
    print("\n=== Population Analysis ===")
    thresholds = {
        'Age': 50,
        'Blood Pressure': 140,
        'Serum Creatinine': 1.2,
        'Hemoglobin': 12,
        'Blood Glucose Random': 200,
        'Blood Urea': 40,
        'Specific Gravity': '1.010',
        'Albumin': 3,
    }

    groups = [
        ('Older Patients (≥50)', data['Age'] >= thresholds['Age']),
        ('Hypertensive (BP≥140)', data['Blood Pressure'] >= thresholds['Blood Pressure']),
        ('High Creatinine (≥1.2)', data['Serum Creatinine'] >= thresholds['Serum Creatinine']),
        ('Low Hemoglobin (<12)', data['Hemoglobin'] < thresholds['Hemoglobin']),
        ('High Glucose (≥200)', data['Blood Glucose Random'] >= thresholds['Blood Glucose Random']),
        ('High Urea (≥40)', data['Blood Urea'] >= thresholds['Blood Urea']),
        ('Low Specific Gravity (≤1.010)', data['Specific Gravity'] <= thresholds['Specific Gravity']),
        ('High Albumin (≥3)', data['Albumin'] >= thresholds['Albumin']),
        ('Abnormal RBC', data['Red Blood Cells'] == 0),
        ('Abnormal Pus Cells', data['Pus Cell'] == 0),
        ('Pus Cell Clumps Present', data['Pus Cell clumps'] == 1),
        ('Bacteria Present', data['Bacteria'] == 1),
        ('Hypertension Present', data['Hypertension'] == 1),
        ('Diabetes Mellitus', data['Diabetes Mellitus'] == 1),
        ('Poor Appetite', data['Appetite'] == 0),
        ('Pedal Edema', data['Pedal Edema'] == 1),
        ('Anemia', data['Anemia'] == 1),
        ('Older with High Creatinine',
         (data['Age'] >= thresholds['Age']) & (data['Serum Creatinine'] >= thresholds['Serum Creatinine'])),
        ('Hypertensive with Diabetes', (data['Hypertension'] == 1) & (data['Diabetes Mellitus'] == 1)),
        ('Low Hemoglobin with Pedal Edema',
         (data['Hemoglobin'] < thresholds['Hemoglobin']) & (data['Pedal Edema'] == 1)),
        ('High Albumin with Proteinuria', (data['Albumin'] >= thresholds['Albumin']) & (data['Pus Cell clumps'] == 1)),
        ('Diabetic with High Creatinine',
         (data['Diabetes Mellitus'] == 1) & (data['Serum Creatinine'] >= thresholds['Serum Creatinine'])),
    ]

    summary_data = []
    prevalences = []
    group_names = []

    # Boxplot for selected groups
    selected_groups = ['Older Patients (≥50)', 'High Creatinine (≥1.2)', 'Low Hemoglobin (<12)', 'Hypertension Present',
                       'High Albumin (≥3)']
    boxplot_data = []
    boxplot_labels = []

    for name, mask in groups:
        group_data = data[mask]
        if len(group_data) > 0:
            prevalence = group_data['Class'].mean()
            mean_creatinine = group_data['Serum Creatinine'].mean()
            mean_hemoglobin = group_data['Hemoglobin'].mean()
            count = len(group_data)
            print(f'\nAnalysis for {name}:')
            print(f'Count: {count}')
            print(f'Prevalence of CKD: {prevalence:.3f}')
            print(f'Average Serum Creatinine: {mean_creatinine:.3f}')
            print(f'Average Hemoglobin: {mean_hemoglobin:.3f}')
            prevalences.append(prevalence)
            group_names.append(name)

            chi2, p = np.nan, np.nan
            if len(group_data) > 1 and len(data['Class'].unique()) > 1:
                contingency_table = pd.crosstab(group_data['Class'], data['Class'])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p = chi2_contingency(contingency_table.values)[0:2]
                    print(f'Chi-square Test for {name} vs CKD: Chi2={chi2:.3f}, p={p:.3f}')
                    print(f"Interpretation: {'Significant' if p < 0.05 else 'Non-significant'} association with CKD.")
                    return p

            summary_data.append([name, count, f"{prevalence:.3f}", f"{mean_creatinine:.3f}",
                                 f"{mean_hemoglobin:.3f}", f"{chi2:.3f}" if not np.isnan(chi2) else 'NaN',
                                 f"{p:.3f}" if not np.isnan(p) else 'NaN',
                                 'Significant' if p < 0.05 else 'Non-significant' if not np.isnan(p) else 'N/A'])

            if name in selected_groups:
                boxplot_data.append(group_data['Serum Creatinine'])
                boxplot_labels.append(name)

    # Summary table
    summary_df = pd.DataFrame(summary_data,
                              columns=['Group', 'Count', 'CKD Prevalence', 'Mean Creatinine',
                                       'Mean Hemoglobin', 'Chi2', 'P-value', 'Significance'])
    print("\n=== Population Analysis Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv('ckd_plots/ckd_population_summary.csv')

    # Bar plot for prevalence
    plt.figure(figsize=(14, 12))
    plt.barh(group_names, prevalences, color='skyblue')
    plt.xlabel('Prevalence of CKD')
    plt.title('CKD Prevalence by Population Group')
    plt.tight_layout()
    plt.savefig('ckd_plots/ckd_prevalence_plot.png')
    plt.close()

    # Boxplot for selected groups
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, labels=boxplot_labels)
    plt.ylabel('Serum Creatinine (mg/dL)')
    plt.title('Serum Creatinine Distribution by Selected Groups')
    plt.savefig('ckd_plots/ckd_creatinine_boxplot.png')
    plt.close()

    # Multiple comparisons correction for Chi-square tests
    print("\n=== Multiple Comparisons Correction for Chi-square Tests ===")
    chi_p_values = [p for name, mask in groups if len(data[mask]) > 0 for p in [chi_square_test(data[mask], 'Class', 'Class')]]
    methods = ['bonferroni', 'holm', 'fdr_bh']
    for method in methods:
        corrected_p = multipletests([p for p in chi_p_values if not np.isnan(p)], alpha=0.05, method=method)[1]
        print(f"\n{method.capitalize()} corrected p-values for Chi-square Tests:")
        for name, p_corr in zip([name for name, mask in groups if len(data[mask]) > 0], corrected_p):
            print(f"{name}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'} association")


# === Run Analyses ===
continuous_vars = ['Age', 'Blood Pressure', 'Blood Glucose Random', 'Blood Urea',
                   'Serum Creatinine', 'Sodium', 'Potassium', 'Hemoglobin',
                   'Packed Cell Volume', 'White Blood Cell Count', 'Red Blood Cell Count']
categorical_vars = ['Specific Gravity', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell',
                    'Pus Cell clumps', 'Bacteria', 'Hypertension', 'Diabetes Mellitus',
                    'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia']
dependent_vars = ['Serum Creatinine', 'Hemoglobin', 'Blood Urea', 'Blood Glucose Random']
covariates = ['Age', 'Blood Pressure', 'Hypertension', 'Diabetes Mellitus']

print("\n=== Data Summary ===")
print(df.describe())
print("\n=== Categorical Variables Summary ===")
for col in categorical_vars:
    print(f"\n{col} Value Counts:")
    print(df[col].value_counts())

normality_tests(df, continuous_vars)

levene_p_values = []
for var in dependent_vars:
    p = levene_test(df, var, 'Class')
    levene_p_values.append(p)
# Multiple comparisons correction for Levene's tests
print("\n=== Multiple Comparisons Correction for Levene's Tests ===")
methods = ['bonferroni', 'holm', 'fdr_bh']
for method in methods:
    corrected_p = multipletests([p for p in levene_p_values if not np.isnan(p)], alpha=0.05, method=method)[1]
    print(f"\n{method.capitalize()} corrected p-values for Levene's Tests:")
    for var, p_corr in zip([v for v, p in zip(dependent_vars, levene_p_values) if not np.isnan(p)], corrected_p):
        print(f"{var}: Corrected p-value={p_corr:.3f}, {'Unequal' if p_corr < 0.05 else 'Equal'} variances")

mann_p_values = []
for var in dependent_vars:
    p = mann_whitney_test(df, var, 'Class')
    mann_p_values.append(p)
# Multiple comparisons correction for Mann-Whitney U tests
print("\n=== Multiple Comparisons Correction for Mann-Whitney U Tests ===")
for method in methods:
    corrected_p = multipletests([p for p in mann_p_values if not np.isnan(p)], alpha=0.05, method=method)[1]
    print(f"\n{method.capitalize()} corrected p-values for Mann-Whitney U Tests:")
    for var, p_corr in zip([v for v, p in zip(dependent_vars, mann_p_values) if not np.isnan(p)], corrected_p):
        print(f"{var}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'} difference")

chi_p_values = []
for var in categorical_vars:
    p = chi_square_test(df, var, 'Class')
    chi_p_values.append(p)
# Multiple comparisons correction for Chi-square tests
print("\n=== Multiple Comparisons Correction for Chi-square Tests ===")
for method in methods:
    corrected_p = multipletests([p for p in chi_p_values if not np.isnan(p)], alpha=0.05, method=method)[1]
    print(f"\n{method.capitalize()} corrected p-values for Chi-square Tests:")
    for var, p_corr in zip([v for v, p in zip(categorical_vars, chi_p_values) if not np.isnan(p)], corrected_p):
        print(f"{var}: Corrected p-value={p_corr:.3f}, {'Significant' if p_corr < 0.05 else 'Non-significant'} association")

categorical_correlation_matrix(df, categorical_vars)

correlation_analysis(df, continuous_vars + ['Class'])

run_manova(df, dependent_vars, 'Class')

run_mancova(df, dependent_vars, 'Class', covariates)

for dv in dependent_vars:
    run_ancova(df, dv, 'Class', covariates)

logistic_regression_analysis(df, continuous_vars, 'Class')

population_analysis(df)

# === Additional Visualizations ===
# Serum Creatinine by CKD Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Serum Creatinine', data=df)
plt.title('Serum Creatinine by CKD Status')
plt.savefig('ckd_plots/ckd_creatinine_by_class.png')
plt.close()

# Hemoglobin by CKD Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Hemoglobin', data=df)
plt.title('Hemoglobin by CKD Status')
plt.savefig('ckd_plots/ckd_hemoglobin_by_class.png')
plt.close()

# Serum Creatinine by Age Group
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.cut(df['Age'], bins=[0, 50, 100], labels=['Young (<50)', 'Old (≥50)']), y='Serum Creatinine', data=df)
plt.title('Serum Creatinine by Age Group')
plt.savefig('ckd_plots/ckd_creatinine_by_age.png')
plt.close()

print("\n=== Analysis Complete ===")
print("All plots and summaries saved in 'ckd_plots' directory.")