import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pingouin import ancova, multivariate_normality
from scipy.stats import chi2_contingency
import os
import warnings

warnings.filterwarnings('ignore')

# Load data
try:
    data = pd.read_csv(r"heart_disease_uci.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'heart_disease_uci.csv' was not found. Please ensure the file is in the correct directory.")

# Verify column names
expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'num']
missing_cols = [col for col in expected_columns if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in dataset: {missing_cols}. Available columns: {list(data.columns)}")

# Data cleaning
# Impute missing values for numerical columns with median
num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Impute missing values for categorical/binary columns with mode
cat_cols = ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'thal']
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Convert categorical variables
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
data['fbs'] = data['fbs'].map({True: 1, False: 0}).astype(int)
data['exang'] = data['exang'].map({True: 1, False: 0}).astype(int)
data['cp'] = data['cp'].map({'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4})
data['restecg'] = data['restecg'].map({'normal': 0, 'stt abnormality': 1, 'lv hypertrophy': 2})
data['slope'] = data['slope'].map({'upsloping': 1, 'flat': 2, 'downsloping': 3})
data['thal'] = data['thal'].map({'normal': 1, 'fixed defect': 2, 'reversable defect': 3})

# Create binary outcome (0 vs 1,2,3,4)
data['num_binary'] = (data['num'] > 0).astype(int)

# Define thresholds
age_threshold_male = 60
age_threshold_female = 65
chol_threshold = 240
fbs_threshold = 1
ca_threshold = 1  # At least one major vessel colored by fluoroscopy

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

# Normality tests (Shapiro-Wilk and Q-Q plots)
def normality_tests(data, columns):
    print("=== Normality Tests ===")
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        stats.probplot(data[col].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {col}')
    plt.tight_layout()
    plt.savefig('plots/qq_plots.png')
    plt.close()

    for col in columns:
        stat, p = stats.shapiro(data[col].dropna())
        print(f'Shapiro-Wilk Test for {col}: Statistic={stat:.3f}, p-value={p:.3f}')
        print(f"Interpretation: {'Non-normal' if p < 0.05 else 'Normal'} distribution. "
              f"A p-value < 0.05 suggests the data deviates from normality, potentially affecting parametric tests like MANOVA/ANCOVA.")

# Levene's test for homogeneity of variances
def levene_test(data, dv, group):
    groups = [data[dv][data[group] == g] for g in data[group].unique() if not data[dv][data[group] == g].isna().all()]
    if len(groups) > 1:
        stat, p = stats.levene(*groups)
        print(f"Levene's Test for {dv} across {group}: Statistic={stat:.3f}, p-value={p:.3f}")
        print(f"Interpretation: {'Unequal' if p < 0.05 else 'Equal'} variances. "
              f"A p-value < 0.05 indicates unequal variances, suggesting caution with parametric tests like ANCOVA.")
    else:
        print(f"Levene's Test for {dv} across {group}: Not enough groups with valid data.")

# Chi-square test for categorical variables
def chi_square_test(data, var1, var2):
    contingency_table = pd.crosstab(data[var1], data[var2])
    if contingency_table.size > 0 and contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f'Chi-square Test between {var1} and {var2}: Chi2={chi2:.3f}, p-value={p:.3f}')
        print(f"Interpretation: {'Significant' if p < 0.05 else 'Non-significant'} association. "
              f"A p-value < 0.05 indicates a significant relationship between {var1} and {var2}, suggesting {var1} is associated with heart disease presence.")
    else:
        print(f'Chi-square Test between {var1} and {var2}: Insufficient data for test.')

# Cramér’s V for categorical correlations
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

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.astype(float), annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.title('Cramér’s V Correlation Matrix for Categorical Variables')
    plt.savefig('plots/cramers_v_correlation.png')
    plt.close()

# Correlation analysis (Pearson and Spearman)
def correlation_analysis(data, columns):
    print("\n=== Correlation Analysis ===")
    pearson_corr = data[columns].corr(method='pearson')
    spearman_corr = data[columns].corr(method='spearman')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix')
    plt.savefig('plots/pearson_correlation.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
    plt.title('Spearman Correlation Matrix')
    plt.savefig('plots/spearman_correlation.png')
    plt.close()

    for col1 in columns:
        for col2 in columns:
            if col1 < col2:
                valid_data = data[[col1, col2]].dropna()
                if len(valid_data) > 1:
                    pearson_r, pearson_p = stats.pearsonr(valid_data[col1], valid_data[col2])
                    spearman_r, spearman_p = stats.spearmanr(valid_data[col1], valid_data[col2])
                    print(f'Pearson {col1} vs {col2}: r={pearson_r:.3f}, p={pearson_p:.3f}')
                    print(f'Spearman {col1} vs {col2}: r={spearman_r:.3f}, p={spearman_p:.3f}')
                    print(
                        f"Interpretation (Pearson): {'Significant' if pearson_p < 0.05 else 'Non-significant'} correlation. "
                        f"A p-value < 0.05 indicates a significant linear relationship.")
                    print(
                        f"Interpretation (Spearman): {'Significant' if spearman_p < 0.05 else 'Non-significant'} correlation. "
                        f"A p-value < 0.05 indicates a significant monotonic relationship.")

# MANOVA
def run_manova(data, dvs, iv):
    print(f"\n=== MANOVA for {iv} ===")
    df = data[dvs + [iv]].dropna()
    if len(df) > len(dvs):
        manova = MANOVA.from_formula(f"{'+'.join(dvs)} ~ {iv}", data=df)
        result = manova.mv_test()
        print(result)
        print(
            "Interpretation: Tests if multiple dependent variables (trestbps, chol, thalch, oldpeak) differ across groups of {iv}. "
            "Significant p-values (<0.05) in Pillai's Trace or Wilks' Lambda suggest group differences in the multivariate distribution. "
            "Non-normal data (as indicated) may affect results; consider PERMANOVA.")
    else:
        print(f"MANOVA for {iv}: Insufficient data after dropping NaNs.")

# MANCOVA
def run_mancova(data, dvs, iv, covariates):
    print(f"\n=== MANCOVA for {iv} with covariates {covariates} ===")
    df = data[dvs + [iv] + covariates].dropna()
    if len(df) > len(dvs) + len(covariates):
        try:
            formula = f"{'+'.join(dvs)} ~ {iv} + {'+'.join(covariates)}"
            manova = MANOVA.from_formula(formula, data=df)
            result = manova.mv_test()
            print(result)
            print(
                "Interpretation: Tests group differences in multiple dependent variables while controlling for covariates. "
                "Significant p-values (<0.05) indicate differences after covariate adjustment. "
                "Non-normal data or unequal variances may affect results; consider PERMANOVA.")
        except ValueError as e:
            print(
                f"MANCOVA for {iv}: Failed due to {str(e)}. Likely due to multicollinearity or insufficient variation in covariates.")
    else:
        print(f"MANCOVA for {iv}: Insufficient data after dropping NaNs.")

# ANCOVA
def run_ancova(data, dv, iv, covariates):
    print(f"\n=== ANCOVA for {dv} with {iv} ===")
    df = data[[dv, iv] + covariates].dropna()
    if len(df) > len(covariates) + 1:
        try:
            formula = f"{dv} ~ {iv} + {'+'.join(covariates)}"
            model = ols(formula, data=df).fit()
            anova_results = anova_lm(model)
            print(anova_results)
            print("Interpretation: Tests if groups of {iv} differ on {dv} while controlling for covariates. "
                  "Significant p-values (<0.05) for {iv} indicate group differences after covariate adjustment. "
                  "Non-normal data or unequal variances (as indicated) may affect reliability.")
        except ValueError as e:
            print(f"ANCOVA for {dv}: Failed due to {str(e)}.")
    else:
        print(f"ANCOVA for {dv}: Insufficient data after dropping NaNs.")

# Enhanced population analysis with summary table and boxplot
def population_analysis(data):
    print("\n=== Population Analysis ===")
    groups = [
        ('Old Males', (data['sex'] == 1) & (data['age'] >= age_threshold_male)),
        ('Old Females', (data['sex'] == 0) & (data['age'] >= age_threshold_female)),
        ('High Cholesterol', data['chol'] >= chol_threshold),
        ('High FBS', data['fbs'] == 1),
        ('Typical Angina', data['cp'] == 1),
        ('Asymptomatic', data['cp'] == 4),
        ('Old Males with High Cholesterol',
         (data['sex'] == 1) & (data['age'] >= age_threshold_male) & (data['chol'] >= chol_threshold)),
        ('Young Males with High FBS', (data['sex'] == 1) & (data['age'] < age_threshold_male) & (data['fbs'] == 1)),
        ('Females with Asymptomatic CP', (data['sex'] == 0) & (data['cp'] == 4)),
        ('Males with Typical Angina and High Chol',
         (data['sex'] == 1) & (data['cp'] == 1) & (data['chol'] >= chol_threshold)),
        ('High Chol with Exercise Angina', (data['chol'] >= chol_threshold) & (data['exang'] == 1)),
        ('Old Patients with High FBS', (data['age'] >= age_threshold_male) & (data['fbs'] == 1)),
        ('LV Hypertrophy', data['restecg'] == 2),
        ('Flat Slope with High Chol', (data['slope'] == 2) & (data['chol'] >= chol_threshold)),
        ('High CA (≥1 vessel)', data['ca'] >= ca_threshold),
        ('Reversible Defect Thal', data['thal'] == 3)
    ]

    summary_data = []
    prevalences = []
    group_names = []

    # Boxplot for selected groups
    selected_groups = ['Old Males', 'Old Females', 'High Cholesterol', 'Asymptomatic']
    boxplot_data = []
    boxplot_labels = []

    for name, mask in groups:
        group_data = data[mask]
        if len(group_data) > 0:
            prevalence = group_data['num_binary'].mean()
            mean_num = group_data['num'].mean()
            count = len(group_data)
            print(f'\nAnalysis for {name}:')
            print(f'Count: {count}')
            print(f'Prevalence of Heart Disease (num > 0): {prevalence:.3f}')
            print(f'Average num (disease stage): {mean_num:.3f}')
            prevalences.append(prevalence)
            group_names.append(name)

            chi2, p = np.nan, np.nan
            if len(group_data) > 1 and len(data['num_binary'].unique()) > 1:
                contingency_table = pd.crosstab(group_data['num_binary'], data['num_binary'])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p = chi2_contingency(contingency_table.values)[0:2]
                    print(f'Chi-square test for {name} vs Heart Disease: Chi2={chi2:.3f}, p={p:.3f}')
                    print(
                        f"Interpretation: {'Significant' if p < 0.05 else 'Non-significant'} association with heart disease. "
                        f"A significant p-value (<0.05) suggests this group is associated with heart disease presence.")

            summary_data.append([name, count, f"{prevalence:.3f}", f"{mean_num:.3f}",
                                 f"{chi2:.3f}" if not np.isnan(chi2) else 'NaN',
                                 f"{p:.3f}" if not np.isnan(p) else 'NaN',
                                 'Significant' if p < 0.05 else 'Non-significant' if not np.isnan(p) else 'N/A'])

            if name in selected_groups:
                boxplot_data.append(group_data['num'])
                boxplot_labels.append(name)

    # Summary table
    summary_df = pd.DataFrame(summary_data,
                              columns=['Group', 'Count', 'Prevalence', 'Mean Num', 'Chi2', 'P-value', 'Significance'])
    print("\n=== Population Analysis Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv('plots/population_summary.csv')

    # Bar plot for prevalence
    plt.figure(figsize=(14, 12))
    plt.barh(group_names, prevalences, color='skyblue')
    plt.xlabel('Prevalence of Heart Disease')
    plt.title('Heart Disease Prevalence by Population Group')
    plt.tight_layout()
    plt.savefig('plots/prevalence_plot.png')
    plt.close()

    # Boxplot for selected groups
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, labels=boxplot_labels)
    plt.ylabel('Heart Disease Stage (num)')
    plt.title('Distribution of Heart Disease Stage by Selected Groups')
    plt.savefig('plots/num_boxplot.png')
    plt.close()

# Run analyses
continuous_vars = ['age', 'chol', 'trestbps', 'thalch', 'oldpeak']
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'slope', 'thal']
dependent_vars = ['trestbps', 'chol', 'thalch', 'oldpeak']
covariates = ['age', 'sex', 'cp', 'fbs']  # Removed 'chol' to avoid multicollinearity

normality_tests(data, continuous_vars)

for var in dependent_vars:
    levene_test(data, var, 'num')

for var in categorical_vars:
    chi_square_test(data, var, 'num_binary')

categorical_correlation_matrix(data, categorical_vars)

correlation_analysis(data, continuous_vars + ['num', 'num_binary'])

run_manova(data, dependent_vars, 'num')

run_mancova(data, dependent_vars, 'num', covariates)

for dv in dependent_vars:
    run_ancova(data, dv, 'num', covariates)

population_analysis(data)

print("\n=== Binary Outcome Analysis ===")
run_manova(data, dependent_vars, 'num_binary')
run_mancova(data, dependent_vars, 'num_binary', covariates)
for dv in dependent_vars:
    run_ancova(data, dv, 'num_binary', covariates)