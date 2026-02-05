import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm
from scipy.stats import chi2_contingency, norm, shapiro, mannwhitneyu
from decimal import Decimal
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import f_oneway, kruskal, levene
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder

#------------------------------Hypothesis 1----------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm, shapiro, mannwhitneyu
from decimal import Decimal

# Load dataset (update filename as needed)
df = pd.read_csv(r"heart_disease_uci.csv")

# Preprocess: binarize target and encode sex
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# ---- Chi-square Test ----
contingency = pd.crosstab(df['sex'], df['num'])
chi2_stat, p_val_chi2, dof, expected = chi2_contingency(contingency)
p_val_chi2_str = "< 1e-6" if p_val_chi2 < 1e-6 else f"{p_val_chi2:.4e}"

print("=== Chi-square Test of Independence ===")
print("Null Hypothesis: Sex and heart disease are independent.")
print(f"p-value = {p_val_chi2_str}")
if p_val_chi2 < 0.05:
    print("Conclusion: Reject H0. There is a significant association between sex and heart disease.\n")
else:
    print("Conclusion: Fail to reject H0. No significant association found.\n")

# ---- Two-Proportion Z-Test ----
count_men = contingency.loc[1, 1]
n_men = contingency.loc[1].sum()
count_women = contingency.loc[0, 1]
n_women = contingency.loc[0].sum()

p1 = count_men / n_men
p2 = count_women / n_women
p_pool = (count_men + count_women) / (n_men + n_women)
z = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1/n_men + 1/n_women))
p_val_z = 2 * (1 - norm.cdf(abs(z)))
p_val_z_str = "< 1e-6" if p_val_z < 1e-6 else f"{p_val_z:.4e}"

print("=== Two-Proportion Z-Test ===")
print("Null Hypothesis: Heart disease rates are the same in men and women.")
print(f"p-value = {p_val_z_str}")
if p_val_z < 0.05:
    print("Conclusion: Reject H0. Significant difference in heart disease rates.")
    if p1 > p2:
        print("Interpretation: Men have a significantly higher rate of heart disease.\n")
    else:
        print("Interpretation: Women have a significantly higher rate of heart disease.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference in heart disease rates.\n")

# ---- Normality Test (Shapiro-Wilk) ----
male_disease = df[df['sex'] == 1]['num']
female_disease = df[df['sex'] == 0]['num']

stat_male, p_male = shapiro(male_disease)
stat_female, p_female = shapiro(female_disease)
p_male_str = "< 1e-6" if p_male < 1e-6 else f"{p_male:.4e}"
p_female_str = "< 1e-6" if p_female < 1e-6 else f"{p_female:.4e}"

print("=== Shapiro-Wilk Normality Test ===")
print("Null Hypothesis: Data follows a normal distribution.")
print(f"Men:    p-value = {p_male_str} → {'Normal' if p_male > 0.05 else 'Not normal'}")
print(f"Women:  p-value = {p_female_str} → {'Normal' if p_female > 0.05 else 'Not normal'}\n")

# ---- Non-Parametric Test: Mann-Whitney U ----
u_stat, p_val_u = mannwhitneyu(male_disease, female_disease, alternative='two-sided')
p_val_u_str = "< 1e-6" if p_val_u < 1e-6 else f"{p_val_u:.4e}"

print("=== Mann-Whitney U Test (Non-Parametric) ===")
print("Null Hypothesis: Distribution of heart disease is the same for men and women.")
print(f"p-value = {p_val_u_str}")
if p_val_u < 0.05:
    print("Conclusion: Reject H0. Significant difference in distributions.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference in distributions.\n")


print('------------------------------Hypothesis 2----------------------------------------------------')

# Load data
df = pd.read_csv(r"heart_disease_uci.csv")

# Ensure numeric chest pain types
df['cp'] = df['cp'].map({
    'typical angina': 1,
    'atypical angina': 2,
    'non-anginal': 3,
    'asymptomatic': 4
}) if df['cp'].dtype == object else df['cp']

# Bin age into two groups
median_age = df['age'].median()
df['age_group'] = df['age'].apply(lambda x: 'Older' if x >= median_age else 'Younger')

# --- Normality Test for All Columns ---
print("=== Shapiro-Wilk Normality Test for All Features ===")
for col in df.columns:
    col_data = df[col].dropna()
    if pd.api.types.is_numeric_dtype(col_data):
        try:
            stat, p = shapiro(col_data.sample(n=min(len(col_data), 500)))  # limit sample for stability
            print(f"{col}: p-value = {'< 1e-6' if p < 1e-6 else f'{p:.4e}'} → {'Not normal' if p < 0.05 else 'Normal'}")
        except Exception as e:
            print(f"{col}: Shapiro-Wilk test error: {e}")
    else:
        print(f"{col}: Skipped (non-numeric)")
print()

# ---- Chi-square Test ----
contingency = pd.crosstab(df['age_group'], df['cp'])
chi2_stat, p_val_chi2, dof, expected = chi2_contingency(contingency)

print("=== Chi-square Test ===")
print("Null Hypothesis: Chest pain type is independent of age group.")
print(f"p-value = {'< 1e-6' if p_val_chi2 < 1e-6 else f'{p_val_chi2:.4e}'}")
if p_val_chi2 < 0.05:
    print("Conclusion: Reject H0. Chest pain type is associated with age group.\n")
else:
    print("Conclusion: Fail to reject H0.\n")

# ---- Fisher's Exact Test (2x2) ----
df['cp_grouped'] = df['cp'].apply(lambda x: 'Atypical' if x in [2, 3] else ('Typical' if x == 1 else 'Other'))
filtered = df[df['cp_grouped'] != 'Other']
fisher_table = pd.crosstab(filtered['age_group'], filtered['cp_grouped'])

if fisher_table.shape == (2, 2):
    _, p_val_fisher = fisher_exact(fisher_table)
    print("=== Fisher's Exact Test ===")
    print("Null Hypothesis: Chest pain type (typical/atypical) is independent of age group.")
    print(f"p-value = {'< 1e-6' if p_val_fisher < 1e-6 else f'{p_val_fisher:.4e}'}")
    if p_val_fisher < 0.05:
        print("Conclusion: Reject H0. Age group is associated with chest pain type.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")
else:
    print("Fisher's test skipped: not a 2x2 table.\n")

# ---- Independent t-test: Is age higher in atypical vs typical? ----
age_typical = df[df['cp'] == 1]['age']
age_atypical = df[df['cp'].isin([2, 3])]['age']

if len(age_typical) > 0 and len(age_atypical) > 0:
    # t-test
    t_stat, p_t = ttest_ind(age_atypical, age_typical, equal_var=False)
    print("=== Independent t-Test ===")
    print("Null Hypothesis: Mean age is equal between typical and atypical chest pain.")
    print(f"p-value = {'< 1e-6' if p_t < 1e-6 else f'{p_t:.4e}'}")
    if p_t < 0.05:
        print("Conclusion: Reject H0. Mean age differs between groups.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")

    # Mann–Whitney U Test
    u_stat, p_u = mannwhitneyu(age_atypical, age_typical, alternative='greater')
    print("=== Mann-Whitney U Test (One-Sided) ===")
    print("Hypothesis: Older patients are more likely to have atypical chest pain.")
    print(f"p-value = {'< 1e-6' if p_u < 1e-6 else f'{p_u:.4e}'}")
    if p_u < 0.05:
        print("Conclusion: Reject H0. Atypical pain is more likely in older patients.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")

    # Permutation Test
    all_ages = np.concatenate([age_typical, age_atypical])
    labels = np.array([0]*len(age_typical) + [1]*len(age_atypical))  # 0: typical, 1: atypical
    obs_diff = np.mean(age_atypical) - np.mean(age_typical)

    n_permutations = 10000
    diffs = []
    for _ in range(n_permutations):
        permuted = np.random.permutation(labels)
        group1 = all_ages[permuted == 0]
        group2 = all_ages[permuted == 1]
        diffs.append(np.mean(group2) - np.mean(group1))

    p_perm = np.mean(np.array(diffs) >= obs_diff)

    print("=== Permutation Test ===")
    print("Null Hypothesis: No difference in age between typical and atypical chest pain.")
    print(f"Observed mean difference: {obs_diff:.4f}")
    print(f"Permutation p-value = {'< 1e-6' if p_perm < 1e-6 else f'{p_perm:.4e}'}")
    if p_perm < 0.05:
        print("Conclusion: Reject H0. Older age is associated with atypical chest pain.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")
else:
    print("Insufficient data for t-test / U-test / permutation test.\n")


print('------------------------------Hypothesis 3----------------------------------------------------')
# Load data
df = pd.read_csv(r"heart_disease_uci.csv")

# Drop missing values for relevant columns
chol_ca_df = df[['chol', 'ca']].dropna()

# Group cholesterol values by 'ca'
groups = [group['chol'].values for name, group in chol_ca_df.groupby('ca') if len(group) > 0]

# ==== Levene’s Test ====
print("=== Levene’s Test for Equal Variance ===")
try:
    stat_lev, p_lev = levene(*groups)
    print("H0: Variances of cholesterol are equal across 'ca' groups.")
    print(f"p-value = {'< 1e-6' if p_lev < 1e-6 else f'{p_lev:.4e}'}")
    if p_lev < 0.05:
        print("Conclusion: Reject H0. Variances are not equal.\n")
    else:
        print("Conclusion: Fail to reject H0. Variances appear equal.\n")
except Exception as e:
    print(f"Levene test error: {e}\n")

# ==== One-way ANOVA ====
print("=== One-way ANOVA (Parametric) ===")
try:
    f_stat, p_anova = f_oneway(*groups)
    print("H0: Mean cholesterol levels are equal across severity levels (ca).")
    print(f"p-value = {'< 1e-6' if p_anova < 1e-6 else f'{p_anova:.4e}'}")
    if p_anova < 0.05:
        print("Conclusion: Reject H0. Mean cholesterol differs by severity level.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")
except Exception as e:
    print(f"ANOVA error: {e}\n")

# ==== Kruskal-Wallis Test ====
print("=== Kruskal-Wallis Test (Non-Parametric) ===")
try:
    h_stat, p_kw = kruskal(*groups)
    print("H0: Cholesterol distributions are the same across severity levels (ca).")
    print(f"p-value = {'< 1e-6' if p_kw < 1e-6 else f'{p_kw:.4e}'}")
    if p_kw < 0.05:
        print("Conclusion: Reject H0. Cholesterol levels vary with severity.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")
except Exception as e:
    print(f"Kruskal-Wallis error: {e}\n")

# ==== Chi-square Test of Independence ====
print("=== Chi-square Test of Independence (Binned Cholesterol vs ca) ===")
try:
    df['chol_bin'] = pd.cut(df['chol'], bins=4, labels=["Low", "Mid-Low", "Mid-High", "High"])
    contingency = pd.crosstab(df['chol_bin'], df['ca'])
    chi2_stat, p_chi, dof, _ = chi2_contingency(contingency)
    print("H0: Serum cholesterol is independent of severity (ca).")
    print(f"p-value = {'< 1e-6' if p_chi < 1e-6 else f'{p_chi:.4e}'}")
    if p_chi < 0.05:
        print("Conclusion: Reject H0. Association detected.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")
except Exception as e:
    print(f"Chi-square error: {e}\n")

# ==== Permutation Test ====
print("=== Permutation Test (No Assumptions) ===")
try:
    np.random.seed(42)
    chol_values = chol_ca_df['chol'].values
    ca_labels = chol_ca_df['ca'].values

    # Compute observed SSB (Sum of Squares Between Groups)
    def group_means_variance(values, labels):
        unique_labels = np.unique(labels)
        overall_mean = values.mean()
        return sum([
            len(values[labels == g]) * (values[labels == g].mean() - overall_mean) ** 2
            for g in unique_labels
        ])

    obs_ssb = group_means_variance(chol_values, ca_labels)

    n_permutations = 10000
    perm_stats = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(ca_labels)
        perm_stat = group_means_variance(chol_values, shuffled)
        perm_stats.append(perm_stat)

    p_perm = np.mean(np.array(perm_stats) >= obs_ssb)
    print("H0: Cholesterol is unrelated to heart disease severity (ca).")
    print(f"Observed test stat (SSB) = {obs_ssb:.2f}")
    print(f"Permutation p-value = {'< 1e-6' if p_perm < 1e-6 else f'{p_perm:.4e}'}")
    if p_perm < 0.05:
        print("Conclusion: Reject H0. Cholesterol is related to severity.\n")
    else:
        print("Conclusion: Fail to reject H0.\n")
except Exception as e:
    print(f"Permutation test error: {e}\n")



#------------------------------Hypothesis 4----------------------------------------------------

df = pd.read_csv(r"heart_disease_uci.csv")

# Ensure proper types and remove missing/invalid rows
df = df[['ca', 'chol']].dropna()
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['chol'] = pd.to_numeric(df['chol'], errors='coerce')
df.dropna(inplace=True)

# Prepare cholesterol data grouped by 'ca' (0–3)
groups = [group['chol'].values for name, group in df.groupby('ca') if name in [0, 1, 2, 3]]

# ---- One-way ANOVA ----
f_stat, p_val_anova = f_oneway(*groups)

# ---- Kruskal-Wallis Test ----
h_stat, p_val_kw = kruskal(*groups)

# ---- Output ----
print("One-Way ANOVA")
print("Null Hypothesis: Mean cholesterol is equal across ca groups (0–3).")
print(f"p-value = {p_val_anova:.4e}")
if p_val_anova < 0.05:
    print("Conclusion: Reject H0. Cholesterol differs significantly between at least two ca groups.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference in cholesterol.\n")

print("Kruskal–Wallis Test")
print("Null Hypothesis: Distribution of cholesterol is the same across ca groups (0–3).")
print(f"p-value = {p_val_kw:.4e}")
if p_val_kw < 0.05:
    print("Conclusion: Reject H0. Cholesterol distribution differs significantly between groups.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference.\n")


#------------------------------Hypothesis 5----------------------------------------------------

df = pd.read_csv(r"heart_disease_uci.csv")  # Update path if needed

# Ensure numeric and clean data
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)  # 1 = disease, 0 = no disease
df['oldpeak'] = pd.to_numeric(df['oldpeak'], errors='coerce')
df = df[['num', 'oldpeak']].dropna()

# Split groups: with and without heart disease
oldpeak_disease = df[df['num'] == 1]['oldpeak']
oldpeak_healthy = df[df['num'] == 0]['oldpeak']

# ---- Independent Samples t-Test ----
t_stat, p_val_ttest = ttest_ind(oldpeak_disease, oldpeak_healthy, equal_var=False)  # Welch's t-test

# ---- Mann–Whitney U Test ----
u_stat, p_val_mwu = mannwhitneyu(oldpeak_disease, oldpeak_healthy, alternative='greater')  # one-tailed

# ---- Output ----
print("Independent Samples t-Test")
print("Null Hypothesis: Mean ST depression is the same in patients with and without heart disease.")
print("Alternative: ST depression is higher in patients with heart disease.")
print(f"p-value = {p_val_ttest:.4e}")
if p_val_ttest < 0.05:
    print("Conclusion: Reject H0. Patients with heart disease have significantly higher ST depression.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference.\n")

print("Mann–Whitney U Test (Non-parametric)")
print("Null Hypothesis: ST depression distribution is the same.")
print("Alternative: Heart disease patients have higher ST depression.")
print(f"p-value = {p_val_mwu:.4e}")
if p_val_mwu < 0.05:
    print("Conclusion: Reject H0. ST depression is significantly higher in heart disease patients.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference.\n")

#------------------------------Hypothesis 6----------------------------------------------------
df = pd.read_csv(r"heart_disease_uci.csv")

# --- Data Preparation ---
df = df[['num', 'age', 'chol', 'trestbps']].dropna()
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# --- Logistic Regression ---
X = df[['age', 'chol', 'trestbps']]
X = sm.add_constant(X)
y = df['num']
model = sm.Logit(y, X).fit(disp=0)

# --- Model Summary ---
print("\n=== Logistic Regression Summary ===")
print(model.summary())

# --- Odds Ratios with 95% CI ---
odds_ratios = np.exp(model.params)
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
conf_exp = np.exp(conf)
odds_df = pd.concat([odds_ratios, conf_exp], axis=1)
odds_df.columns = ['Odds Ratio', 'CI Lower', 'CI Upper']
print("\n=== Odds Ratios and 95% CI ===")
print(odds_df)

print("\n=== P-values with full precision ===")
with pd.option_context('display.float_format', '{:.16e}'.format):
    print(model.pvalues)

# --- Residuals and Influence ---
fitted_vals = model.predict(X)
residuals = model.resid_response
influence = model.get_influence()
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

# --- Plot Residuals vs Fitted ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(fitted_vals, residuals, edgecolor='k', alpha=0.7)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title("Residuals vs Fitted")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Response residuals")

# --- Plot Leverage vs Residuals with Cook's Distance as bubble size ---
bubble_size = 1000 * cooks_d
axes[1].scatter(leverage, residuals, s=bubble_size, alpha=0.5, edgecolors='k')
axes[1].set_xlabel("Leverage")
axes[1].set_ylabel("Response residuals")
axes[1].set_title("Leverage vs Residuals (Bubble = Cook's D)")

plt.tight_layout()
plt.show()

# --- Pearson Correlation Matrix ---
print("\n=== Pearson Correlation Matrix ===")
corr_matrix = df[['num', 'age', 'chol', 'trestbps']].corr()
print(corr_matrix.round(3))

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix: Age, Cholesterol, BP, Heart Disease")
plt.tight_layout()
plt.show()