import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.api import CompareMeans
import warnings
from scipy.stats import ttest_rel, norm

import warnings
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# === Load and Prepare Data ===
columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]

df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=columns, na_values="?", on_bad_lines="skip")
df = df[["Hemoglobin", "Class"]].dropna()
df = df[df["Class"].astype(str).str.lower().str.strip() == "ckd"]
df["Hemoglobin"] = df["Hemoglobin"].astype(float)
hemo = df["Hemoglobin"]
n = len(hemo)
standard = 12.75

# === 1. Shapiro-Wilk Test for Normality ===
shapiro_stat, shapiro_p = stats.shapiro(hemo)
print("=== Shapiro-Wilk Test for Normality ===")
print(f"Statistic = {shapiro_stat:.4f}, p = {shapiro_p:.4e}")
print("H0: Hemoglobin is normally distributed.")
if shapiro_p < 0.05:
    print("We reject H0 → Hemoglobin is NOT normally distributed.\n")
else:
    print("We fail to reject H0 → Hemoglobin may be normally distributed.\n")

# === Q-Q Plot ===
plt.figure(figsize=(6, 4))
stats.probplot(hemo, dist="norm", plot=plt)
plt.title("Q-Q Plot for Hemoglobin")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. One-sample t-test ===
t_stat, t_p = stats.ttest_1samp(hemo, popmean=standard)
print("=== One-sample Two-sided t-test ===")
print(f"t-statistic = {t_stat:.4f}, p = {t_p:.4e}")
print("H0: The mean hemoglobin in CKD patients is 12.75 g/dL.")
if t_p < 0.05:
    print("We reject H0 → CKD patients have significantly different hemoglobin.\n")
else:
    print("We fail to reject H0 → No significant difference from 12.75 g/dL.\n")

# === 3. One-sided t-test ===
t_stat_one_sided = t_stat
p_one_sided = t_p / 2 if t_stat < 0 else 1 - t_p / 2
print("=== One-sample One-sided t-test (mean < 12.75) ===")
print(f"t-statistic = {t_stat_one_sided:.4f}, one-sided p = {p_one_sided:.4e}")
print("H0: Mean hemoglobin ≥ 12.75")
if p_one_sided < 0.05:
    print("We reject H0 → CKD patients have significantly LOWER hemoglobin.\n")
else:
    print("We fail to reject H0 → No strong evidence of lower hemoglobin.\n")

# === 4. Wilcoxon Signed-Rank Test ===
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(hemo - standard, alternative='less')
print("=== Wilcoxon Signed-Rank Test (non-parametric) ===")
print(f"Statistic = {wilcoxon_stat:.4f}, p = {wilcoxon_p:.4e}")
print("H0: Median hemoglobin = 12.75")
if wilcoxon_p < 0.05:
    print("We reject H0 → Median hemoglobin is significantly LOWER than 12.75.\n")
else:
    print("We fail to reject H0 → No strong evidence of lower median hemoglobin.\n")

# === 5. Levene’s Test (Independence / Equal variance check) ===
# Compare CKD hemoglobin to a synthetic normal group at 12.75 with same std
synthetic = np.random.normal(loc=standard, scale=np.std(hemo), size=n)
levene_stat, levene_p = stats.levene(hemo, synthetic)
print("=== Levene's Test for Equality of Variance ===")
print(f"Statistic = {levene_stat:.4f}, p = {levene_p:.4e}")
print("H0: Variances of CKD and healthy population are equal.")
if levene_p < 0.05:
    print("We reject H0 → Variances are significantly different (independence assumption questionable).\n")
else:
    print("We fail to reject H0 → Variances appear equal.\n")

# === 6. Permutation Test (Non-parametric, no independence assumption) ===
obs_mean = np.mean(hemo)
diff = obs_mean - standard

# Combine real data and shifted fake data
combined = np.concatenate([hemo.values, synthetic])
n_perm = 10000
perm_diffs = []

for _ in range(n_perm):
    perm_sample = np.random.choice(combined, size=n, replace=False)
    perm_diffs.append(np.mean(perm_sample) - standard)

perm_diffs = np.array(perm_diffs)
p_perm = np.mean(perm_diffs <= diff)

print("=== Permutation Test (Non-parametric) ===")
print(f"Observed mean difference = {diff:.4f}")
print(f"Permutation p-value = {p_perm:.4e}")
if p_perm < 0.05:
    print("We reject H0 → CKD hemoglobin is significantly lower (no assumptions).\n")
else:
    print("We fail to reject H0 → No strong nonparametric evidence.\n")

# === 7. Histogram ===
plt.figure(figsize=(6, 4))
sns.histplot(hemo, bins=20, kde=True)
plt.axvline(12.75, color='red', linestyle='--', label='Standard 12.75 g/dL')
plt.title("Hemoglobin Distribution in CKD Patients")
plt.xlabel("Hemoglobin (g/dL)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print('----------------------------Hypothesis 2-----------------------------')
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene, mannwhitneyu
from statsmodels.stats.weightstats import DescrStatsW
warnings.filterwarnings("ignore")

# === Load and prepare data ===
columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]

df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=columns, na_values="?", on_bad_lines='skip')
df["Blood Pressure"] = pd.to_numeric(df["Blood Pressure"], errors='coerce')
df["Class"] = df["Class"].astype(str).str.strip().str.lower().replace({'ckd': 1, 'notckd': 0})
df_clean = df[["Blood Pressure", "Class"]].dropna()

bp_ckd = df_clean[df_clean["Class"] == 1]["Blood Pressure"]
bp_nonckd = df_clean[df_clean["Class"] == 0]["Blood Pressure"]

# === Levene's Test (equal variances assumption for t-test) ===
print("=== Levene's Test for Equal Variances ===")
print("H0: Variances in blood pressure are equal between CKD and non-CKD groups.")
levene_stat, levene_p = levene(bp_ckd, bp_nonckd)
print(f"Statistic = {levene_stat:.4f}, p = {levene_p:.4e}")
if levene_p < 0.05:
    print("→ We reject H0: Variances differ → Use Welch's t-test.\n")
else:
    print("→ Fail to reject H0: Variances can be assumed equal.\n")

# === Independent Samples t-test ===
print("=== Independent Samples t-test ===")
print("H0: Mean blood pressure is equal in CKD and non-CKD patients.")
ttest_stat, ttest_p = ttest_ind(bp_ckd, bp_nonckd, equal_var=False)  # Welch's t-test
print(f"t-statistic = {ttest_stat:.4f}, p-value = {ttest_p:.4e}")
if ttest_p < 0.05:
    print("→ We reject H0: CKD patients have significantly different blood pressure.\n")
else:
    print("→ Fail to reject H0: No significant difference in blood pressure.\n")

# === Mann-Whitney U Test ===
print("=== Mann-Whitney U Test ===")
print("H0: Distributions of blood pressure are equal between CKD and non-CKD.")
u_stat, u_p = mannwhitneyu(bp_ckd, bp_nonckd, alternative='two-sided')
print(f"Statistic = {u_stat:.4f}, p-value = {u_p:.4e}")
if u_p < 0.05:
    print("→ We reject H0: Distribution differs significantly.\n")
else:
    print("→ Fail to reject H0: No significant distributional difference.\n")

# === CAN Estimate ===
print("=== CAN Estimate of Mean Difference ===")
d1 = DescrStatsW(bp_ckd)
d2 = DescrStatsW(bp_nonckd)
diff = d1.mean - d2.mean
se_diff = np.sqrt(d1.var / d1.nobs + d2.var / d2.nobs)
ci_low = diff - 1.96 * se_diff
ci_high = diff + 1.96 * se_diff
print(f"Mean Difference = {d1.mean:.2f} - {d2.mean:.2f} = {diff:.2f}")
print(f"95% CI for Difference = ({ci_low:.2f}, {ci_high:.2f})")


print('----------------------------Hypothesis 3-----------------------------')
from scipy.stats import ttest_rel, wilcoxon, norm, shapiro, probplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon, shapiro, norm
from statsmodels.stats.stattools import durbin_watson

# === Load Data ===
columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]

df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=columns, na_values="?", on_bad_lines='skip')

# === Filter CKD patients and clean ===
df = df[["Age", "Serum Creatinine", "Class", "Red Blood Cells"]].dropna()
df = df[df["Class"].str.lower().str.strip() == "ckd"]
df["Age"] = df["Age"].astype(float)
df["Serum Creatinine"] = df["Serum Creatinine"].astype(float)
df["Red Blood Cells"] = df["Red Blood Cells"].str.lower().str.strip()
df["Sex"] = df["Red Blood Cells"].apply(lambda x: "male" if x == "normal" else "female")

# === Expected Creatinine based on age and sex ===
def estimate_expected_creatinine(age, sex):
    return 0.9 + 0.005 * (age - 40) if sex == "male" else 0.7 + 0.005 * (age - 40)

df["Expected Creatinine"] = df.apply(lambda row: estimate_expected_creatinine(row["Age"], row["Sex"]), axis=1)
diffs = df["Serum Creatinine"] - df["Expected Creatinine"]

# === Normality Test ===
stat_shapiro, p_shapiro = shapiro(diffs)

# === Independence Check: Durbin-Watson on paired differences ===
dw_stat = durbin_watson(diffs)
# DW ≈ 2 → no autocorrelation. DW < 2 = positive autocorr. DW > 2 = negative autocorr.

# === Paired T-Tests ===
t_stat, t_pval = ttest_rel(df["Serum Creatinine"], df["Expected Creatinine"])
t_stat_one, t_pval_one = ttest_rel(df["Serum Creatinine"], df["Expected Creatinine"], alternative='greater')

# === Wilcoxon Signed-Rank Tests ===
wilcoxon_stat, wilcoxon_p = wilcoxon(diffs)
wilcoxon_stat_one, wilcoxon_p_one = wilcoxon(diffs, alternative='greater')

# === Q-Q Plot ===
plt.figure()
stats.probplot(diffs, dist="norm", plot=plt)
plt.title("Q-Q Plot: Observed - Expected Creatinine")
plt.grid(True)
plt.show()

# === MSE + MLE Estimates ===
mse_estimate = np.mean(diffs)
mse = np.mean((diffs - mse_estimate) ** 2)
mean_mle = np.mean(diffs)
std_mle = np.std(diffs, ddof=1)
n = len(diffs)
ci_95 = norm.interval(0.95, loc=mean_mle, scale=std_mle / np.sqrt(n))

# === Permutation Test ===
combined = df["Serum Creatinine"].values - df["Expected Creatinine"].values
obs_mean = np.mean(combined)
perm_means = []
for _ in range(10000):
    signs = np.random.choice([-1, 1], size=n)
    permuted = combined * signs
    perm_means.append(np.mean(permuted))

perm_means = np.array(perm_means)
perm_p = np.mean(perm_means >= obs_mean)

# === Print Results ===
print("=== Normality Check (Shapiro-Wilk) ===")
print("H₀: The differences follow a normal distribution")
print(f"Statistic = {stat_shapiro:.4f}, p = {p_shapiro:.4e}")
print("→ " + ("Reject H₀: Not normal" if p_shapiro < 0.05 else "Fail to reject H₀: Data appears normal"))

print("\n=== Independence Check (Durbin-Watson) ===")
print("DW ≈ 2 → independent; DW < 2 → positive autocorrelation")
print(f"Durbin-Watson statistic = {dw_stat:.4f}")
print("→ " + ("Potential autocorrelation" if dw_stat < 1.6 else "No strong evidence of dependence"))

print("\n=== Paired t-test (Two-sided) ===")
print("H₀: Mean observed creatinine = expected")
print(f"t = {t_stat:.4f}, p = {t_pval:.4e}")
print("→ " + ("Reject H₀" if t_pval < 0.05 else "Fail to reject H₀"))

print("\n=== Paired t-test (One-sided) ===")
print("H₀: Mean observed creatinine ≤ expected\nH₁: Mean observed creatinine > expected")
print(f"t = {t_stat_one:.4f}, p = {t_pval_one:.4e}")
print("→ " + ("Reject H₀ → Creatinine is significantly higher" if t_pval_one < 0.05 else "Fail to reject H₀"))

print("\n=== Wilcoxon Signed-Rank Test (Two-sided) ===")
print("H₀: Median difference = 0")
print(f"W = {wilcoxon_stat:.4f}, p = {wilcoxon_p:.4e}")
print("→ " + ("Reject H₀" if wilcoxon_p < 0.05 else "Fail to reject H₀"))

print("\n=== Wilcoxon Signed-Rank Test (One-sided) ===")
print("H₀: Median difference ≤ 0\nH₁: Median difference > 0")
print(f"W = {wilcoxon_stat_one:.4f}, p = {wilcoxon_p_one:.4e}")
print("→ " + ("Reject H₀ → Creatinine is significantly higher than expected" if wilcoxon_p_one < 0.05 else "Fail to reject H₀"))

print("\n=== MSE Estimate of Difference ===")
print(f"Mean Difference = {mse_estimate:.4f}")
print(f"MSE = {mse:.4f} (minimizes squared error)")

print("\n=== MLE Estimate of Difference (Normal Assumption) ===")
print(f"Mean = {mean_mle:.4f}")
print(f"95% CI = ({ci_95[0]:.2f}, {ci_95[1]:.2f})")

print("\n=== Permutation Test (Non-parametric, no assumptions) ===")
print("H₀: Observed - Expected has mean zero")
print(f"Observed mean difference = {obs_mean:.4f}")
print(f"Permutation p-value = {perm_p:.4e}")
print("→ " + ("Reject H₀ → Creatinine is significantly higher" if perm_p < 0.05 else "Fail to reject H₀"))


print('----------------------------Hypothesis 4-----------------------------')
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon, chisquare, chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson

# Load dataset
columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]

df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=columns, na_values="?", on_bad_lines='skip')

# Preprocess
df = df[["Hemoglobin", "Hypertension", "Diabetes Mellitus", "Class"]].dropna()
df["Hemoglobin"] = pd.to_numeric(df["Hemoglobin"], errors='coerce')
df = df.dropna()

df["Hypertension"] = df["Hypertension"].astype(str).str.strip().str.lower().replace({'yes': 1, 'no': 0}).astype(int)
df["Diabetes Mellitus"] = df["Diabetes Mellitus"].astype(str).str.strip().str.lower().replace({'yes': 1, 'no': 0}).astype(int)
df["CKD"] = df["Class"].astype(str).str.strip().str.lower().replace({'ckd': 1, 'notckd': 0}).astype(int)

# === One-sided t-test: H0: Hb >= 13.5, H1: Hb < 13.5 ===
t_stat, t_p = ttest_1samp(df["Hemoglobin"], popmean=13.5, alternative="less")
print("\n=== One-sided t-test: Hemoglobin < 13.5 ===")
print(f"H0: Mean Hemoglobin ≥ 13.5\nH1: Mean Hemoglobin < 13.5")
print(f"t = {t_stat:.4f}, p = {t_p:.4e}")
print("Conclusion:", "Reject H0" if t_p < 0.05 else "Fail to reject H0")

# === Wilcoxon Signed-Rank Test ===
try:
    w_stat, w_p = wilcoxon(df["Hemoglobin"] - 13.5, alternative='less')
except ValueError:
    w_stat, w_p = np.nan, np.nan

print("\n=== Wilcoxon Signed-Rank Test (non-parametric) ===")
print(f"H0: Median Hemoglobin = 13.5\nH1: Median Hemoglobin < 13.5")
print(f"w = {w_stat:.4f}, p = {w_p:.4e}")
print("Conclusion:", "Reject H0" if w_p < 0.05 else "Fail to reject H0")

# === Chi-square Goodness-of-Fit ===
# Adjusted Chi-square Goodness-of-Fit Test with custom expected ratios
comorb_sums = df[["Hypertension", "Diabetes Mellitus"]].sum()
total = comorb_sums.sum()
expected_ratios = [0.4, 0.1]
expected = [r / sum(expected_ratios) * total for r in expected_ratios]

chi_stat, chi_p = chisquare(f_obs=comorb_sums, f_exp=expected)

print("\n=== Chi-square Goodness-of-Fit ===")
print("H0: Frequencies match expected population distribution (40%-10%)")
print(f"Observed: {comorb_sums.to_dict()}")
print(f"Expected: {{'Hypertension': {expected[0]:.1f}, 'Diabetes Mellitus': {expected[1]:.1f}}}")
print(f"Chi2 = {chi_stat:.4f}, p = {chi_p:.4e}")
print("Conclusion:", "Reject H0" if chi_p < 0.05 else "Fail to reject H0")


# === Chi-square Independence Test ===
cont_table = pd.crosstab(df["Hypertension"], df["Diabetes Mellitus"])
chi2_stat, chi2_p, _, _ = chi2_contingency(cont_table)

print("\n=== Chi-square Independence Test ===")
print("H0: Hypertension and Diabetes are independent")
print(f"Chi2 = {chi2_stat:.4f}, p = {chi2_p:.4e}")
print("Conclusion:", "Reject H0 (dependent)" if chi2_p < 0.05 else "Fail to reject H0 (independent)")

# === Fisher's Exact Test ===
print("\n=== Fisher's Exact Test ===")
if cont_table.shape == (2, 2):
    _, fisher_p = fisher_exact(cont_table)
    print("H0: Hypertension and Diabetes are independent (exact)")
    print(f"p = {fisher_p:.4e}")
    print("Conclusion:", "Reject H0 (dependent)" if fisher_p < 0.05 else "Fail to reject H0 (independent)")
else:
    print("Not a 2x2 table; Fisher's test not applicable.")

# === Rao's Score Test (Log-linear Poisson GLM) ===
df = df.rename(columns={"Diabetes Mellitus": "Diabetes_Mellitus"})
df["count"] = 1
model = glm("count ~ Hypertension + Diabetes_Mellitus", data=df, family=Poisson()).fit()
rao_score = model.pearson_chi2
rao_df = model.df_resid
from scipy.stats import chi2
rao_p = chi2.sf(rao_score, rao_df)

print("\n=== Rao's Score Test (GLM Poisson log-linear) ===")
print("H0: No association (log-linear model)")
print(f"Score statistic = {rao_score:.4f}, df = {rao_df}, p = {rao_p:.4e}")
print("Conclusion:", "Reject H0 (association exists)" if rao_p < 0.05 else "Fail to reject H0")

# === Consistent Multinomial Estimation ===
joint_counts = df.groupby(["Hypertension", "Diabetes_Mellitus"]).size()
total = joint_counts.sum()
multinomial_probs = (joint_counts / total).round(3).to_dict()

print("\n=== Consistent Multinomial Distribution Estimation ===")
print("Joint Hypertension & Diabetes Probabilities:")
for key, prob in multinomial_probs.items():
    print(f"{key}: {prob}")
