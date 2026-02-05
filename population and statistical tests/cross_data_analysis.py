import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, anderson_ksamp, shapiro, levene
import warnings
from scipy.stats import ks_2samp, norm
from scipy.stats import anderson_ksamp
import numpy as np


def compare_age_distribution():
    # === Load datasets ===
    ckd_columns = [
        "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
        "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
        "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
        "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
        "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
        "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
    ]
    ckd_df = pd.read_csv('Chronic_Kidney_Disease.csv', header=None, names=ckd_columns, na_values="?", on_bad_lines='skip')
    heart_df = pd.read_csv('heart_disease_uci.csv')

    # === Extract and clean Age columns ===
    ckd_age = pd.to_numeric(ckd_df["Age"], errors='coerce').dropna()

    if 'age' in heart_df.columns:
        heart_age = pd.to_numeric(heart_df['age'], errors='coerce').dropna()
    elif 'Age' in heart_df.columns:
        heart_age = pd.to_numeric(heart_df['Age'], errors='coerce').dropna()
    else:
        print("Error: No 'age' column found in heart dataset.")
        return

    if len(ckd_age) == 0 or len(heart_age) == 0:
        print("Insufficient age data in one or both datasets.")
        return

    print(f"Number of CKD ages: {len(ckd_age)}, Heart Disease ages: {len(heart_age)}")

    # === Normality Check ===
    print("\n--- Normality Check (Shapiro-Wilk) ---")
    for label, data in zip(["CKD", "Heart"], [ckd_age, heart_age]):
        stat, p = shapiro(data)
        print(f"{label} Age: W = {stat:.4f}, p = {p:.4e} → ", end="")
        print("Not normal" if p < 0.05 else "Normal")

    # === Equality of Variance ===
    levene_stat, levene_p = levene(ckd_age, heart_age)
    print("\n--- Levene's Test for Equal Variance ---")
    print(f"Statistic = {levene_stat:.4f}, p = {levene_p:.4e} → ", end="")
    print("Reject equal variance" if levene_p < 0.05 else "Equal variances assumed")

    # === Independent Samples t-test ===
    t_stat, t_p = ttest_ind(ckd_age, heart_age, equal_var=True)
    print("\n--- Independent Samples t-test ---")
    print("H0: μ_age(CKD) = μ_age(Heart)")
    print("H1: μ_age(CKD) ≠ μ_age(Heart)")
    print(f"t-statistic = {t_stat:.4f}, p-value = {t_p:.4e}")
    print("Conclusion:", "Reject H₀" if t_p < 0.05 else "Fail to reject H₀")

    # === Mann-Whitney U Test ===
    u_stat, u_p = mannwhitneyu(ckd_age, heart_age, alternative="two-sided")
    print("\n--- Mann-Whitney U Test ---")
    print("H0: Age distributions are equal")
    print("H1: Age distributions differ")
    print(f"U statistic = {u_stat:.4f}, p-value = {u_p:.4e}")
    print("Conclusion:", "Reject H₀" if u_p < 0.05 else "Fail to reject H₀")

    # === Kolmogorov–Smirnov Test ===
    ks_stat, ks_p = ks_2samp(ckd_age, heart_age)
    print("\n--- Kolmogorov–Smirnov Test ---")
    print("H0: CKD and Heart age come from the same distribution")
    print("H1: Distributions differ in location, scale, or shape")
    print(f"KS statistic = {ks_stat:.4f}, p-value = {ks_p:.4e}")
    print("Conclusion:", "Reject H₀" if ks_p < 0.05 else "Fail to reject H₀")

    # === Anderson–Darling Two-Sample Test ===
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            ad_result = anderson_ksamp([ckd_age.values, heart_age.values])
            print("\n--- Anderson–Darling Two-Sample Test ---")
            print("H0: Age samples come from the same continuous distribution")
            print("H1: Age samples come from different distributions")
            print(f"Statistic = {ad_result.statistic:.4f}, p-value = {ad_result.significance_level:.4f}")
            print("Conclusion:", "Reject H₀" if ad_result.significance_level < 0.05 else "Fail to reject H₀")
        except Exception as e:
            print("Anderson–Darling test failed:", e)

    # === Assumption Summary ===
    print("\n--- Assumptions of Each Test ---")
    print("1. Shapiro-Wilk: Assumes normality (for checking normality of each group)")
    print("2. Levene's Test: Assumes independence, tests equal variances")
    print("3. t-test: Assumes normality + equal variances + independence")
    print("4. Mann-Whitney: Does not assume normality but assumes independence")
    print("5. Kolmogorov-Smirnov: Non-parametric, assumes independence")
    print("6. Anderson-Darling: Non-parametric, assumes continuous distributions")
    print("\n→ All non-parametric tests (Mann-Whitney, KS, AD) do not require normality.")
    print("→ None of the tests fully eliminate the assumption of independence (random sampling still expected).")


import pandas as pd
import numpy as np
from scipy.stats import levene, fligner, shapiro
from statsmodels.sandbox.stats.runs import runstest_1samp


def compare_bp_variability():
    # === Load datasets ===
    ckd_columns = [
        "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
        "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
        "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
        "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
        "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
        "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
    ]
    ckd_df = pd.read_csv(r"Chronic_Kidney_Disease.csv", header=None, names=ckd_columns, na_values="?",
                         on_bad_lines='skip')
    heart_df = pd.read_csv(r"heart_disease_uci.csv")

    # === Extract and clean blood pressure columns ===
    ckd_bp = pd.to_numeric(ckd_df["Blood Pressure"], errors='coerce').dropna()
    heart_bp = pd.to_numeric(heart_df.get('trestbps', pd.Series([])), errors='coerce').dropna()

    if len(ckd_bp) < 2 or len(heart_bp) < 2:
        print("Insufficient BP data for at least one group.")
        return

    print(f"CKD BP count: {len(ckd_bp)}, Heart BP count: {len(heart_bp)}")

    # === Normality Check ===
    print("\n--- Shapiro-Wilk Normality Test ---")
    for label, data in zip(["CKD", "Heart"], [ckd_bp, heart_bp]):
        stat, p = shapiro(data)
        print(f"{label} BP: W = {stat:.4f}, p = {p:.4e} →", "Not normal" if p < 0.05 else "Normal")

    # === Independence Check: Runs test (approximate, for randomness) ===
    print("\n--- Runs Test for Independence ---")
    for label, data in zip(["CKD", "Heart"], [ckd_bp, heart_bp]):
        median_val = np.median(data)
        signs = np.where(data > median_val, 1, 0)
        z, p = runstest_1samp(signs)
        print(f"{label} BP: Z = {z:.4f}, p = {p:.4e} →", "Likely independent" if p > 0.05 else "Dependence detected")

    # === Levene's Test (center='mean') ===
    stat_levene, p_levene = levene(ckd_bp, heart_bp, center='mean')
    print("\n--- Levene's Test for Equality of Variances (Parametric) ---")
    print("H0: Variances are equal (homoscedasticity)")
    print("H1: Variances are different (heteroscedasticity)")
    print(f"Statistic = {stat_levene:.4f}, p-value = {p_levene:.4e}")
    print("Conclusion:", "Reject H₀" if p_levene < 0.05 else "Fail to reject H₀")

    # === Brown-Forsythe Test (center='median') ===
    stat_bf, p_bf = levene(ckd_bp, heart_bp, center='median')
    print("\n--- Brown-Forsythe Test (Levene’s with Median) ---")
    print("H0: Variances are equal")
    print("H1: Variances are different")
    print(f"Statistic = {stat_bf:.4f}, p-value = {p_bf:.4e}")
    print("Conclusion:", "Reject H₀" if p_bf < 0.05 else "Fail to reject H₀")

    # === Fligner-Killeen Test (Non-parametric, robust) ===
    stat_fligner, p_fligner = fligner(ckd_bp, heart_bp)
    print("\n--- Fligner-Killeen Test (Robust Non-Parametric) ---")
    print("H0: Scale parameters are equal")
    print("H1: Scale parameters differ")
    print(f"Statistic = {stat_fligner:.4f}, p-value = {p_fligner:.4e}")
    print("Conclusion:", "Reject H₀" if p_fligner < 0.05 else "Fail to reject H₀")

    # === Kolmogorov–Smirnov Test ===
    ks_stat, ks_p = ks_2samp(ckd_bp, heart_bp)
    print("\n--- Kolmogorov–Smirnov Test ---")
    print("H0: CKD and Heart BP come from the same distribution")
    print("H1: Distributions differ in location, scale, or shape")
    print(f"KS statistic = {ks_stat:.4f}, p-value = {ks_p:.4e}")
    print("Conclusion:", "Reject H₀" if ks_p < 0.05 else "Fail to reject H₀")

    # === Anderson–Darling Two-Sample Test ===
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            ad_result = anderson_ksamp([ckd_bp.values, heart_bp.values])
            print("\n--- Anderson–Darling Two-Sample Test ---")
            print("H0: Samples come from the same continuous distribution")
            print("H1: Samples come from different distributions")
            print(f"Statistic = {ad_result.statistic:.4f}, p-value = {ad_result.significance_level:.4f}")
            print("Conclusion:", "Reject H₀" if ad_result.significance_level < 0.05 else "Fail to reject H₀")
        except Exception as e:
            print("Anderson–Darling test failed:", e)

    # === Summary of Assumptions ===
    print("\n--- Assumption Summary ---")
    print("1. Shapiro-Wilk: Normality of each group")
    print("2. Runs Test: Assesses randomness/independence")
    print("3. Levene/Brown-Forsythe: Equal variances, assumes independence")
    print("4. Fligner: Non-parametric, robust, assumes independence")
    print("5. Kolmogorov-Smirnov: Non-parametric, assumes independence")
    print("6. Anderson–Darling: Non-parametric, assumes continuity and independence")



from scipy.stats import ks_2samp, anderson_ksamp

def compare_bp_distribution_shape(ckd_path: str, heart_path: str):
    # === Load datasets ===
    ckd_columns = [
        "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
        "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
        "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
        "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
        "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
        "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
    ]
    ckd_df = pd.read_csv(r"Chronic_Kidney_Disease.csv", header=None, names=ckd_columns, na_values="?",
                         on_bad_lines='skip')
    heart_df = pd.read_csv(r"heart_disease_uci.csv")

    # === Extract and clean BP columns ===
    ckd_bp = pd.to_numeric(ckd_df["Blood Pressure"], errors='coerce').dropna()
    heart_bp = pd.to_numeric(heart_df["trestbps"], errors='coerce').dropna()

    if len(ckd_bp) < 2 or len(heart_bp) < 2:
        print("Insufficient blood pressure data in one or both datasets.")
        return

    print(f"CKD BP count: {len(ckd_bp)}, Heart BP count: {len(heart_bp)}")

    # === Normalize both to standard normal (for K-S test assumption) ===
    ckd_norm = (ckd_bp - ckd_bp.mean()) / ckd_bp.std()
    heart_norm = (heart_bp - heart_bp.mean()) / heart_bp.std()

    # === Kolmogorov-Smirnov Test (parametric) ===
    ks_stat, ks_p = ks_2samp(ckd_norm, heart_norm)
    print("\n--- Kolmogorov-Smirnov Test (Parametric) ---")
    print("H0: BP follows the same normal distribution in both datasets")
    print("H1: BP distributions differ in location, scale, or shape")
    print(f"KS statistic = {ks_stat:.4f}, p-value = {ks_p:.4e}")
    print("Conclusion:", "Reject H₀" if ks_p < 0.05 else "Fail to reject H₀")

    # === Anderson-Darling Two-Sample Test (non-parametric) ===
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ad_result = anderson_ksamp([ckd_bp.values, heart_bp.values])
        print("\n--- Anderson-Darling Two-Sample Test (Non-parametric) ---")
        print("H0: Both samples come from the same continuous distribution")
        print("H1: Samples come from different distributions")
        print(f"Statistic = {ad_result.statistic:.4f}, p-value = {ad_result.significance_level:.4f}")
        print("Conclusion:", "Reject H₀" if ad_result.significance_level < 0.05 else "Fail to reject H₀")
    except Exception as e:
        print("Anderson-Darling test failed:", e)

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import Table2x2


def compare_hypertension_prevalence(ckd_path: str, heart_path: str):
    # === Load datasets ===
    ckd_columns = [
        "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
        "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
        "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
        "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
        "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
        "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
    ]
    ckd_df = pd.read_csv(ckd_path, header=None, names=ckd_columns, na_values="?", on_bad_lines='skip')
    heart_df = pd.read_csv(heart_path)

    # === CKD: Clean Hypertension column ===
    ckd_htn = ckd_df["Hypertension"].astype(str).str.strip().str.lower()
    ckd_htn = ckd_htn[ckd_htn.isin(["yes", "no", "1", "0"])]

    # Fix the pandas downcasting warning
    ckd_htn = ckd_htn.replace({"yes": 1, "no": 0, "1": 1, "0": 0})
    ckd_htn = pd.to_numeric(ckd_htn, errors='coerce').astype("int64")

    n_ckd = len(ckd_htn)
    n_htn_ckd = ckd_htn.sum()

    # === Heart: define HTN as trestbps > 140 mmHg ===
    heart_bp = pd.to_numeric(heart_df["trestbps"], errors='coerce').dropna()
    n_heart = len(heart_bp)
    n_htn_heart = (heart_bp > 140).sum()

    if n_ckd == 0 or n_heart == 0:
        print("Insufficient data to compare hypertension prevalence.")
        return

    print(f"\nCKD: {n_htn_ckd} / {n_ckd} hypertensive patients ({n_htn_ckd / n_ckd * 100:.1f}%)")
    print(f"Heart: {n_htn_heart} / {n_heart} hypertensive patients ({n_htn_heart / n_heart * 100:.1f}%)")

    # === Parametric Test: Z-test for Two Proportions ===
    count = np.array([n_htn_ckd, n_htn_heart])
    nobs = np.array([n_ckd, n_heart])
    z_stat, z_p = proportions_ztest(count, nobs)

    print("\n--- Z-Test for Two Proportions (Parametric) ---")
    print("H0: Hypertension rates are equal (CKD = Heart)")
    print("H1: Hypertension rates differ")
    print(f"Z statistic = {z_stat:.4f}, p-value = {z_p:.4e}")
    print("Conclusion:", "Reject H₀" if z_p < 0.05 else "Fail to reject H₀")

    # === Fisher's Exact Test ===
    contingency_table = [
        [n_htn_ckd, n_ckd - n_htn_ckd],
        [n_htn_heart, n_heart - n_htn_heart]
    ]

    try:
        oddsratio, p_fisher = fisher_exact(contingency_table)
        print("\n--- Fisher's Exact Test (Non-parametric) ---")
        print("H0: Hypertension prevalence is independent of disease group")
        print("H1: Prevalence differs by group")
        print(f"Odds Ratio = {oddsratio:.4f}, p-value = {p_fisher:.4e}")
        print("Conclusion:", "Reject H₀" if p_fisher < 0.05 else "Fail to reject H₀")
    except Exception as e:
        print("Fisher's Exact Test failed:", e)

    # === Alternative: Chi-square test and Barnard's test (corrected) ===
    try:
        # Chi-square test as additional parametric alternative
        from scipy.stats import chi2_contingency
        chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
        print("\n--- Chi-square Test of Independence ---")
        print("H0: Hypertension prevalence is independent of disease group")
        print("H1: Prevalence differs by group")
        print(f"Chi-square statistic = {chi2_stat:.4f}, p-value = {chi2_p:.4e}")
        print("Conclusion:", "Reject H₀" if chi2_p < 0.05 else "Fail to reject H₀")

    except Exception as e:
        print("Chi-square test failed:", e)

    # === Corrected Barnard's test (if available) ===
    try:
        # Try using barnard_exact from scipy (newer versions)
        from scipy.stats import barnard_exact
        result = barnard_exact(contingency_table)
        print("\n--- Barnard's Exact Test ---")
        print("H0: Hypertension prevalence is equal across groups")
        print("H1: Prevalence differs by group")
        print(f"Statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4e}")
        print("Conclusion:", "Reject H₀" if result.pvalue < 0.05 else "Fail to reject H₀")

    except ImportError:
        print("\n--- Barnard's Exact Test ---")
        print("Barnard's exact test not available in this scipy version")
    except Exception as e:
        print("Barnard's test failed:", e)

    # === Summary Statistics ===
    print("\n=== SUMMARY ===")
    print(f"CKD hypertension rate: {n_htn_ckd / n_ckd * 100:.1f}%")
    print(f"Heart disease hypertension rate: {n_htn_heart / n_heart * 100:.1f}%")
    print(f"Difference: {(n_htn_ckd / n_ckd - n_htn_heart / n_heart) * 100:.1f} percentage points")
    print(f"Odds ratio: {oddsratio:.2f} (CKD patients are {oddsratio:.2f}x more likely to have hypertension)")



print("------------Hypothesis 1------------")
compare_age_distribution()

print("------------Hypothesis 2------------")
compare_bp_variability()

print("------------Hypothesis 3------------")
compare_bp_distribution_shape("Chronic_Kidney_Disease.csv", "heart.csv")

print("------------Hypothesis 4------------")
compare_hypertension_prevalence("Chronic_Kidney_Disease.csv", "heart_disease_uci.csv")

