import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn

# Create output directory
output_dir = 'regression_ckd'
os.makedirs(output_dir, exist_ok=True)


# Function to format p-values
def format_p_value(p):
    if p < 0.0001:
        return f"{p:.1e}"
    return f"{p:.4f}"


# Print scikit-learn version for compatibility check
print(f"scikit-learn version: {sklearn.__version__}")

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
missing_cols = [col for col in columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in dataset: {missing_cols}. Available columns: {list(df.columns)}")

# Convert numerical columns to numeric and impute
num_cols = ['Age', 'Blood Pressure', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine',
            'Sodium', 'Potassium', 'Hemoglobin', 'Packed Cell Volume',
            'White Blood Cell Count', 'Red Blood Cell Count']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"Imputed missing values in {col} with median: {median_val:.2f}")

# Impute categorical columns
cat_cols = ['Specific Gravity', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell',
            'Pus Cell clumps', 'Bacteria', 'Hypertension', 'Diabetes Mellitus',
            'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia', 'Class']

for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"Imputed missing values in {col} with mode: {mode_val}")

# Encode categorical variables
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

# === Create RBC Vectors ===
rbc_ckd = df[df['Class'] == 1]['Red Blood Cell Count'].dropna()
rbc_non_ckd = df[df['Class'] == 0]['Red Blood Cell Count'].dropna()

print(f"\nSample sizes: CKD (n={len(rbc_ckd)}), Non-CKD (n={len(rbc_non_ckd)})")

# Initialize output for saving results
results = []

# === Shapiro-Wilk Test for Normality ===
# Null hypothesis: The data is normally distributed
shapiro_ckd = stats.shapiro(rbc_ckd)
shapiro_non_ckd = stats.shapiro(rbc_non_ckd)
results.append(f"Shapiro-Wilk Test for Normality (CKD RBC): p-value = {format_p_value(shapiro_ckd.pvalue)}")
results.append(
    f"Interpretation: {'Not normal' if shapiro_ckd.pvalue < 0.05 else 'Normal'} (p < 0.05 indicates non-normality)")
results.append(f"Shapiro-Wilk Test for Normality (Non-CKD RBC): p-value = {format_p_value(shapiro_non_ckd.pvalue)}")
results.append(
    f"Interpretation: {'Not normal' if shapiro_non_ckd.pvalue < 0.05 else 'Normal'} (p < 0.05 indicates non-normality)")
print("\n" + "\n".join(results[-4:]))

# === Chi-Square Test for Independence ===
# Bin RBC for chi-square test
bins = pd.qcut(df['Red Blood Cell Count'], q=4, duplicates='drop').cat.categories
df['RBC_Binned'] = pd.qcut(df['Red Blood Cell Count'], q=4, labels=False, duplicates='drop')
contingency_table = pd.crosstab(df['RBC_Binned'], df['Class'])
chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
results.append(f"Chi-Square Test for Independence: p-value = {format_p_value(chi2_p)}")
results.append(
    f"Interpretation: {'Dependent' if chi2_p < 0.05 else 'Independent'} (p < 0.05 indicates RBC bins and CKD status are dependent)")
results.append(
    "Note: Chi-square test performed despite non-normality; results should be interpreted cautiously as normality is typically assumed for continuous data binning")
print("\n" + "\n".join(results[-3:]))

# === Levene's Test for Equal Variances ===
# Null hypothesis: Variances are equal
levene_stat, levene_p = stats.levene(rbc_ckd, rbc_non_ckd)
results.append(f"Levene's Test for Equal Variances: p-value = {format_p_value(levene_p)}")
results.append(
    f"Interpretation: {'Unequal variances' if levene_p < 0.05 else 'Equal variances'} (p < 0.05 indicates unequal variances)")
print("\n" + "\n".join(results[-2:]))

# === Kolmogorov-Smirnov (KS) Test ===
# Null hypothesis: Both samples come from the same distribution
ks_stat, ks_p = stats.ks_2samp(rbc_ckd, rbc_non_ckd)
results.append(f"Kolmogorov-Smirnov Test: p-value = {format_p_value(ks_p)}")
results.append(
    f"Interpretation: {'Different distributions' if ks_p < 0.05 else 'Same distribution'} (p < 0.05 indicates different distributions)")
print("\n" + "\n".join(results[-2:]))


# === Two-Sample Anderson-Darling Test ===
# Null hypothesis: Both samples come from the same distribution
def anderson_darling_2samp(x, y, n_permutations=1000):
    # Combine and sort samples
    combined = np.concatenate([x, y])
    combined_sorted = np.sort(combined)
    n1, n2 = len(x), len(y)
    n = n1 + n2

    # Compute ECDFs
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    ecdf_x = np.searchsorted(x_sorted, combined_sorted, side='right') / n1
    ecdf_y = np.searchsorted(y_sorted, combined_sorted, side='right') / n2

    # Anderson-Darling statistic
    h = n1 * n2 / n
    ad_stat = h * np.sum((ecdf_x - ecdf_y) ** 2 / (combined_sorted * (1 - combined_sorted / np.max(combined_sorted))))

    # Permutation test for p-value
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:n1]
        perm_y = combined[n1:]
        perm_x_sorted = np.sort(perm_x)
        perm_y_sorted = np.sort(perm_y)
        perm_ecdf_x = np.searchsorted(perm_x_sorted, combined_sorted, side='right') / n1
        perm_ecdf_y = np.searchsorted(perm_y_sorted, combined_sorted, side='right') / n2
        perm_stat = h * np.sum(
            (perm_ecdf_x - perm_ecdf_y) ** 2 / (combined_sorted * (1 - combined_sorted / np.max(combined_sorted))))
        if perm_stat >= ad_stat:
            count += 1
    ad_p = count / n_permutations

    return ad_stat, ad_p


ad_stat, ad_p = anderson_darling_2samp(rbc_ckd, rbc_non_ckd)
results.append(f"Anderson-Darling Test (Two-Sample): p-value = {format_p_value(ad_p)}")
results.append(
    f"Interpretation: {'Different distributions' if ad_p < 0.05 else 'Same distribution'} (p < 0.05 indicates different distributions)")
print("\n" + "\n".join(results[-2:]))

# === Mean/Median Difference Tests (if KS or Anderson-Darling indicate different distributions) ===
if ks_p < 0.05 or ad_p < 0.05:
    # T-Test (assumes normality and equal variances)
    t_stat, t_p = stats.ttest_ind(rbc_ckd, rbc_non_ckd, equal_var=True)
    results.append(f"T-Test (Two-Sided): p-value = {format_p_value(t_p)}")
    results.append(
        f"Interpretation: {'Different means' if t_p < 0.05 else 'No difference in means'} (p < 0.05 indicates different means, assumes normality and equal variances)")

    # Welch's T-Test (assumes normality, unequal variances)
    welch_stat, welch_p = stats.ttest_ind(rbc_ckd, rbc_non_ckd, equal_var=False)
    results.append(f"Welch's T-Test (Two-Sided): p-value = {format_p_value(welch_p)}")
    results.append(
        f"Interpretation: {'Different means' if welch_p < 0.05 else 'No difference in means'} (p < 0.05 indicates different means, assumes normality)")

    # Mann-Whitney U Test (non-parametric, tests medians)
    mw_stat, mw_p = stats.mannwhitneyu(rbc_ckd, rbc_non_ckd, alternative='two-sided')
    results.append(f"Mann-Whitney U Test (Two-Sided): p-value = {format_p_value(mw_p)}")
    results.append(
        f"Interpretation: {'Different medians' if mw_p < 0.05 else 'No difference in medians'} (p < 0.05 indicates different medians, non-parametric)")


    # Permutation Test (non-parametric, tests means)
    def permutation_test(x, y, n_permutations=1000):
        observed_diff = np.mean(x) - np.mean(y)
        combined = np.concatenate([x, y])
        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[:len(x)]) - np.mean(combined[len(x):])
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        return count / n_permutations


    perm_p = permutation_test(rbc_ckd, rbc_non_ckd)
    results.append(f"Permutation Test (Two-Sided): p-value = {format_p_value(perm_p)}")
    results.append(
        f"Interpretation: {'Different means' if perm_p < 0.05 else 'No difference in means'} (p < 0.05 indicates different means, non-parametric)")

    # Wilcoxon Signed-Rank Test (not applicable for independent samples)
    results.append(
        "Wilcoxon Signed-Rank Test: Skipped (requires paired data, not applicable for independent CKD and non-CKD samples)")
    results.append("Interpretation: Test not suitable for this data")

    print("\n" + "\n".join(results[-10:]))

    # === One-Sided Tests (Non-CKD mean/median > CKD) ===
    # T-Test (one-sided: non-CKD > CKD)
    t_stat, t_p_two_sided = stats.ttest_ind(rbc_non_ckd, rbc_ckd, equal_var=True)
    t_p_one_sided = t_p_two_sided / 2 if np.mean(rbc_non_ckd) > np.mean(rbc_ckd) else 1 - (t_p_two_sided / 2)
    results.append(f"T-Test (One-Sided, Non-CKD > CKD): p-value = {format_p_value(t_p_one_sided)}")
    results.append(
        f"Interpretation: {'Non-CKD mean > CKD' if t_p_one_sided < 0.05 else 'No evidence Non-CKD mean > CKD'} (p < 0.05 indicates Non-CKD mean is greater)")

    # Welch's T-Test (one-sided: non-CKD > CKD)
    welch_stat, welch_p_two_sided = stats.ttest_ind(rbc_non_ckd, rbc_ckd, equal_var=False)
    welch_p_one_sided = welch_p_two_sided / 2 if np.mean(rbc_non_ckd) > np.mean(rbc_ckd) else 1 - (
                welch_p_two_sided / 2)
    results.append(f"Welch's T-Test (One-Sided, Non-CKD > CKD): p-value = {format_p_value(welch_p_one_sided)}")
    results.append(
        f"Interpretation: {'Non-CKD mean > CKD' if welch_p_one_sided < 0.05 else 'No evidence Non-CKD mean > CKD'} (p < 0.05 indicates Non-CKD mean is greater)")

    # Mann-Whitney U Test (one-sided: non-CKD > CKD)
    mw_stat, mw_p_one_sided = stats.mannwhitneyu(rbc_non_ckd, rbc_ckd, alternative='greater')
    results.append(f"Mann-Whitney U Test (One-Sided, Non-CKD > CKD): p-value = {format_p_value(mw_p_one_sided)}")
    results.append(
        f"Interpretation: {'Non-CKD median > CKD' if mw_p_one_sided < 0.05 else 'No evidence Non-CKD median > CKD'} (p < 0.05 indicates Non-CKD median is greater)")


    # Permutation Test (one-sided: non-CKD > CKD)
    def permutation_test_one_sided(x, y, n_permutations=1000):
        observed_diff = np.mean(x) - np.mean(y)
        combined = np.concatenate([x, y])
        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[:len(x)]) - np.mean(combined[len(x):])
            if perm_diff >= observed_diff:
                count += 1
        return count / n_permutations


    perm_p_one_sided = permutation_test_one_sided(rbc_non_ckd, rbc_ckd)
    results.append(f"Permutation Test (One-Sided, Non-CKD > CKD): p-value = {format_p_value(perm_p_one_sided)}")
    results.append(
        f"Interpretation: {'Non-CKD mean > CKD' if perm_p_one_sided < 0.05 else 'No evidence Non-CKD mean > CKD'} (p < 0.05 indicates Non-CKD mean is greater)")

    print("\n" + "\n".join(results[-8:]))
else:
    results.append("Mean/Median Difference Tests: Skipped (both KS and Anderson-Darling indicate same distribution)")
    results.append("Interpretation: No evidence of different distributions, so difference tests are not performed")
    print("\n" + "\n".join(results[-2:]))

# Save results to file
with open(os.path.join(output_dir, 'rbc_analysis_ckd.txt'), 'w') as f:
    f.write("\n".join(results))

print(f"\nResults saved to {os.path.join(output_dir, 'rbc_analysis_ckd.txt')}")



import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
import sklearn

# Create output directory
output_dir = 'regression_ckd'
os.makedirs(output_dir, exist_ok=True)

# Function to format p-values (for consistency, though not used here)
def format_p_value(p):
    if p < 0.0001:
        return f"{p:.1e}"
    return f"{p:.4f}"

# Print scikit-learn version for compatibility check
print(f"scikit-learn version: {sklearn.__version__}")

# === Load and Preprocess CKD Dataset ===
ckd_columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]

try:
    ckd_df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=ckd_columns, na_values=["?", "\t?"],
                         on_bad_lines='skip')
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'Chronic_Kidney_Disease.csv' was not found. Please ensure the file is in the correct directory.")

# Verify CKD column names
missing_cols = [col for col in ckd_columns if col not in ckd_df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in CKD dataset: {missing_cols}. Available columns: {list(ckd_df.columns)}")

# Convert numerical columns to numeric and impute
ckd_num_cols = ['Age', 'Blood Pressure', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine',
                'Sodium', 'Potassium', 'Hemoglobin', 'Packed Cell Volume',
                'White Blood Cell Count', 'Red Blood Cell Count']

for col in ckd_num_cols:
    ckd_df[col] = pd.to_numeric(ckd_df[col], errors='coerce')
    median_val = ckd_df[col].median()
    ckd_df[col] = ckd_df[col].fillna(median_val)
    print(f"CKD: Imputed missing values in {col} with median: {median_val:.2f}")

# Impute categorical columns and handle unexpected values
ckd_cat_cols = ['Specific Gravity', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell',
                'Pus Cell clumps', 'Bacteria', 'Hypertension', 'Diabetes Mellitus',
                'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia', 'Class']

for col in ckd_cat_cols:
    mode_val = ckd_df[col].mode()[0] if not ckd_df[col].mode().empty else 'unknown'
    ckd_df[col] = ckd_df[col].fillna(mode_val)
    print(f"CKD: Imputed missing values in {col} with mode: {mode_val}")

# Encode CKD categorical variables with explicit handling
ckd_df['Specific Gravity'] = ckd_df['Specific Gravity'].astype(str)
ckd_df['Albumin'] = pd.to_numeric(ckd_df['Albumin'], errors='coerce').fillna(0).astype(int)
ckd_df['Sugar'] = pd.to_numeric(ckd_df['Sugar'], errors='coerce').fillna(0).astype(int)
ckd_df['Red Blood Cells'] = ckd_df['Red Blood Cells'].map({'normal': 1, 'abnormal': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Pus Cell'] = ckd_df['Pus Cell'].map({'normal': 1, 'abnormal': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Pus Cell clumps'] = ckd_df['Pus Cell clumps'].map({'present': 1, 'notpresent': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Bacteria'] = ckd_df['Bacteria'].map({'present': 1, 'notpresent': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Hypertension'] = ckd_df['Hypertension'].map({'yes': 1, 'no': 0, 1: 1, 0: 0, ' yes': 1, ' no': 0}).fillna(0)
ckd_df['Diabetes Mellitus'] = ckd_df['Diabetes Mellitus'].map({'yes': 1, 'no': 0, 1: 1, 0: 0, ' yes': 1, ' no': 0}).fillna(0)
ckd_df['Coronary Artery Disease'] = ckd_df['Coronary Artery Disease'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Appetite'] = ckd_df['Appetite'].map({'good': 1, 'poor': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Pedal Edema'] = ckd_df['Pedal Edema'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Anemia'] = ckd_df['Anemia'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
ckd_df['Class'] = ckd_df['Class'].map({'ckd': 1, 'notckd': 0, 1: 1, 0: 0}).fillna(0)

# Debug: Print CKD class distribution
print("\nCKD Dataset: Class Distribution")
print(ckd_df['Class'].value_counts())

# Debug: Check for NaN in CKD features and print statistics
features = ['Hemoglobin', 'Red Blood Cell Count', 'Blood Pressure', 'Hypertension', 'Diabetes Mellitus']
print("\nCKD Dataset: NaN counts in features before training")
print(ckd_df[features].isna().sum())
print("\nCKD Dataset: Feature Statistics")
print(ckd_df[features].describe())

# Ensure no NaN in selected features
for col in features:
    if ckd_df[col].isna().sum() > 0:
        if col in ckd_num_cols:
            ckd_df[col] = ckd_df[col].fillna(ckd_df[col].median())
            print(f"CKD: Re-imputed {col} with median: {ckd_df[col].median():.2f}")
        else:
            ckd_df[col] = ckd_df[col].fillna(ckd_df[col].mode()[0])
            print(f"CKD: Re-imputed {col} with mode: {ckd_df[col].mode()[0]}")

# === Load and Preprocess Heart Disease Dataset ===
try:
    heart_df = pd.read_csv("heart_disease_uci.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'heart_disease_uci.csv' was not found. Please ensure the file is in the correct directory.")

# Debug: Print data types and unique values
print("\nHeart Disease Dataset: Data Types")
print(heart_df.dtypes)
print("\nUnique values in 'sex':", heart_df['sex'].unique())
print("Unique values in 'age':", heart_df['age'].unique())
print("Unique values in 'num':", heart_df['num'].unique())

# Explicitly encode sex as 0=Female, 1=Male
heart_df['sex'] = heart_df['sex'].map({'Female': 0, 'Male': 1})

# Ensure numeric chest pain types
heart_df['cp'] = heart_df['cp'].map({
    'typical angina': 1,
    'atypical angina': 2,
    'non-anginal': 3,
    'asymptomatic': 4
}) if heart_df['cp'].dtype == 'object' else heart_df['cp']

# Encode other categorical columns
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
encoding_notes = {'sex': {0: 'Female', 1: 'Male'}}
for col in categorical_cols:
    heart_df[col] = heart_df[col].astype('category')
    encoding_notes[col] = dict(enumerate(heart_df[col].cat.categories, start=1))
    heart_df[col] = heart_df[col].cat.codes + 1  # +1 to start codes from 1

# Handle boolean columns
for bool_col in ['fbs', 'exang']:
    heart_df[bool_col] = heart_df[bool_col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

# Ensure trestbps is numeric, impute if needed, and rename to Blood Pressure
heart_df['trestbps'] = pd.to_numeric(heart_df['trestbps'], errors='coerce')
if heart_df['trestbps'].isna().sum() > 0:
    median_trestbps = heart_df['trestbps'].median()
    heart_df['trestbps'] = heart_df['trestbps'].fillna(median_trestbps)
    print(f"Heart Disease: Imputed missing values in trestbps with median: {median_trestbps:.2f}")
heart_df = heart_df.rename(columns={'trestbps': 'Blood Pressure'})

# Create target column
heart_df['num'] = pd.to_numeric(heart_df['num'], errors='coerce')
heart_df['target'] = heart_df['num'].apply(lambda x: 0 if x == 0 else 1)

# Drop id and dataset columns
heart_df = heart_df.drop(['id', 'dataset'], axis=1, errors='ignore')

# Drop rows with any NaN values
heart_df = heart_df.dropna()
print(f"\nHeart Disease Dataset: Dropped rows with NaN values. Remaining rows: {len(heart_df)}")

# Generate synthetic columns
# Hemoglobin: 13.5 + 1.5 * sex - 0.007 * age - Halfnormal(0,1) * target
heart_df['Hemoglobin'] = 13.5 + 1.5 * heart_df['sex'] - 0.007 * heart_df['age'] - stats.norm.rvs(loc=0, scale=5, size=len(heart_df)) * heart_df['target']

# Red Blood Cell Count: 0.45 + 0.23 * Hemoglobin + 0.35 * sex - 0.005 * age - Halfnormal(0,0.1) * target
heart_df['Red Blood Cell Count'] = 0.45 + 0.23 * heart_df['Hemoglobin'] + 0.35 * heart_df['sex'] - 0.005 * heart_df['age'] - stats.norm.rvs(loc=0, scale=0.1, size=len(heart_df)) * heart_df['target']

# Diabetes Mellitus: Logistic regression
# P(Diabetes Mellitus) = sigmoid(-1.1244 - 1.1043 * Hemoglobin + 1.1014 * age)
logit_diabetes = -1.1244 - 1.1043 * heart_df['Hemoglobin'] + 1.1014 * heart_df['age']
prob_diabetes = 1 / (1 + np.exp(-logit_diabetes))
heart_df['Diabetes Mellitus'] = np.random.binomial(1, prob_diabetes)

# Hypertension: Logistic regression
# P(Hypertension) = sigmoid(-0.9732 - 1.1247 * Hemoglobin + 1.0086 * age - 1.0482 * Red Blood Cell Count)
logit_hypertension = -0.9732 - 1.1247 * heart_df['Hemoglobin'] + 1.0086 * heart_df['age'] - 1.0482 * heart_df['Red Blood Cell Count']
prob_hypertension = 1 / (1 + np.exp(-logit_hypertension))
heart_df['Hypertension'] = np.random.binomial(1, prob_hypertension)

# Debug: Check for NaN in heart disease features and print statistics
print("\nHeart Disease Dataset: NaN counts in features")
print(heart_df[features].isna().sum())
print("\nHeart Disease Dataset: Feature Statistics")
print(heart_df[features].describe())

# === Train Logistic Regression on CKD Dataset ===
X_ckd = ckd_df[features]
y_ckd = ckd_df['Class']

# Scale numerical features
scaler = StandardScaler()
X_ckd_scaled = X_ckd.copy()
X_ckd_scaled[['Hemoglobin', 'Red Blood Cell Count', 'Blood Pressure']] = scaler.fit_transform(
    X_ckd[['Hemoglobin', 'Red Blood Cell Count', 'Blood Pressure']])

# Debug: Check for NaN in scaled features
print("\nCKD Dataset: NaN counts in scaled features before training")
print(X_ckd_scaled.isna().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_ckd_scaled, y_ckd, test_size=0.2, random_state=42)

# Train logistic regression with balanced class weights
model = LogisticRegression(C=2.4155819892879817, penalty='l2', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_error = log_loss(y_train, model.predict_proba(X_train))
test_error = log_loss(y_test, model.predict_proba(X_test))

# Print and save results
results = [
    f"CKD Logistic Regression Results:",
    f"Training Accuracy: {train_accuracy:.4f}",
    f"Test Accuracy: {test_accuracy:.4f}",
    f"Training F1 Score: {train_f1:.4f}",
    f"Test F1 Score: {test_f1:.4f}",
    f"Training Error (Log Loss): {train_error:.4f}",
    f"Test Error (Log Loss): {test_error:.4f}"
]

print("\n" + "\n".join(results))

# === Predict CKD on Heart Disease Dataset ===
X_heart = heart_df[features]
X_heart_scaled = X_heart.copy()
X_heart_scaled[['Hemoglobin', 'Red Blood Cell Count', 'Blood Pressure']] = scaler.transform(
    X_heart[['Hemoglobin', 'Red Blood Cell Count', 'Blood Pressure']])

# Debug: Print predicted probabilities for heart disease dataset
ckd_probs = model.predict_proba(X_heart_scaled)[:, 1]  # Probability of ckd == 1
print("\nHeart Disease Dataset: Predicted CKD Probabilities (first 5)")
print(ckd_probs[:5])

heart_df['ckd'] = model.predict(X_heart_scaled)

# Print head of heart disease dataset and CKD counts
print("\nHeart Disease Dataset Head (with ckd column):")
print(heart_df[['age', 'sex', 'Hemoglobin', 'Red Blood Cell Count', 'Blood Pressure', 'Hypertension', 'Diabetes Mellitus', 'target', 'ckd']].head())
print("\nCKD Counts in Heart Disease Dataset:")
print(f"CKD (1): {heart_df['ckd'].value_counts().get(1, 0)}")
print(f"Non-CKD (0): {heart_df['ckd'].value_counts().get(0, 0)}")

# Save results to file
with open(os.path.join(output_dir, 'ckd_logistic_results.txt'), 'w') as f:
    f.write("\n".join(results))

print(f"\nResults saved to {os.path.join(output_dir, 'ckd_logistic_results.txt')}")


# population research

import pandas as pd
import numpy as np
import os
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

# Check for scipy installation
try:
    from scipy.stats import binomtest, f_oneway, kruskal, ttest_ind, mannwhitneyu, chi2_contingency
except ImportError as e:
    print(f"Error: SciPy is not installed or inaccessible. Please install it using 'pip install scipy'. Details: {e}")
    raise

# Create output directory
output_dir = 'regression_ckd'
os.makedirs(output_dir, exist_ok=True)


# Function to format p-values
def format_p_value(p):
    if p < 0.0001:
        return f"{p:.1e}"
    return f"{p:.4f}"


# Debug: Verify heart_df columns and distributions
print("\nHeart Disease Dataset: Columns")
print(heart_df.columns)
print("\nCKD Distribution:")
print(heart_df['ckd'].value_counts())
print("\nHeart Disease (target) Distribution:")
print(heart_df['target'].value_counts())

# === Transform Features ===
# Age: 0 = young (<55), 1 = old (>=55)
heart_df['age_binary'] = (heart_df['age'] >= 55).astype(int)

# Blood Pressure: 0 = low/normal (<140), 1 = high (>=140)
heart_df['Blood Pressure_binary'] = (heart_df['Blood Pressure'] >= 140).astype(int)

# Debug: Print transformed feature distributions
print("\nAge Binary Distribution (0=young, 1=old):")
print(heart_df['age_binary'].value_counts())
print("\nBlood Pressure Binary Distribution (0=low/normal, 1=high):")
print(heart_df['Blood Pressure_binary'].value_counts())
print("\nSex Distribution (0=Female, 1=Male):")
print(heart_df['sex'].value_counts())

# === Hypothesis Testing with Bonferroni Correction ===
# Hypothesis 1: Most people with heart disease (target == 1) have CKD (ckd == 1)
heart_disease = heart_df[heart_df['target'] == 1]
n_heart_disease = len(heart_disease)
n_heart_disease_ckd = len(heart_disease[heart_disease['ckd'] == 1])
prop_heart_disease_ckd = n_heart_disease_ckd / n_heart_disease if n_heart_disease > 0 else 0
# Z-test
z_stat1, p_val1 = proportions_ztest(n_heart_disease_ckd, n_heart_disease, value=0.5, alternative='larger')
# Binomial test (non-parametric)
binom_p_val1 = binomtest(n_heart_disease_ckd, n_heart_disease, p=0.5, alternative='greater').pvalue
# Bonferroni correction for hypothesis tests
hyp_p_vals = [p_val1, binom_p_val1]
_, hyp_p_vals_corrected, _, _ = multipletests(hyp_p_vals, alpha=0.05, method='bonferroni')
p_val1_corrected, binom_p_val1_corrected = hyp_p_vals_corrected
hyp1_result = f"Hypothesis 1: Most people with heart disease have CKD (proportion > 0.5)\n" \
              f"Null Hypothesis: The proportion of people with heart disease who have CKD is <= 0.5\n" \
              f"Proportion: {prop_heart_disease_ckd:.4f}\n" \
              f"Z-test - Z-statistic: {z_stat1:.4f}, P-value: {format_p_value(p_val1)}, Corrected P-value: {format_p_value(p_val1_corrected)}\n" \
              f"Binomial test - P-value: {format_p_value(binom_p_val1)}, Corrected P-value: {format_p_value(binom_p_val1_corrected)}\n" \
              f"Conclusion: {'Reject null (proportion > 0.5)' if p_val1_corrected < 0.05 else 'Fail to reject null (proportion <= 0.5)'} (Z-test, corrected)"

# Hypothesis 2: Most people with CKD (ckd == 1) do not have heart disease (target == 0)
ckd_patients = heart_df[heart_df['ckd'] == 1]
n_ckd = len(ckd_patients)
n_ckd_no_heart = len(ckd_patients[ckd_patients['target'] == 0])
prop_ckd_no_heart = n_ckd_no_heart / n_ckd if n_ckd > 0 else 0
# Z-test
z_stat2, p_val2 = proportions_ztest(n_ckd_no_heart, n_ckd, value=0.5, alternative='larger')
# Binomial test (non-parametric)
binom_p_val2 = binomtest(n_ckd_no_heart, n_ckd, p=0.5, alternative='greater').pvalue
# Bonferroni correction for hypothesis tests
hyp_p_vals = [p_val2, binom_p_val2]
_, hyp_p_vals_corrected, _, _ = multipletests(hyp_p_vals, alpha=0.05, method='bonferroni')
p_val2_corrected, binom_p_val2_corrected = hyp_p_vals_corrected
hyp2_result = f"Hypothesis 2: Most people with CKD do not have heart disease (proportion > 0.5)\n" \
              f"Null Hypothesis: The proportion of people with CKD who do not have heart disease is <= 0.5\n" \
              f"Proportion: {prop_ckd_no_heart:.4f}\n" \
              f"Z-test - Z-statistic: {z_stat2:.4f}, P-value: {format_p_value(p_val2)}, Corrected P-value: {format_p_value(p_val2_corrected)}\n" \
              f"Binomial test - P-value: {format_p_value(binom_p_val2)}, Corrected P-value: {format_p_value(binom_p_val2_corrected)}\n" \
              f"Conclusion: {'Reject null (proportion > 0.5)' if p_val2_corrected < 0.05 else 'Fail to reject null (proportion <= 0.5)'} (Z-test, corrected)"

# === Population Analysis ===
# Define populations
pop1 = heart_df[(heart_df['target'] == 1) & (heart_df['ckd'] == 1)]  # Both heart disease and CKD
pop2 = heart_df[(heart_df['ckd'] == 1) & (heart_df['target'] == 0)]  # Only CKD
pop3 = heart_df[(heart_df['target'] == 1) & (heart_df['ckd'] == 0)]  # Only heart disease
pop4 = heart_df[(heart_df['ckd'] == 0) & (heart_df['target'] == 0)]  # Neither
populations = {
    'Population 1 (Heart Disease + CKD)': pop1,
    'Population 2 (Only CKD)': pop2,
    'Population 3 (Only Heart Disease)': pop3,
    'Population 4 (Neither)': pop4
}

# Feature combinations
combinations = [
    ('old men', (1, 1)),  # age=1, sex=1
    ('young men', (0, 1)),  # age=0, sex=1
    ('old women', (1, 0)),  # age=1, sex=0
    ('young women', (0, 0)),  # age=0, sex=0
    ('men with high BP', (None, 1, 1)),  # sex=1, BP=1
    ('women with high BP', (None, 0, 1)),  # sex=0, BP=1
    ('old with high BP', (1, None, 1)),  # age=1, BP=1
    ('young with high BP', (0, None, 1))  # age=0, BP=1
]

# Initialize results for population analysis
pop_results = []
binary_vars = ['age_binary', 'sex', 'Blood Pressure_binary']
pop_test_p_vals = []  # Collect p-values for Bonferroni correction across populations

# Analyze each population
for name, pop in populations.items():
    pop_results.append(f"\n{name}:")
    pop_results.append(f"Size: {len(pop)}")
    if len(pop) == 0:
        pop_results.append("No individuals in this population")
        continue

    # Proportions of feature combinations
    pop_results.append("\nFeature Combination Proportions:")
    comb_proportions = []
    comb_counts = []
    for comb_name, comb_vals in combinations:
        if len(comb_vals) == 2:  # age, sex
            age_val, sex_val = comb_vals
            count = len(pop[(pop['age_binary'] == age_val) & (pop['sex'] == sex_val)])
            prop = count / len(pop) if len(pop) > 0 else 0
            comb_proportions.append((comb_name, prop, count))
            comb_counts.append(count)
            pop_results.append(f"{comb_name}: {prop:.4f}")
        else:  # age, sex, BP
            age_val, sex_val, bp_val = comb_vals
            if age_val is None:
                count = len(pop[(pop['sex'] == sex_val) & (pop['Blood Pressure_binary'] == bp_val)])
            else:
                count = len(pop[(pop['age_binary'] == age_val) & (pop['Blood Pressure_binary'] == bp_val)])
            prop = count / len(pop) if len(pop) > 0 else 0
            comb_proportions.append((comb_name, prop, count))
            comb_counts.append(count)
            pop_results.append(f"{comb_name}: {prop:.4f}")


    # Dominant combination test (one-sample proportion test)
    max_prop = max(comb_proportions, key=lambda x: x[1], default=(None, 0, 0))
    max_comb_name, max_prop_value, max_count = max_prop
    if len(pop) > 0:
        _, max_p_val = proportions_ztest(max_count, len(pop), value=0.5, alternative='larger')
        pop_test_p_vals.append(max_p_val)
        pop_results.append(f"\nDominant Combination Test (One-Sample Proportion):")
        pop_results.append(f"Most frequent combination: {max_comb_name} (Proportion: {max_prop_value:.4f})")
        pop_results.append(f"Null Hypothesis: The proportion of '{max_comb_name}' in {name} is <= 0.5")
        pop_results.append(f"Z-test - P-value: {format_p_value(max_p_val)}")

    # Characterization based on dominant combinations
    char = []
    old_prop = pop['age_binary'].mean()
    male_prop = pop['sex'].mean()
    high_bp_prop = pop['Blood Pressure_binary'].mean()
    if old_prop > 0.5:
        char.append("mostly older")
    else:
        char.append("mostly younger")
    if male_prop > 0.5:
        char.append("mostly men")
    else:
        char.append("mostly women")
    if high_bp_prop > 0.5:
        char.append("with high blood pressure")
    else:
        char.append("with low/normal blood pressure")
    pop_results.append(f"Characterization: {', '.join(char)}")

# Bonferroni correction for population tests (ANOVA, Kruskal-Wallis, T-test, Mann-Whitney U, one-sample proportion)
if pop_test_p_vals:
    _, pop_test_p_vals_corrected, _, _ = multipletests(pop_test_p_vals, alpha=0.05, method='bonferroni')
    pop_results.append(
        f"\nCorrected P-values for Population Tests (Bonferroni): {[format_p_value(p) for p in pop_test_p_vals_corrected]}")

# Chi-squared tests for binary variables
pop_results.append("\nChi-squared Tests Between Populations:")
chi2_results = []
pop_pairs = [(i, j) for i in range(1, 5) for j in range(i + 1, 5)]
for var in binary_vars:
    chi2_results.append(f"\nChi-squared test for {var}:")
    p_vals = []
    for i, j in pop_pairs:
        pop_i_name = f"Population {i} (Heart Disease + CKD)" if i == 1 else \
            f"Population {i} (Only CKD)" if i == 2 else \
                f"Population {i} (Only Heart Disease)" if i == 3 else \
                    f"Population {i} (Neither)"
        pop_j_name = f"Population {j} (Heart Disease + CKD)" if j == 1 else \
            f"Population {j} (Only CKD)" if j == 2 else \
                f"Population {j} (Only Heart Disease)" if j == 3 else \
                    f"Population {j} (Neither)"
        pop_i = populations[pop_i_name]
        pop_j = populations[pop_j_name]
        if len(pop_i) == 0 or len(pop_j) == 0:
            chi2_results.append(f"Pop {i} vs Pop {j}: Skipped (empty population)")
            continue
        # Create contingency table
        pop_i_counts = pop_i[var].value_counts().reindex([0, 1], fill_value=0)
        pop_j_counts = pop_j[var].value_counts().reindex([0, 1], fill_value=0)
        contingency_table = pd.DataFrame({
            pop_i_name: pop_i_counts,
            pop_j_name: pop_j_counts
        }).fillna(0).values
        try:
            chi2, p, dof, _ = chi2_contingency(contingency_table)
            p_vals.append(p)
            chi2_results.append(f"Pop {i} vs Pop {j}: Chi2={chi2:.4f}, P-value={format_p_value(p)}")
        except:
            chi2_results.append(f"Pop {i} vs Pop {j}: Skipped (invalid contingency table)")
    # Bonferroni correction
    if p_vals:
        _, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='bonferroni')
        chi2_results.append(f"Corrected P-values (Bonferroni): {[format_p_value(p) for p in p_vals_corrected]}")

# Combine all results
all_results = [hyp1_result, hyp2_result] + pop_results + chi2_results

# Print results
print("\n" + "\n".join(all_results))

# Save results to file
with open(os.path.join(output_dir, 'population_analysis.txt'), 'w') as f:
    f.write("\n".join(all_results))

print(f"\nResults saved to {os.path.join(output_dir, 'population_analysis.txt')}")
print("\nNote: The CKD distribution shows heavy imbalance (272 CKD vs. 31 non-CKD). "
      "This may indicate issues with the logistic regression model or feature alignment. "
      "Please share the original script's output (CKD class distribution, feature statistics, "
      "predicted probabilities) to diagnose further.")