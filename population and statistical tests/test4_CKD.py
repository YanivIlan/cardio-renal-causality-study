import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ks_2samp, chisquare
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import optuna
import statsmodels.api as sm
import patsy
import warnings

warnings.filterwarnings("ignore")

# === Load and Clean Data ===
columns = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
    "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria",
    "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
    "Potassium", "Hemoglobin", "Packed  Cell Volume", "White Blood Cell Count",
    "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia", "Class"
]
df = pd.read_csv("Chronic_Kidney_Disease.csv", header=None, names=columns, na_values="?", on_bad_lines='skip')

# Clean the binary mappings first
df = df.replace({"yes": 1, "no": 0})

# Select and clean the required columns
df = df[["Age", "Hemoglobin", "Packed  Cell Volume", "Class", "Hypertension", "Diabetes Mellitus",
         "Coronary Artery Disease"]].dropna()

# === Convert types and filter CKD patients ===
df["Age"] = df["Age"].astype(float)
df["Hemoglobin"] = df["Hemoglobin"].astype(float)
df["Packed  Cell Volume"] = df["Packed  Cell Volume"].astype(float)
df = df[df["Class"].str.strip().str.lower() == "ckd"]


# === Define anemia severity ===
def classify_anemia(hgb):
    if hgb < 8:
        return "severe"
    elif hgb < 11:
        return "moderate"
    else:
        return "mild"


df["Anemia_Level"] = df["Hemoglobin"].apply(classify_anemia)
label_encoder = LabelEncoder()
df["Anemia_Label"] = label_encoder.fit_transform(df["Anemia_Level"])

# === Normality test for Age (Kolmogorov-Smirnov vs Normal) ===
ks_stat, ks_p = ks_2samp(df["Age"], np.random.normal(df["Age"].mean(), df["Age"].std(), len(df)))
print("\n=== Kolmogorov-Smirnov Test for Age (vs Normal) ===")
print(f"KS statistic = {ks_stat:.4f}, p-value = {ks_p:.4e} → {'Not normal' if ks_p < 0.05 else 'Normal'}")

# === Correlation Tests ===
pearson_corr, p_p = pearsonr(df["Age"], df["Hemoglobin"])
spearman_corr, p_s = spearmanr(df["Age"], df["Hemoglobin"])
print("\n=== Age-Hemoglobin Correlation ===")
print(f"Pearson r = {pearson_corr:.4f}, p = {p_p:.4e} → {'Negative correlation' if pearson_corr < 0 else 'Positive'}")
print(
    f"Spearman ρ = {spearman_corr:.4f}, p = {p_s:.4e} → {'Monotonic correlation' if p_s < 0.05 else 'No clear monotonic trend'}")

# === Define Features and Labels ===
X = df[["Age"]]
y = df["Anemia_Label"]


# === OPTUNA for XGBoost Classification ===
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return np.mean(scores)


print("\nRunning Optuna hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best parameters:", study.best_params)

# === Final Model with Best Params ===
model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X, y)
y_pred = model.predict(X)

# === Classification Metrics ===
print("\n=== Classification Report ===")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))

# === Print Confusion Matrix ===
print("\n=== Confusion Matrix ===")
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

# === Optional ROC-AUC ===
y_proba = model.predict_proba(X)
try:
    auc = roc_auc_score(y, y_proba, multi_class='ovr')
    print(f"Overall AUC: {auc:.4f}")
except:
    print("AUC not computable for current label setup.")

# === Fixed Multinomial Goodness-of-Fit Test for Comorbidity Patterns ===
print("\n=== Cleaning Comorbidity Data ===")


# Function to clean and convert binary columns
def clean_binary_column(series, col_name):
    # Convert to string and strip whitespace
    cleaned = series.astype(str).str.strip().str.lower()

    # Print unique values for debugging
    print(f"Unique values in {col_name}: {cleaned.unique()}")

    # Create mapping dictionary
    mapping = {
        'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0,
        'true': 1, 'false': 0, 't': 1, 'f': 0
    }

    # Apply mapping
    result = cleaned.map(mapping)

    # Check for unmapped values
    if result.isnull().any():
        unmapped = cleaned[result.isnull()].unique()
        print(f"Warning: Unmapped values in {col_name}: {unmapped}")
        # Fill unmapped values with mode (most common value)
        mode_val = result.mode().iloc[0] if len(result.mode()) > 0 else 0
        result = result.fillna(mode_val)
        print(f"Filled unmapped values with mode: {mode_val}")

    return result.astype(int)


# Clean the comorbidity columns
df["Hypertension_clean"] = clean_binary_column(df["Hypertension"], "Hypertension")
df["Diabetes_clean"] = clean_binary_column(df["Diabetes Mellitus"], "Diabetes Mellitus")
df["CAD_clean"] = clean_binary_column(df["Coronary Artery Disease"], "Coronary Artery Disease")

# Create comorbidity patterns
df["pattern"] = list(zip(df["Hypertension_clean"], df["Diabetes_clean"], df["CAD_clean"]))
pattern_counts = df["pattern"].value_counts().sort_index()

print(f"\nComorbidity pattern counts:")
for pattern, count in pattern_counts.items():
    print(f"HTN={pattern[0]}, DM={pattern[1]}, CAD={pattern[2]}: {count} patients")

# Only proceed with chi-square test if we have all 8 possible patterns
all_patterns = [(h, d, c) for h in [0, 1] for d in [0, 1] for c in [0, 1]]
missing_patterns = [p for p in all_patterns if p not in pattern_counts.index]

if missing_patterns:
    print(f"\nMissing patterns: {missing_patterns}")
    # Add missing patterns with count 0
    for pattern in missing_patterns:
        pattern_counts[pattern] = 0
    pattern_counts = pattern_counts.sort_index()

# Based on a medically plausible prior distribution (adjustable if needed)
expected_ratios = np.array([
    0.0675,  # (0,0,0)
    0.0225,  # (0,0,1)
    0.045,  # (0,1,0)
    0.015,  # (0,1,1)
    0.3825,  # (1,0,0)
    0.1275,  # (1,0,1)
    0.255,  # (1,1,0)
    0.085  # (1,1,1)
])
expected_ratios /= expected_ratios.sum()
expected_counts = expected_ratios * pattern_counts.sum()

# Filter out patterns with expected count < 5 for valid chi-square test
valid_mask = expected_counts >= 5
if not all(valid_mask):
    print(f"Warning: Some expected counts < 5. Test may not be reliable.")
    print(f"Expected counts: {expected_counts}")

chi_stat, chi_p = chisquare(f_obs=pattern_counts.values, f_exp=expected_counts)
print("\n=== Multinomial Goodness-of-Fit ===")
print("H0: Observed comorbidity patterns match expected multinomial distribution")
print("Patterns (HTN, DM, CAD):", pattern_counts.index.tolist())
print("Observed:", pattern_counts.values)
print("Expected:", expected_counts.round(1))
print(f"Chi2 = {chi_stat:.4f}, p = {chi_p:.4e}")
print("Conclusion:", "Reject H0" if chi_p < 0.05 else "Fail to reject H0")

# === Log-Linear Model (Poisson GLM) for 3-way Contingency Table ===
loglin_df = df.copy()
loglin_df["HTN"] = loglin_df["Hypertension_clean"]
loglin_df["DM"] = loglin_df["Diabetes_clean"]
loglin_df["CAD"] = loglin_df["CAD_clean"]

counts = loglin_df.groupby(["HTN", "DM", "CAD"]).size().reset_index(name="count")

# Only proceed if we have sufficient data
if len(counts) > 0:
    formula = 'count ~ HTN + DM + CAD + HTN:DM + HTN:CAD + DM:CAD'
    try:
        y_mat, X_mat = patsy.dmatrices(formula, counts, return_type="dataframe")
        glm_model = sm.GLM(y_mat, X_mat, family=sm.families.Poisson()).fit()
        print("\n=== Log-Linear Model (Poisson GLM) ===")
        print(glm_model.summary())
    except Exception as e:
        print(f"\nLog-linear model failed: {e}")
        print("This may be due to insufficient data or perfect separation.")
else:
    print("\nInsufficient data for log-linear model.")