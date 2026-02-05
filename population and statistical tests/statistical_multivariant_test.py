import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.multivariate.manova import MANOVA
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings('ignore')

# PERMANOVA implementation
def permanova(X, y, n_permutations=99, distance_metric='euclidean'):
    def calculate_f_stat(X, y):
        unique_groups = np.unique(y)
        n_groups = len(unique_groups)
        n_total = len(y)

        distances = pairwise_distances(X, metric=distance_metric)
        within_ss = 0
        for group in unique_groups:
            group_mask = y == group
            if np.sum(group_mask) > 1:
                group_distances = distances[group_mask][:, group_mask]
                within_ss += np.sum(group_distances ** 2) / (2 * np.sum(group_mask))

        total_ss = np.sum(distances ** 2) / (2 * n_total)
        between_ss = total_ss - within_ss

        df_between = n_groups - 1
        df_within = n_total - n_groups

        if df_within > 0 and within_ss > 0:
            f_stat = (between_ss / df_between) / (within_ss / df_within)
        else:
            f_stat = 0
        return f_stat

    observed_f = calculate_f_stat(X, y)
    permuted_f_stats = []
    for _ in range(n_permutations):
        permuted_y = np.random.permutation(y)
        permuted_f = calculate_f_stat(X, permuted_y)
        permuted_f_stats.append(permuted_f)

    p_value = np.sum(np.array(permuted_f_stats) >= observed_f) / (n_permutations + 1)  # +1 to avoid division by zero
    return observed_f, p_value, permuted_f_stats

# Load UCI Heart Disease dataset
try:
    df = pd.read_csv('heart_disease_uci.csv')
except FileNotFoundError:
    print("Error: 'heart_disease_uci.csv' not found. Please check the file path.")
    exit(1)

# Verify column names and initial shape
print("Initial dataset columns:", df.columns.tolist())
print("Initial dataset shape:", df.shape)

# Define key columns
key_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Check for missing columns
missing_cols = [col for col in key_columns if col not in df.columns]
if missing_cols:
    raise KeyError(f"Columns {missing_cols} not found in dataset")

# Preprocess the data
df['heart_disease'] = pd.to_numeric(df['num'], errors='coerce').apply(lambda x: 1 if x >= 1 else 0)
df['heart_disease'] = df['heart_disease'].fillna(0).astype(int)

numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'slope', 'ca']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

df['sex_encoded'] = df['sex'].map({'Male': 1, 'Female': 0}).fillna(df['sex'].mode()[0]).astype(int)
df['fbs_binary'] = df['fbs'].map({'TRUE': 1, 'FALSE': 0}).fillna(df['fbs'].mode()[0]).astype(int)
df['exang'] = df['exang'].map({'TRUE': 1, 'FALSE': 0}).fillna(df['exang'].mode()[0]).astype(int)
df['cp_encoded'] = LabelEncoder().fit_transform(df['cp'].fillna(df['cp'].mode()[0]))
df['restecg_encoded'] = LabelEncoder().fit_transform(df['restecg'].fillna(df['restecg'].mode()[0]))
df['thal_encoded'] = LabelEncoder().fit_transform(df['thal'].fillna(df['thal'].mode()[0]))

print("Missing values per column before dropna:")
print(df[key_columns].isna().sum())

df = df.dropna(subset=['age', 'heart_disease', 'trestbps', 'chol', 'thalch'])

if df.empty:
    print("Error: No data remaining after dropping NaN values. Check the dataset.")
    exit(1)

df['age_group'] = pd.cut(df['age'], bins=[0, 45, 55, 65, 100], labels=['Young', 'Middle', 'Older', 'Elderly'])
df['old'] = (df['age'] > 55).astype(int)
df['high_chol'] = (df['chol'] > 240).astype(int)
df['high_bps'] = (df['trestbps'] > 140).astype(int)

df['old_male'] = (df['old'] == 1) & (df['sex_encoded'] == 1)
df['young_female'] = (df['age'] <= 45) & (df['sex_encoded'] == 0)
df['middle_male'] = ((df['age'] > 45) & (df['age'] <= 55) & (df['sex_encoded'] == 1))
df['elderly_female'] = (df['age'] > 65) & (df['sex_encoded'] == 0)

# Feature combinations
feature_combinations = [
    ('old', 'high_chol'),
    ('old', 'high_bps'),
    ('high_chol', 'high_bps')
]
for sex in [0, 1]:
    sex_label = 'female' if sex == 0 else 'male'
    for feat1, feat2 in feature_combinations:
        combo_name = f"{sex_label}_{feat1}_{feat2}"
        df[combo_name] = (df[feat1] == 1) & (df[feat2] == 1) & (df['sex_encoded'] == sex)

print("🫀 HEART DISEASE POPULATION SUBGROUP ANALYSIS")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Heart Disease Distribution: {df['heart_disease'].value_counts().to_dict()}")
print(f"Heart Disease Rate: {df['heart_disease'].mean():.2%}")

print("\n👥 POPULATION SUBGROUP DISTRIBUTIONS")
print("=" * 45)
subgroups = ['old_male', 'young_female', 'middle_male', 'elderly_female']
for group in subgroups:
    count = df[group].sum()
    hd_rate = df[df[group]]['heart_disease'].mean() if count > 0 else 0
    print(f"{group}: {count} people, HD rate: {hd_rate:.2%}")

print("\n👥 FEATURE COMBINATION DISTRIBUTIONS")
print("=" * 45)
for col in [c for c in df.columns if c.startswith(('male_', 'female_'))]:
    count = df[col].sum()
    hd_rate = df[df[col]]['heart_disease'].mean() if count > 0 else 0
    print(f"{col}: {count} people, HD rate: {hd_rate:.2%}")

# Select dependent variables with non-zero variance
dependent_vars = ['trestbps', 'chol', 'thalch']  # Removed exang due to zero variance
control_vars = ['age', 'sex_encoded', 'fbs_binary', 'cp_encoded']

if df[dependent_vars].isna().any().any():
    print("Warning: NaNs found in dependent variables. Imputing with median.")
    for col in dependent_vars:
        df[col] = df[col].fillna(df[col].median())

X_dependent = df[dependent_vars].copy()
X_control = df[control_vars].copy()

# Check variance to diagnose singularity
print("\nVariance of dependent variables:")
print(X_dependent.var())

scaler_dep = StandardScaler()
scaler_ctrl = StandardScaler()
X_dependent_scaled = scaler_dep.fit_transform(X_dependent)
X_control_scaled = scaler_ctrl.fit_transform(X_control)

if np.any(np.isnan(X_dependent_scaled)):
    print("Error: NaNs in standardized dependent variables. Skipping statistical tests.")
else:
    print("\n🧪 MULTIVARIATE TESTS FOR HEART DISEASE CLASSIFICATION")
    print("=" * 65)

    print("\n1️⃣ OVERALL MANOVA: Heart Disease vs No Heart Disease")
    print("-" * 55)
    manova_data = pd.DataFrame(X_dependent_scaled, columns=dependent_vars)
    manova_data['heart_disease'] = df['heart_disease'].values
    dependent_formula = ' + '.join(dependent_vars)
    manova_formula = f"{dependent_formula} ~ C(heart_disease)"

    try:
        manova_result = MANOVA.from_formula(manova_formula, data=manova_data)
        manova_stats = manova_result.mv_test()
        print("MANOVA Results:")
        print(manova_stats)
        stats_df = manova_stats.results['C(heart_disease)']['stat']
        wilks_lambda = stats_df.iloc[0, 0]
        f_stat = stats_df.iloc[0, 1]
        p_value = stats_df.iloc[0, 3]
        print(f"\n📊 Overall Heart Disease Effect:")
        print(f"   Wilks' Lambda: {wilks_lambda:.4f}")
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("   ✅ SIGNIFICANT")
        else:
            print("   ❌ NOT SIGNIFICANT")
    except Exception as e:
        print(f"MANOVA Error: {e}")

    print("\n2️⃣ MANCOVA: Controlling for Demographics")
    print("-" * 45)
    mancova_data = pd.DataFrame(X_dependent_scaled, columns=dependent_vars)
    mancova_data['heart_disease'] = df['heart_disease'].values
    for i, var in enumerate(control_vars):
        mancova_data[var] = X_control_scaled[:, i]
    covariate_formula = ' + '.join(control_vars)
    mancova_formula = f"{dependent_formula} ~ C(heart_disease) + {covariate_formula}"

    try:
        mancova_result = MANOVA.from_formula(mancova_formula, data=mancova_data)
        mancova_stats = mancova_result.mv_test()
        print("MANCOVA Results:")
        print(mancova_stats)
        if 'C(heart_disease)' in mancova_stats.results:
            stats_df = mancova_stats.results['C(heart_disease)']['stat']
            wilks_cov = stats_df.iloc[0, 0]
            f_stat_cov = stats_df.iloc[0, 1]
            p_value_cov = stats_df.iloc[0, 3]
            print(f"\n📊 Heart Disease Effect (Controlled):")
            print(f"   Wilks' Lambda: {wilks_cov:.4f}")
            print(f"   F-statistic: {f_stat_cov:.4f}")
            print(f"   P-value: {p_value_cov:.4f}")
            if p_value_cov < 0.05:
                print("   ✅ SIGNIFICANT")
            else:
                print("   ❌ NOT SIGNIFICANT")
    except Exception as e:
        print(f"MANCOVA Error: {e}")

    print("\n3️⃣ POPULATION SUBGROUP MULTIVARIATE TESTS")
    print("-" * 50)
    subgroup_results = {}
    for subgroup in subgroups:
        print(f"\n🎯 Testing {subgroup.upper().replace('_', ' ')} subgroup:")
        print("-" * 40)
        subgroup_mask = df[subgroup]
        if subgroup_mask.sum() < 30:
            print(f"   ⚠️  Too few samples ({subgroup_mask.sum()}) for reliable analysis")
            continue
        subgroup_data = df[subgroup_mask].copy()
        hd_counts = subgroup_data['heart_disease'].value_counts()
        if len(hd_counts) < 2 or min(hd_counts) < 10:
            print(f"   ⚠️  Insufficient samples in both HD classes")
            continue
        X_sub_dep = subgroup_data[dependent_vars].copy()
        X_sub_dep_scaled = scaler_dep.transform(X_sub_dep)
        sub_manova_data = pd.DataFrame(X_sub_dep_scaled, columns=dependent_vars)
        sub_manova_data['heart_disease'] = subgroup_data['heart_disease'].values
        try:
            sub_manova = MANOVA.from_formula(manova_formula, data=sub_manova_data)
            sub_stats = sub_manova.mv_test()
            stats_df = sub_stats.results['C(heart_disease)']['stat']
            wilks_sub = stats_df.iloc[0, 0]
            f_stat_sub = stats_df.iloc[0, 1]
            p_value_sub = stats_df.iloc[0, 3]
            print(f"   Sample size: {len(subgroup_data)}")
            print(f"   HD rate: {subgroup_data['heart_disease'].mean():.2%}")
            print(f"   Wilks' Lambda: {wilks_sub:.4f}")
            print(f"   F-statistic: {f_stat_sub:.4f}")
            print(f"   P-value: {p_value_sub:.4f}")
            if p_value_sub < 0.05:
                print("   ✅ SIGNIFICANT")
                significance = "SIGNIFICANT"
            else:
                print("   ❌ NOT SIGNIFICANT")
                significance = "NOT SIGNIFICANT"
            subgroup_results[subgroup] = {
                'n': len(subgroup_data),
                'hd_rate': subgroup_data['heart_disease'].mean(),
                'wilks_lambda': wilks_sub,
                'f_stat': f_stat_sub,
                'p_value': p_value_sub,
                'significance': significance
            }
        except Exception as e:
            print(f"   ❌ Error in subgroup analysis: {e}")

    print("\n4️⃣ PERMANOVA: Non-parametric Validation")
    print("-" * 45)
    try:
        perm_f_stat, perm_p_value, perm_dist = permanova(X_dependent_scaled, df['heart_disease'].values, n_permutations=99)
        print(f"Overall PERMANOVA Results:")
        print(f"   F-statistic: {perm_f_stat:.4f}")
        print(f"   P-value: {perm_p_value:.4f}")
        if perm_p_value < 0.05:
            print("   ✅ SIGNIFICANT")
        else:
            print("   ❌ NOT SIGNIFICANT")
    except KeyboardInterrupt:
        print("PERMANOVA interrupted by user. Reducing permutations may help.")
    except Exception as e:
        print(f"PERMANOVA Error: {e}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Heart Disease Population Subgroup Analysis', fontsize=16)

axes[0, 0].bar(['No HD', 'Has HD'], df['heart_disease'].value_counts().sort_index(),
               color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Heart Disease Distribution')
axes[0, 0].set_ylabel('Count')

pivot_table = df.pivot_table(values='heart_disease', index='age_group', columns='sex', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='Reds', ax=axes[0, 1])
axes[0, 1].set_title('Heart Disease Rate by Age Group and Sex')

subgroup_hd_rates = [df[df[sg]]['heart_disease'].mean() if df[sg].sum() > 0 else 0 for sg in subgroups]
axes[1, 0].bar(range(len(subgroups)), subgroup_hd_rates,
               color=['red', 'blue', 'green', 'purple'], alpha=0.7, edgecolor='black')
axes[1, 0].set_xticks(range(len(subgroups)))
axes[1, 0].set_xticklabels([sg.replace('_', '\n') for sg in subgroups], rotation=45)
axes[1, 0].set_title('Heart Disease Rates by Subgroup')
axes[1, 0].set_ylabel('HD Rate')

if subgroup_results:
    subgroup_names = list(subgroup_results.keys())
    p_values = [subgroup_results[sg]['p_value'] for sg in subgroup_names]
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    axes[1, 1].bar(range(len(subgroup_names)), p_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    axes[1, 1].set_xticks(range(len(subgroup_names)))
    axes[1, 1].set_xticklabels([sg.replace('_', '\n') for sg in subgroup_names], rotation=45)
    axes[1, 1].set_title('Subgroup MANOVA P-values')
    axes[1, 1].set_ylabel('P-value')
    axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n🎯 COMPREHENSIVE RESULTS SUMMARY")
print("=" * 50)
print("\n📊 POPULATION SUBGROUP FINDINGS:")
for subgroup, results in subgroup_results.items():
    print(f"\n🔍 {subgroup.upper().replace('_', ' ')}:")
    print(f"   • Sample size: {results['n']}")
    print(f"   • Heart disease rate: {results['hd_rate']:.1%}")
    print(f"   • Multivariate F-statistic: {results['f_stat']:.3f}")
    print(f"   • P-value: {results['p_value']:.3f}")
    print(f"   • Statistical significance: {results['significance']}")

print("\n📊 FEATURE COMBINATION FINDINGS:")
for col in [c for c in df.columns if c.startswith(('male_', 'female_'))]:
    count = df[col].sum()
    if count > 0:
        hd_rate = df[df[col]]['heart_disease'].mean()
        print(f"\n🔍 {col.upper().replace('_', ' ')}:")
        print(f"   • Sample size: {count}")
        print(f"   • Heart disease rate: {hd_rate:.1%}")