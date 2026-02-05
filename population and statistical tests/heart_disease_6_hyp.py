import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, chi2
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import optuna

# === Load and prepare data ===
df = pd.read_csv("heart_disease_uci.csv")
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df = df[['age', 'chol', 'trestbps', 'num']].dropna()

# === Correlation matrix + significance ===
print("=== Correlation Matrix and Significance Tests ===")
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            r, p = pearsonr(df[col1], df[col2])
            rho, p_s = spearmanr(df[col1], df[col2])
            print(f"{col1} vs {col2}:")
            print(f"  Pearson r = {r:.4f}, p = {p:.4e}")
            print(f"  Spearman ρ = {rho:.4f}, p = {p_s:.4e}\n")

# === Logistic Regression ===
logit_model = smf.logit('num ~ age + chol + trestbps', data=df).fit()
print("\n=== Logistic Regression Summary ===")
print(logit_model.summary())

# === Predicted probabilities ===
df['pred_prob'] = logit_model.predict()
df['pred_class'] = (df['pred_prob'] > 0.5).astype(int)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(df['num'], df['pred_prob'])
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(df['num'], df['pred_prob']):.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Likelihood Ratio Test ===
null_model = smf.logit('num ~ 1', data=df).fit(disp=False)
lr_stat = 2 * (logit_model.llf - null_model.llf)
p_lr = chi2.sf(lr_stat, df=3)
print("\n=== Likelihood Ratio Test ===")
print(f"LR Statistic = {lr_stat:.4f}, p = {p_lr:.4e}")

# === Bootstrapped Coefficient Stability ===
print("\n=== Bootstrapped Logistic Coefficient Stability ===")
n_boot = 1000
coefs = []
for _ in range(n_boot):
    sample = resample(df)
    model = smf.logit('num ~ age + chol + trestbps', data=sample).fit(disp=False)
    coefs.append(model.params.values)
coef_df = pd.DataFrame(coefs, columns=['Intercept', 'age', 'chol', 'trestbps'])
print(coef_df.describe(percentiles=[.025, .5, .975]))

# === Boxplot of Coefficients ===
sns.boxplot(data=coef_df)
plt.title("Bootstrapped Coefficient Distribution (Logistic Model)")
plt.ylabel("Coefficient Value")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === Residuals and Influence (Fixed) ===
influence = logit_model.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
resid = logit_model.resid_pearson  # valid for GLM/logit models

plt.figure(figsize=(8, 6))
plt.scatter(leverage, resid, s=1000 * cooks_d, alpha=0.5, edgecolor='k')
plt.xlabel("Leverage", fontsize=12)
plt.ylabel("Pearson Residuals", fontsize=12)
plt.title("Influence Plot (Cook's Distance as Bubble Size)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# === Nonlinear Model Definition ===
def nonlinear_model(X_data, a, b, c, d):
    age, chol, trestbps = X_data
    return 1 / (1 + np.exp(-(a * np.log(age + 1) + b * np.sqrt(chol) + c * trestbps**0.3 + d)))

X_raw = df[['age', 'chol', 'trestbps']].values
X_for_fit = (X_raw[:, 0], X_raw[:, 1], X_raw[:, 2])
y_actual = df['num'].values

initial_guess = [0.01, 0.01, 0.01, -1]
params_opt, _ = curve_fit(nonlinear_model, X_for_fit, y_actual, p0=initial_guess)
print("\n=== Nonlinear Logistic Model Parameters ===")
print(f"a = {params_opt[0]:.4f}, b = {params_opt[1]:.4f}, c = {params_opt[2]:.4f}, d = {params_opt[3]:.4f}")

y_pred_nl = nonlinear_model(X_for_fit, *params_opt)
auc_nl = roc_auc_score(y_actual, y_pred_nl)
print(f"AUC (Nonlinear): {auc_nl:.4f}")

# === Compare ROC for linear logistic vs nonlinear ===
plt.figure(figsize=(6, 4))
fpr1, tpr1, _ = roc_curve(y_actual, df['pred_prob'])
fpr2, tpr2, _ = roc_curve(y_actual, y_pred_nl)
plt.plot(fpr1, tpr1, label=f"Logistic AUC = {roc_auc_score(y_actual, df['pred_prob']):.3f}")
plt.plot(fpr2, tpr2, label=f"Nonlinear AUC = {auc_nl:.3f}", linestyle='--')
plt.plot([0, 1], [0, 1], linestyle=':')
plt.title("ROC Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# === Optuna Hyperparameter Tuning ===
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import optuna

def objective(trial):
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = 'liblinear'
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty=penalty, solver=solver))
    score = cross_val_score(clf, df[['age', 'chol', 'trestbps']], df['num'], cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
print("Best AUC:", study.best_value)
