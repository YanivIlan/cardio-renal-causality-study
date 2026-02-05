import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import optuna
import pickle
import sklearn

# Create output directory
output_dir = 'regression_ckd'
os.makedirs(output_dir, exist_ok=True)

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

# === Encoding categorical variables ===
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

# === Create categorical version of Age ===
bins = [0, 30, 50, float('inf')]
labels = [0, 1, 2]  # 0: young, 1: middle-aged, 2: older
df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True).astype(int)
print(f"Created Age_Category with bins {bins} and labels {labels}")

# === Correlation Analysis ===
corr_columns = ['Age_Category', 'Blood Pressure', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine',
                'Sodium', 'Potassium', 'Hemoglobin', 'Packed Cell Volume', 'White Blood Cell Count',
                'Red Blood Cell Count', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell', 'Pus Cell clumps',
                'Bacteria', 'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 'Appetite',
                'Pedal Edema', 'Anemia', 'Class']

pearson_corr = df[corr_columns].corr(method='pearson')
spearman_corr = df[corr_columns].corr(method='spearman')

def compute_p_values(df, columns, method='pearson'):
    p_values = pd.DataFrame(index=columns, columns=columns)
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i <= j:
                valid_rows = df[[col1, col2]].dropna()
                if len(valid_rows) < 2:
                    p_values.loc[col1, col2] = np.nan
                    p_values.loc[col2, col1] = np.nan
                    continue
                x, y = valid_rows[col1], valid_rows[col2]
                try:
                    if method == 'pearson':
                        corr, p_val = pearsonr(x, y)
                    else:
                        corr, p_val = spearmanr(x, y)
                    p_values.loc[col1, col2] = p_val
                    p_values.loc[col2, col1] = p_val
                except ValueError:
                    p_values.loc[col1, col2] = np.nan
                    p_values.loc[col2, col1] = np.nan
    return p_values

pearson_p_values = compute_p_values(df, corr_columns, method='pearson')
spearman_p_values = compute_p_values(df, corr_columns, method='spearman')

categorical_features = ['Age_Category', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell',
                        'Pus Cell clumps', 'Bacteria', 'Hypertension', 'Diabetes Mellitus',
                        'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia', 'Class']

print("\nPearson Correlation Matrix:")
print(pearson_corr.round(3))
print("\nSpearman Correlation Matrix:")
print(spearman_corr.round(3))

print("\nPearson Correlation Significance (p-values, significant if p < 0.05):")
for col1 in corr_columns:
    for col2 in corr_columns:
        if col1 < col2:
            p_val = pearson_p_values.loc[col1, col2]
            is_categorical_pair = col1 in categorical_features or col2 in categorical_features
            label = " (categorical pair)" if is_categorical_pair else ""
            if pd.isna(p_val):
                print(f"Pearson: {col1} vs {col2}: correlation = {pearson_corr.loc[col1, col2]:.3f}, "
                      f"p-value = NaN (insufficient data){label}")
                continue
            corr_val = pearson_corr.loc[col1, col2]
            significance = "significant" if p_val < 0.05 else "not significant"
            print(f"Pearson: {col1} vs {col2}: correlation = {corr_val:.3f}, p-value = {p_val:.3f} "
                  f"({significance}){label}")

print("\nSpearman Correlation Significance (p-values, significant if p < 0.05):")
for col1 in corr_columns:
    for col2 in corr_columns:
        if col1 < col2:
            p_val = spearman_p_values.loc[col1, col2]
            is_categorical_pair = col1 in categorical_features or col2 in categorical_features
            label = " (categorical pair)" if is_categorical_pair else ""
            if pd.isna(p_val):
                print(f"Spearman: {col1} vs {col2}: correlation = {spearman_corr.loc[col1, col2]:.3f}, "
                      f"p-value = NaN (insufficient data){label}")
                continue
            corr_val = spearman_corr.loc[col1, col2]
            significance = "significant" if p_val < 0.05 else "not significant"
            print(f"Spearman: {col1} vs {col2}: correlation = {corr_val:.3f}, p-value = {p_val:.3f} "
                  f"({significance}){label}")

# === Visualization ===
plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
plt.title("Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pearson_corr_heatmap_ckd.png'))
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spearman_corr_heatmap_ckd.png'))
plt.close()

# === Model Training ===
# Select features for modeling
model_features = ['Hemoglobin', 'Packed Cell Volume', 'Red Blood Cell Count', 'Hypertension', 'Diabetes Mellitus']
X = df[model_features]
y = df['Class']

# Check for NaN values in selected features and target
print("\nChecking for NaN values in selected features and target:")
print(X.isnull().sum())
print("NaN values in target (Class):", y.isnull().sum())

# Impute any remaining NaNs in selected features
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=model_features, index=X.index)
y = y.fillna(y.mode()[0])  # Impute target with mode if necessary

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check for NaN values after scaling
print("\nChecking for NaN values after scaling:")
print("X_train_scaled NaN count:", np.isnan(X_train_scaled).sum())
print("X_test_scaled NaN count:", np.isnan(X_test_scaled).sum())

# Define objective functions for Optuna with cross-validation using Pipeline
def objective_logreg(trial):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000))
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    return np.mean(scores)

def objective_knn(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 9)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    return np.mean(scores)

def objective_svm(trial):
    C = trial.suggest_float('C', 0.1, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVC(C=C, kernel=kernel, gamma=gamma, probability=True))
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    return np.mean(scores)

def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, random_state=42))
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    return np.mean(scores)

def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, subsample=subsample,
                          colsample_bytree=colsample_bytree, random_state=42, eval_metric='logloss')
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    return np.mean(scores)

def objective_lgbm(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    return np.mean(scores)

# Define models and their Optuna objective functions
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), objective_logreg),
    'KNN': (KNeighborsClassifier(), objective_knn),
    'SVM': (SVC(probability=True), objective_svm),
    'Random Forest': (RandomForestClassifier(random_state=42), objective_rf),
    'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), objective_xgb),
    'LightGBM': (LGBMClassifier(random_state=42), objective_lgbm)
}

# Store results
results = {}
roc_data = {}

# Train and evaluate models
for name, (model, objective) in models.items():
    print(f"\nTraining {name}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Train model with best parameters
    if name == 'Logistic Regression':
        best_model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=study.best_params['C'],
                penalty=study.best_params['penalty'],
                solver='liblinear',
                max_iter=1000
            ))
        ])
    elif name == 'KNN':
        best_model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(
                n_neighbors=study.best_params['n_neighbors'],
                weights=study.best_params['weights']
            ))
        ])
    elif name == 'SVM':
        best_model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', SVC(
                C=study.best_params['C'],
                kernel=study.best_params['kernel'],
                gamma=study.best_params['gamma'],
                probability=True
            ))
        ])
    elif name == 'Random Forest':
        best_model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=study.best_params['n_estimators'],
                max_depth=study.best_params['max_depth'],
                min_samples_split=study.best_params['min_samples_split'],
                random_state=42
            ))
        ])
    elif name == 'XGBoost':
        best_model = XGBClassifier(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            learning_rate=study.best_params['learning_rate'],
            subsample=study.best_params['subsample'],
            colsample_bytree=study.best_params['colsample_bytree'],
            random_state=42,
            eval_metric='logloss'
        )
    elif name == 'LightGBM':
        best_model = LGBMClassifier(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            learning_rate=study.best_params['learning_rate'],
            random_state=42
        )

    # Fit model
    if name == 'XGBoost':
        best_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    elif name == 'LightGBM':
        best_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
                       callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
    else:
        best_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Metrics
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)

    # ROC curve
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

    # Store results
    results[name] = {
        'model': best_model,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': test_f1,
        'train_f1': train_f1,
        'best_params': study.best_params
    }

    # Print results
    print(f"Best parameters: {study.best_params}")
    print(f"Training F1-score: {train_f1:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")

# Find best model based on test F1-score
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
print(f"\nBest model based on F1-score: {best_model_name} with Test F1-score: {results[best_model_name]['f1']:.4f}")

# Print and save best model metrics
best_metrics = results[best_model_name]
metrics_output = (
    f"Best Model: {best_model_name}\n"
    f"Best parameters: {best_metrics['best_params']}\n"
    f"Training F1-score: {best_metrics['train_f1']:.4f}\n"
    f"Test F1-score: {best_metrics['f1']:.4f}\n"
    f"Confusion Matrix:\n{best_metrics['confusion_matrix']}\n"
    f"Accuracy: {best_metrics['accuracy']:.4f}\n"
    f"Precision: {best_metrics['precision']:.4f}\n"
    f"Recall: {best_metrics['recall']:.4f}\n"
    f"F1-score: {best_metrics['f1']:.4f}"
)
print("\n=== Best Model Metrics ===")
print(metrics_output)
with open(os.path.join(output_dir, 'best_model_metrics_ckd.txt'), 'w') as f:
    f.write(metrics_output)

# Generate learning curve for the best model
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Calculate mean and std for training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 8))
plt.plot(train_sizes, train_scores_mean, label='Training F1-score')
plt.plot(train_sizes, val_scores_mean, label='Validation F1-score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1)
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('F1-score')
plt.title(f'Learning Curve for {best_model_name}')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'learning_curve_ckd.png'))
plt.close()

# Feature importance for tree-based models
if best_model_name in ['Random Forest']:
    feature_importance = best_model.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({'Feature': model_features, 'Importance': feature_importance})
    print("\nFeature Importance for Best Model:")
    print(importance_df.sort_values(by='Importance', ascending=False))
elif best_model_name in ['XGBoost', 'LightGBM']:
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': model_features, 'Importance': feature_importance})
    print("\nFeature Importance for Best Model:")
    print(importance_df.sort_values(by='Importance', ascending=False))

# Save best model and scaler
with open(os.path.join(output_dir, 'best_model_ckd.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(output_dir, 'scaler_ckd.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# === Test Loading and Predicting with Saved Model ===
print("\nTesting loading of saved model and scaler...")
try:
    with open(os.path.join(output_dir, 'best_model_ckd.pkl'), 'rb') as f:
        loaded_model = pickle.load(f)
    with open(os.path.join(output_dir, 'scaler_ckd.pkl'), 'rb') as f:
        loaded_scaler = pickle.load(f)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading .pkl files: {e}")

# Create a sample input for prediction (using median values from X_train for numerical features)
sample_input = np.array([[
    X_train['Hemoglobin'].median(),
    X_train['Packed Cell Volume'].median(),
    X_train['Red Blood Cell Count'].median(),
    X_train['Hypertension'].mode()[0],
    X_train['Diabetes Mellitus'].mode()[0]
]])
print("\nSample input for prediction:")
print(pd.DataFrame(sample_input, columns=model_features))

# === Additional Plots for ROC and Loss Curves ===
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend to avoid tkinter conflicts
from sklearn.metrics import log_loss  # Import log_loss for loss curve

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'roc_curves_ckd.png'))
plt.close()

# Generate learning curves for the best model (F1-score, accuracy, and loss)
# Use n_jobs=1 to avoid threading issues with matplotlib
train_sizes, train_scores_f1, val_scores_f1 = learning_curve(
    best_model, X_train_scaled if best_model_name in ['XGBoost', 'LightGBM'] else X_train,
    y_train, cv=5, scoring='f1', n_jobs=1,  # Single-threaded to avoid tkinter issues
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)
train_sizes, train_scores_acc, val_scores_acc = learning_curve(
    best_model, X_train_scaled if best_model_name in ['XGBoost', 'LightGBM'] else X_train,
    y_train, cv=5, scoring='accuracy', n_jobs=1,  # Single-threaded to avoid tkinter issues
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Initialize loss scores
train_scores_loss = []
val_scores_loss = []

# Compute loss curves for models that support predict_proba or have built-in loss
for train_size in train_sizes:
    train_idx = X_train.index[:int(train_size)]
    X_train_subset = X_train_scaled[:int(train_size)] if best_model_name in ['XGBoost', 'LightGBM', 'Logistic Regression', 'KNN', 'SVM', 'Random Forest'] else X_train.iloc[:int(train_size)]
    y_train_subset = y_train.iloc[:int(train_size)]
    try:
        if best_model_name == 'XGBoost':
            model = XGBClassifier(**best_model.get_params())
            model.fit(X_train_subset, y_train_subset, eval_set=[(X_train_subset, y_train_subset), (X_test_scaled, y_test)], verbose=False)
            eval_results = model.evals_result()
            train_scores_loss.append(eval_results['validation_0']['logloss'][-1])
            val_scores_loss.append(eval_results['validation_1']['logloss'][-1])
        elif best_model_name == 'LightGBM':
            model = LGBMClassifier(**best_model.get_params())
            model.fit(X_train_subset, y_train_subset, eval_set=[(X_train_subset, y_train_subset), (X_test_scaled, y_test)],
                      callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
            eval_results = model.evals_result_
            train_scores_loss.append(eval_results['training']['binary_logloss'][-1])
            val_scores_loss.append(eval_results['valid_1']['binary_logloss'][-1])
        elif best_model_name in ['Logistic Regression', 'KNN', 'SVM', 'Random Forest']:
            model = best_model  # Pipeline already includes scaler
            model.fit(X_train_subset, y_train_subset)
            if hasattr(model, 'predict_proba'):
                train_pred_proba = model.predict_proba(X_train_subset)[:, 1]
                val_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                # Ensure probabilities are valid
                train_pred_proba = np.clip(train_pred_proba, 1e-15, 1 - 1e-15)
                val_pred_proba = np.clip(val_pred_proba, 1e-15, 1 - 1e-15)
                train_loss = log_loss(y_train_subset, train_pred_proba)
                val_loss = log_loss(y_test, val_pred_proba)
                train_scores_loss.append(train_loss)
                val_scores_loss.append(val_loss)
            else:
                train_scores_loss.append(np.nan)
                val_scores_loss.append(np.nan)
        else:
            train_scores_loss.append(np.nan)
            val_scores_loss.append(np.nan)
    except Exception as e:
        print(f"Error computing log loss for train size {train_size}: {e}")
        train_scores_loss.append(np.nan)
        val_scores_loss.append(np.nan)

train_scores_loss = np.array(train_scores_loss)
val_scores_loss = np.array(val_scores_loss)

# Calculate mean and std for training and validation scores
train_scores_f1_mean = np.mean(train_scores_f1, axis=1)
train_scores_f1_std = np.std(train_scores_f1, axis=1)
val_scores_f1_mean = np.mean(val_scores_f1, axis=1)
val_scores_f1_std = np.std(val_scores_f1, axis=1)

train_scores_acc_mean = np.mean(train_scores_acc, axis=1)
train_scores_acc_std = np.std(train_scores_acc, axis=1)
val_scores_acc_mean = np.mean(val_scores_acc, axis=1)
val_scores_acc_std = np.std(val_scores_acc, axis=1)

# Plot learning curves (F1-score, accuracy, and loss)
plt.figure(figsize=(15, 5))

# F1-score plot
plt.subplot(1, 3, 1)
plt.plot(train_sizes, train_scores_f1_mean, label='Training F1-score')
plt.plot(train_sizes, val_scores_f1_mean, label='Validation F1-score')
plt.fill_between(train_sizes, train_scores_f1_mean - train_scores_f1_std,
                 train_scores_f1_mean + train_scores_f1_std, alpha=0.1)
plt.fill_between(train_sizes, val_scores_f1_mean - val_scores_f1_std,
                 val_scores_f1_mean + val_scores_f1_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('F1-score')
plt.title(f'Learning Curve (F1-score) for {best_model_name}')
plt.legend(loc='best')
plt.grid(True)

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(train_sizes, train_scores_acc_mean, label='Training Accuracy')
plt.plot(train_sizes, val_scores_acc_mean, label='Validation Accuracy')
plt.fill_between(train_sizes, train_scores_acc_mean - train_scores_acc_std,
                 train_scores_acc_mean + train_scores_acc_std, alpha=0.1)
plt.fill_between(train_sizes, val_scores_acc_mean - val_scores_acc_std,
                 val_scores_acc_mean + val_scores_acc_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title(f'Learning Curve (Accuracy) for {best_model_name}')
plt.legend(loc='best')
plt.grid(True)

# Loss plot
plt.subplot(1, 3, 3)
if not np.all(np.isnan(train_scores_loss)):
    plt.plot(train_sizes, train_scores_loss, label='Training Log Loss')
    plt.plot(train_sizes, val_scores_loss, label='Validation Log Loss')
    plt.xlabel('Training Set Size')
    plt.ylabel('Log Loss')
    plt.title(f'Learning Curve (Log Loss) for {best_model_name}')
    plt.legend(loc='best')
    plt.grid(True)
else:
    print(f"Warning: Log loss plot skipped for {best_model_name} due to invalid or missing loss values.")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_curves_ckd.png'))
plt.close()

# Predict with loaded model
try:
    if best_model_name in ['XGBoost', 'LightGBM']:
        sample_input_scaled = loaded_scaler.transform(sample_input)
        prediction = loaded_model.predict(sample_input_scaled)
    else:
        prediction = loaded_model.predict(sample_input)
    print(f"Prediction for sample input (0 = notckd, 1 = ckd): {prediction[0]}")
except Exception as e:
    print(f"Error making prediction with loaded model: {e}")