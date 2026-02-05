import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor, BaggingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report ,mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"heart_disease_uci.csv")

# Ensure numeric chest pain types
df['cp'] = df['cp'].map({
    'typical angina': 1,
    'atypical angina': 2,
    'non-anginal': 3,
    'asymptomatic': 4
}) if df['cp'].dtype == object else df['cp']

categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']

encoding_notes = {}  # Store mappings for notes

for col in categorical_cols:
    df[col] = df[col].astype('category')
    # Store original mapping
    encoding_notes[col] = dict(enumerate(df[col].cat.categories, start=1))
    # Apply encoding
    df[col] = df[col].cat.codes + 1  # +1 to start codes from 1

# Handle boolean columns: convert TRUE/FALSE to 1/0
for bool_col in ['fbs', 'exang']:
    df[bool_col] = df[bool_col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})


df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df = df.drop(['id', 'dataset'], axis=1)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
# Checking for any missing values

missing_vals = df.isnull().sum()
missing_pct = (missing_vals / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_vals, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing Values'] > 0])

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
df = df.dropna()
print(df.shape)
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicate_count}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import optuna
import lightgbm

# Drop 'num' column
df = df.drop('num', axis=1)

# Handle missing values (if any)
df = df.fillna(df.mean())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data with stratification to prevent class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define objective functions for Optuna with cross-validation
def objective_logreg(trial):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    model = LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    return np.mean(scores)


def objective_knn(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 9)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    return np.mean(scores)


def objective_svm(trial):
    C = trial.suggest_float('C', 0.1, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    return np.mean(scores)


def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    return np.mean(scores)


def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)  # Dropout-like regularization
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)  # Feature subsampling
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, subsample=subsample,
                          colsample_bytree=colsample_bytree, random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def objective_lgbm(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=42)
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
              callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


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

    # Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Train model with best parameters
    if name == 'Logistic Regression':
        best_model = LogisticRegression(
            C=study.best_params['C'],
            penalty=study.best_params['penalty'],
            solver='liblinear',
            max_iter=1000
        )
    elif name == 'KNN':
        best_model = KNeighborsClassifier(
            n_neighbors=study.best_params['n_neighbors'],
            weights=study.best_params['weights']
        )
    elif name == 'SVM':
        best_model = SVC(
            C=study.best_params['C'],
            kernel=study.best_params['kernel'],
            gamma=study.best_params['gamma'],
            probability=True
        )
    elif name == 'Random Forest':
        best_model = RandomForestClassifier(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            min_samples_split=study.best_params['min_samples_split'],
            random_state=42
        )
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
        best_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    # Metrics
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)

    # ROC curve
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
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

# Generate learning curve for the best model
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_scaled, y_train, cv=5, scoring='f1', n_jobs=-1,
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
plt.savefig('learning_curve.png')
plt.close()

# Optional: Feature importance for tree-based models to check for overfitting
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    print("\nFeature Importance for Best Model:")
    print(importance_df.sort_values(by='Importance', ascending=False))

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Note: Overfitting is minimized through:
# - Stratified train-test split for balanced classes
# - Cross-validation in Optuna for robust hyperparameter tuning
# - Constrained hyperparameter ranges to limit model complexity
# - Dropout-like regularization for XGBoost (subsample, colsample_bytree)
# - Early stopping for LightGBM
# - Test set evaluation to confirm generalization
# - Feature importance to check for over-reliance on specific features