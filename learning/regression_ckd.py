import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# Encode categorical variables
df['Hypertension'] = df['Hypertension'].map({'yes': 1, 'no': 0})
df['Diabetes Mellitus'] = df['Diabetes Mellitus'].map({'yes': 1, 'no': 0})

# === Regression for Packed Cell Volume ===
# Select features and target
reg_features = ['Age', 'Hemoglobin', 'Red Blood Cell Count']
reg_target = 'Packed Cell Volume'
X_reg = df[reg_features]
y_reg = df[reg_target]

# Check for NaN values
print("\nChecking for NaN values in regression features and target:")
print(X_reg.isnull().sum())
print("NaN values in target (Packed Cell Volume):", y_reg.isnull().sum())

# Impute any remaining NaNs
imputer = SimpleImputer(strategy='median')
X_reg = pd.DataFrame(imputer.fit_transform(X_reg), columns=reg_features, index=X_reg.index)
y_reg = y_reg.fillna(y_reg.median())

# Split data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale features
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Check for NaN values after scaling
print("\nChecking for NaN values after scaling (regression):")
print("X_reg_train_scaled NaN count:", np.isnan(X_reg_train_scaled).sum())
print("X_reg_test_scaled NaN count:", np.isnan(X_reg_test_scaled).sum())

# Optimize polynomial degree for non-linear regression
def objective_poly(trial):
    degree = trial.suggest_int('degree', 1, 5)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree)),
        ('model', LinearRegression())
    ])
    scores = cross_val_score(pipeline, X_reg_train, y_reg_train, cv=5, scoring='r2')
    return np.mean(scores)

print("\nOptimizing polynomial degree for non-linear regression...")
study_poly = optuna.create_study(direction='maximize')
study_poly.optimize(objective_poly, n_trials=10)
best_degree = study_poly.best_params['degree']
print(f"Best polynomial degree: {best_degree}")

# Train linear regression
linear_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
linear_pipeline.fit(X_reg_train, y_reg_train)

# Train non-linear regression with best degree
nonlinear_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=best_degree)),
    ('model', LinearRegression())
])
nonlinear_pipeline.fit(X_reg_train, y_reg_train)

# OLS Summary for Linear Regression
X_reg_train_scaled_with_const = sm.add_constant(X_reg_train_scaled)
X_reg_test_scaled_with_const = sm.add_constant(X_reg_test_scaled)
ols_linear = sm.OLS(y_reg_train, X_reg_train_scaled_with_const).fit()
with open(os.path.join(output_dir, 'ols_summary_linear_ckd.txt'), 'w') as f:
    f.write(str(ols_linear.summary()))
print("\nOLS Summary for Linear Regression:")
print(ols_linear.summary())

# OLS Summary for Non-Linear Regression
poly = PolynomialFeatures(degree=best_degree)
X_reg_train_poly = poly.fit_transform(X_reg_train_scaled)
X_reg_test_poly = poly.transform(X_reg_test_scaled)
X_reg_train_poly_with_const = sm.add_constant(X_reg_train_poly)
ols_nonlinear = sm.OLS(y_reg_train, X_reg_train_poly_with_const).fit()
with open(os.path.join(output_dir, 'ols_summary_nonlinear_ckd.txt'), 'w') as f:
    f.write(str(ols_nonlinear.summary()))
print("\nOLS Summary for Non-Linear Regression (degree={}):".format(best_degree))
print(ols_nonlinear.summary())

# Evaluate regression models
def evaluate_regression(model, X_train, X_test, y_train, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = {
        'Model': model_name,
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred),
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred)
    }
    return metrics

# Collect regression results
reg_results = []
reg_results.append(evaluate_regression(linear_pipeline, X_reg_train, X_reg_test, y_reg_train, y_reg_test, 'Linear Regression'))
reg_results.append(evaluate_regression(nonlinear_pipeline, X_reg_train, X_reg_test, y_reg_train, y_reg_test, f'Non-Linear Regression (degree={best_degree})'))

# Print and save regression quality table
reg_results_df = pd.DataFrame(reg_results)
print("\nRegression Quality Table for Packed Cell Volume:")
print(reg_results_df.round(4))
reg_results_df.to_csv(os.path.join(output_dir, 'regression_quality_ckd.csv'), index=False)

# Save regression models
with open(os.path.join(output_dir, 'linear_model_pcv_ckd.pkl'), 'wb') as f:
    pickle.dump(linear_pipeline, f)
with open(os.path.join(output_dir, 'nonlinear_model_pcv_ckd.pkl'), 'wb') as f:
    pickle.dump(nonlinear_pipeline, f)
with open(os.path.join(output_dir, 'scaler_pcv_ckd.pkl'), 'wb') as f:
    pickle.dump(scaler_reg, f)

# === Logistic Regression for Hypertension and Diabetes Mellitus ===
log_targets = [
    ('Hypertension', ['Hemoglobin', 'Age', 'Red Blood Cell Count']),
    ('Diabetes Mellitus', ['Hemoglobin', 'Age', 'Red Blood Cell Count']),
    ('Diabetes Mellitus (No RBC)', ['Hemoglobin', 'Age'])
]

# Store logistic regression results
log_results = {}

for target, features in log_targets:
    print(f"\nTraining Logistic Regression for {target}...")
    X_log = df[features]
    y_log = df[target.split(' (')[0]]  # Handle 'Diabetes Mellitus (No RBC)' by using 'Diabetes Mellitus'

    # Check for NaN values
    print(f"\nChecking for NaN values in {target} features and target:")
    print(X_log.isnull().sum())
    print(f"NaN values in target ({target}):", y_log.isnull().sum())

    # Impute any remaining NaNs
    X_log = pd.DataFrame(imputer.fit_transform(X_log), columns=features, index=X_log.index)
    y_log = y_log.fillna(y_log.mode()[0])

    # Split data with stratification
    X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42, stratify=y_log)

    # Scale features
    scaler_log = StandardScaler()
    X_log_train_scaled = scaler_log.fit_transform(X_log_train)
    X_log_test_scaled = scaler_log.transform(X_log_test)

    # Check for NaN values after scaling
    print(f"\nChecking for NaN values after scaling ({target}):")
    print(f"X_log_train_scaled NaN count:", np.isnan(X_log_train_scaled).sum())
    print(f"X_log_test_scaled NaN count:", np.isnan(X_log_test_scaled).sum())

    # Define objective function for Optuna
    def objective_logreg(trial):
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000))
        ])
        scores = cross_val_score(pipeline, X_log_train, y_log_train, cv=5, scoring='f1')
        return np.mean(scores)

    # Optimize hyperparameters
    study_log = optuna.create_study(direction='maximize')
    study_log.optimize(objective_logreg, n_trials=20)

    # Train model with best parameters
    best_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            C=study_log.best_params['C'],
            penalty=study_log.best_params['penalty'],
            solver='liblinear',
            max_iter=1000
        ))
    ])
    best_model.fit(X_log_train, y_log_train)

    # Logistic Regression Summary with statsmodels
    X_log_train_scaled_with_const = sm.add_constant(X_log_train_scaled)
    logit_model = sm.Logit(y_log_train, X_log_train_scaled_with_const).fit(disp=0)
    summary_filename = f'logistic_summary_{target.lower().replace(" ", "_").replace("_(", "_").replace(")", "")}_ckd.txt'
    with open(os.path.join(output_dir, summary_filename), 'w') as f:
        f.write(str(logit_model.summary()))
    print(f"\nLogistic Regression Summary for {target}:")
    print(logit_model.summary())

    # Predictions
    y_log_train_pred = best_model.predict(X_log_train)
    y_log_test_pred = best_model.predict(X_log_test)

    # Metrics
    train_f1 = f1_score(y_log_train, y_log_train_pred)
    test_f1 = f1_score(y_log_test, y_log_test_pred)
    cm = confusion_matrix(y_log_test, y_log_test_pred)
    accuracy = accuracy_score(y_log_test, y_log_test_pred)
    precision = precision_score(y_log_test, y_log_test_pred)
    recall = recall_score(y_log_test, y_log_test_pred)

    # Store results
    log_results[target] = {
        'model': best_model,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': test_f1,
        'train_f1': train_f1,
        'best_params': study_log.best_params
    }

    # Print results
    print(f"\nResults for {target}:")
    print(f"Best parameters: {study_log.best_params}")
    print(f"Training F1-score: {train_f1:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")

    # Save model
    model_filename = f'logistic_model_{target.lower().replace(" ", "_").replace("_(", "_").replace(")", "")}_ckd.pkl'
    scaler_filename = f'scaler_{target.lower().replace(" ", "_").replace("_(", "_").replace(")", "")}_ckd.pkl'
    with open(os.path.join(output_dir, model_filename), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(output_dir, scaler_filename), 'wb') as f:
        pickle.dump(scaler_log, f)

    # Test loading and predicting with saved model
    print(f"\nTesting loading of saved model and scaler for {target}...")
    try:
        with open(os.path.join(output_dir, model_filename), 'rb') as f:
            loaded_model = pickle.load(f)
        with open(os.path.join(output_dir, scaler_filename), 'rb') as f:
            loaded_scaler = pickle.load(f)
        print(f"Model and scaler for {target} loaded successfully.")
    except Exception as e:
        print(f"Error loading .pkl files for {target}: {e}")

    # Create sample input
    sample_input = np.array([[X_log_train[feat].median() for feat in features]])
    print(f"\nSample input for {target} prediction:")
    print(pd.DataFrame(sample_input, columns=features))

    # Predict with loaded model
    try:
        prediction = loaded_model.predict(sample_input)
        print(f"Prediction for {target} (0 = no, 1 = yes): {prediction[0]}")
    except Exception as e:
        print(f"Error making prediction for {target} with loaded model: {e}")