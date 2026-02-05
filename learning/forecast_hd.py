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
sns.pairplot(df, hue = 'target')
plt.show()
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in numerical_cols:
    print(f"\nValue counts for {col}:\n{df[col].value_counts()}")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30, color='mediumseagreen')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.violinplot(x=df[col], color='orchid')
    plt.title(f'Violin Plot of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='RdBu_r', square=True)
plt.title("Correlation Matrix of Numerical Features")
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()
print(df[numerical_cols].skew())
for col in categorical_cols:
    print(f"\nValue counts for {col}:\n{df[col].value_counts()}")
plt.figure(figsize=(18, 12))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=col, data=df, palette='viridis')
    plt.title(f'Count Plot of {col}')
plt.tight_layout()
plt.show()
print(df[numerical_cols + ['target']].corr()['target'].sort_values(ascending=False))
target_counts = df['target'].value_counts().sort_index()
print(target_counts)
target_percent = (target_counts / target_counts.sum()) * 100
print(target_percent)
plt.figure(figsize=(6,4))
ax = sns.countplot(x='target', data=df, palette='pastel')

for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9)

plt.title('Heart Disease Classification')
plt.xlabel('0 = No Disease, Others = Disease')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(6,4))
ax = sns.countplot(x='target', data=df, palette='pastel')

for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9)

plt.title('Binary Classification of Heart Disease')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Count')
plt.show()

# Print counts separately
print("Target Class Counts:\n", df['target'].value_counts())
print("\nTarget Class Percentages:\n", df['target'].value_counts(normalize=True) * 100)
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    sns.violinplot(data=df, x='target', y=col, palette='pastel')
    plt.title(f'{col} by Heart Disease Presence')
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    ct = pd.crosstab(df[col], df['target'], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=axes[i], colormap='Set2', edgecolor='black')
    axes[i].set_title(f'Target Distribution by {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Proportion')
    axes[i].legend(title='Heart Disease', loc='best')

for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Categorical Features vs Heart Disease (Target)', fontsize=16, y=1.02)
plt.show()
from scipy.stats import ttest_ind

print("T-tests for numerical columns:\n")
for col in numerical_cols:
    group0 = df[df['target'] == 0][col]
    group1 = df[df['target'] == 1][col]
    stat, p = ttest_ind(group0, group1, nan_policy='omit')
    print(f"{col}: p-value = {p:.4f} {'(Significant)' if p < 0.05 else '(Not significant)'}")

from scipy.stats import chi2_contingency
print("Chi-square test for categorical variables:\n")
for col in categorical_cols:
    table = pd.crosstab(df[col], df['target'])
    stat, p, _, _ = chi2_contingency(table)
    print(f"{col}: p-value = {p:.4f} {'(Significant)' if p < 0.05 else '(Not significant)'}")

plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols + ['target']].corr(), annot=True, fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.show()

print(df['age'].describe())

sns.histplot(df['age'], bins=20, kde=True)

plt.axvline(df['age'].mean(), color='red', linestyle='--', label=f"Mean: {df['age'].mean():.2f}")
plt.axvline(df['age'].median(), color='green', linestyle='--', label=f"Median: {df['age'].median():.2f}")
plt.axvline(df['age'].mode()[0], color='blue', linestyle='--', label=f"Mode: {df['age'].mode()[0]}")

plt.legend()
plt.title("Distribution of Age with Mean, Median, and Mode")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

fig = px.histogram(data_frame=df, x='age')
fig.show()

print ("Mean of the dataset: ",df['age'].mean())
print ("Median of the dataset: ",df['age'].median())
print ("Mode of the dataset: ",df['age'].agg(pd.Series.mode))

sns.boxplot(x='target', y='age', data=df, palette='Set2')
plt.title('Age vs Heart Disease')
plt.show()

from scipy.stats import ttest_ind
print(ttest_ind(df[df['target']==0]['age'], df[df['target']==1]['age'], nan_policy='omit'))

sns.scatterplot(x='age', y='thalch', hue='target', data=df, palette='coolwarm')
plt.title("Age vs Max Heart Rate by Target")
plt.show()

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()

print(df['sex'].value_counts())

sns.countplot(df,x='sex')
plt.show()

male_count = 206
female_count = 97

total_count = male_count + female_count

# calculate percentages
male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

# display the results
print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')

# Difference
difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')

print(df.groupby('sex')['age'].value_counts())

# Exploring CP (Chest Pain) column

print(df['cp'].value_counts())
sns.countplot(df,x='cp')
plt.show()

sns.countplot(df, x='cp', hue= 'sex')
plt.show()

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

print(df['trestbps'].describe())

sns.histplot(data=df, x='trestbps', kde=True,)

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')
plt.show()

sns.histplot(df, x='trestbps', kde=True, hue ='sex')
plt.show()

# Impute missing values using iterative imputer for selected columns.

imputer = SimpleImputer(strategy='most_frequent')
df['ca'] = imputer.fit_transform(df[['ca']]).ravel()
df['thal'] = imputer.fit_transform(df[['thal']]).ravel()

# let's check again for missing values
print((df.isnull().sum()).sort_values(ascending=False))

print(df.info())

print(df.columns)

print(df.head())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import optuna

# Drop 'num' column
df = df.drop('num', axis=1)

# Handle missing values (if any)
df = df.fillna(df.mean())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define objective functions for Optuna
def objective_logreg(trial):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    model = LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def objective_knn(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 9)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def objective_svm(trial):
    C = trial.suggest_float('C', 0.1, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
              early_stopping_rounds=10, verbose=False)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def objective_lgbm(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=42)
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
              early_stopping_rounds=10, verbose=False)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


# Define models and their Optuna objective functions
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), objective_logreg),
    'KNN': (KNeighborsClassifier(), objective_knn),
    'SVM': (SVC(probability=True), objective_svm),
    'Random Ascending'
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
    if name in ['XGBoost', 'LightGBM']:
        best_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
                       early_stopping_rounds=10, verbose=False)
    else:
        best_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = best_model.predict(X_test_scaled)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

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
        'f1': f1,
        'best_params': study.best_params
    }

    # Print results
    print(f"Best parameters: {study.best_params}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curves.png')
plt.close()

# Find best model based on F1-score
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
print(f"\nBest model based on F1-score: {best_model_name} with F1-score: {results[best_model_name]['f1']:.4f}")

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


