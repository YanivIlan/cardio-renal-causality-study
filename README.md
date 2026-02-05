# Statistical-Theory

**Final Project in Statistical Theory**  
Authors: *Itamar & Yaniv*  
GitHub: [https://github.com/Itamarzer/Statistical-Theory](https://github.com/Itamarzer/Statistical-Theory)

---

##  Project Summary

This study explores the **relationship and potential causality** between **heart disease** and **chronic kidney disease (CKD)**. Our central hypothesis is that **heart disease may lead to the development of CKD**, but not necessarily the reverse.

### Key Contributions:
- **Population-based analysis** of the two diseases
- **Statistical testing and hypothesis validation**
- **Regression modeling** (linear, nonlinear, and logistic)
- **Machine learning classification and forecasting**
- **Merging datasets** to extract common patterns

### Datasets Used:
- `heart_disease_uci.csv`
- `Chronic_Kidney_Disease.csv`  
(Original source: UCI Machine Learning Repository)

---

##  Environment Setup

This project can be run in any **neutral Python environment**, such as:

- **Google Colab** (Recommended for quick testing)
- **Jupyter Notebook / JupyterLab**
- **VSCode or PyCharm**
- **Command-line Python (3.8+)**

###  1. Install Dependencies

Clone the repository and install required packages:

```bash
git clone https://github.com/Itamarzer/Statistical-Theory.git
cd Statistical-Theory
pip install -r requirements.txt

```
##  2. Download the Datasets
If not already present, download the datasets using Python (in Colab, Jupyter, or any terminal):

```bash
import pandas as pd

# UCI dataset download links (manual step if needed)
!wget -O heart_disease_uci.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
!wget -O Chronic_Kidney_Disease.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00383/Chronic_Kidney_Disease.csv

# Load the data into pandas
heart_df = pd.read_csv('heart_disease_uci.csv')
ckd_df = pd.read_csv('Chronic_Kidney_Disease.csv')
```
Note: Some scripts in the project assume these files exist in the project root. You can also move them using shutil or adjust read_csv() paths in code and also refer to them as only df like in the population and statistical tests folder.


## Running the Code
All .py files are executable directly, including those in subfolders. Example:

```bash
python "regression and models/forecast_ckd.py"
python "regression and models/regression_ckd.py"
python add_ckd_to_hd.py
```

Or from within Jupyter/Colab:

```bash
!python "regression and models/forecast_ckd.py"
```
##  Contact Me

For questions, feedback, or collaboration inquiries, feel free to reach out:

-  Email: itamar.zernitsky@gmail.com  
-  GitHub: [@Itamarzer](https://github.com/Itamarzer)

---
note that there are some parts in the github codes and files that are not mentioned in the article itself like why is the logistic regression is the best model, we also run models like random forests, Xgboost and more. Second there are some EDA files that are not mentioned in the article itself but were presented in the class presentations in short.
