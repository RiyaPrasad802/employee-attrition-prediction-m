# Employee Attrition Prediction Using Machine Learning

## Project Overview

Employee attrition occurs when employees leave an organization. Understanding factors associated with attrition can help organizations study workforce trends and employee retention.

This project uses Machine Learning to predict employee attrition based on employee and job-related information. It includes data preprocessing, exploratory data analysis, model training and evaluation, and an interactive Streamlit application.

**Project type:** Classification
**Machine Learning algorithm:** Logistic Regression
**Dataset:** IBM HR Analytics Employee Attrition
**Application:** Streamlit

## Objectives

* Analyze employee attrition patterns.
* Prepare data for Machine Learning.
* Train and evaluate classification models.
* Estimate attrition probability for user-provided employee details.
* Build an interactive application for demonstrating predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Streamlit
* Joblib
* Git and GitHub

## Project Structure

```text
employee-attrition-prediction-ml/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── raw/
├── models/
│   └── employee_attrition_logistic_regression.joblib
├── notebooks/
│   └── 01_Employee_Attrition_Analysis.ipynb
├── reports/
│   └── figures/
│       └── logistic_regression_coefficients.png
├── sql/
└── src/
```

## Workflow

1. **Data loading:** Load the IBM HR Analytics employee attrition dataset.
2. **Data inspection:** Check dataset dimensions, missing values, duplicates, and target distribution.
3. **Data preprocessing:** Separate features from the target and remove identifier or constant columns.
4. **Exploratory data analysis:** Examine attrition patterns and visualize relevant features.
5. **Model training:** Train Logistic Regression and Random Forest classifiers.
6. **Model evaluation:** Compare classification metrics and use cross-validation and hyperparameter tuning.
7. **Model saving:** Save the selected Logistic Regression pipeline using Joblib.
8. **Application development:** Build a Streamlit interface that accepts employee details and displays predictions.

## Model Evaluation

The selected Logistic Regression model was evaluated on a held-out test set.

| Metric    | Test Result |
| --------- | ----------: |
| Accuracy  |      86.05% |
| Precision |      61.54% |
| Recall    |      34.04% |
| F1-score  |      43.84% |
| ROC-AUC   |      81.15% |

### Understanding the Results

* **Accuracy:** The proportion of all test predictions that were correct.
* **Precision:** Among employees predicted to leave, the proportion who actually left.
* **Recall:** Among employees who actually left, the proportion correctly identified by the model.
* **F1-score:** The harmonic mean of precision and recall.
* **ROC-AUC:** Measures how well the model distinguishes between the two classes across different classification thresholds.

The model achieved 86.05% accuracy and a ROC-AUC of 81.15%. However, its recall was 34.04%, meaning it identified only a portion of employees who left in the test set. Accuracy alone is therefore not sufficient to assess its usefulness.

## Running the Application Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd employee-attrition-prediction-ml
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the Streamlit application

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## Application Features

* Interactive employee information form.
* Machine Learning-based attrition prediction.
* Estimated attrition probability.
* Visual probability progress bar.
* Clear prediction messages.
* Project information and model disclaimer.

## Limitations and Responsible Use

* The model is trained on the available IBM HR Analytics dataset; its performance may not generalize to other organizations.
* The recall score indicates that many employees who left were not identified by the selected model at its current classification threshold.
* Predicted probabilities are estimates, not guarantees.
* Model associations do not establish causation.
* Employee-related predictions can affect people and should not be used as the sole basis for hiring, firing, promotion, or other employment decisions.

## Future Improvements

* Investigate strategies to improve recall while evaluating precision and other metrics.
* Tune the classification threshold using validation data.
* Add more detailed model evaluation visualizations.
* Improve application testing and deployment.
* Evaluate generalization using suitable external data.

## Author

Riya Prasad

B.Tech in Computer Science and Engineering | 2026 Graduate

This project was developed for learning and demonstrating practical Machine Learning skills.
