# Fraud Detection in Financial Transactions

## 1. Project Overview

The Fraud Detection in Financial Transactions project uses machine learning techniques to identify potentially fraudulent financial transactions. The main objective is to detect unusual transaction patterns and classify transactions as legitimate or fraudulent.

The project implements machine learning algorithms, including Isolation Forest and Random Forest with SMOTE, to identify suspicious transactions. It also provides an interactive dashboard built using Streamlit to visualize transaction data, monitor fraud alerts, and explore model predictions.

This project demonstrates how machine learning can be applied to financial data analysis to improve fraud detection.

## 2. Project Objectives

* Detect potentially fraudulent financial transactions using machine learning.
* Identify unusual transaction patterns through anomaly detection.
* Handle imbalanced transaction data using SMOTE.
* Compare different machine learning approaches.
* Visualize transaction information and fraud predictions through an interactive dashboard.
* Generate reports and export flagged transactions for further analysis.

## 3. Key Features

* Synthetic financial transaction dataset generation.
* Data preprocessing and feature scaling.
* Unsupervised anomaly detection using Isolation Forest.
* Supervised classification using Random Forest.
* Class imbalance handling using SMOTE.
* Optional AutoEncoder-based anomaly detection.
* Model evaluation using multiple performance metrics.
* Interactive Streamlit dashboard.
* Transaction filtering and fraud alert visualization.
* Export of flagged transactions to CSV files.
* Saving trained machine learning models for reuse.

## 4. Technologies Used

The following technologies and libraries are used in this project:

* **Python:** Main programming language.
* **Pandas:** Data manipulation and analysis.
* **NumPy:** Numerical computations.
* **Scikit-learn:** Machine learning algorithms and evaluation.
* **Imbalanced-learn:** Handling imbalanced datasets using SMOTE.
* **Streamlit:** Building the interactive web dashboard.
* **Plotly:** Interactive data visualization.
* **Joblib:** Saving and loading trained models.
* **Jupyter Notebook:** Exploratory data analysis.
* **TensorFlow:** Optional AutoEncoder implementation.

## 5. Project Structure

```text
fraud-detection-project/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── .gitkeep
│   └── transactions.csv
│
├── models/
│   ├── isolation_forest.joblib
│   ├── rf_smote.joblib
│   └── scaler.joblib
│
├── notebooks/
│   └── EDA.ipynb
│
├── outputs/
│   ├── metrics.json
│   └── predictions.csv
│
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train_isolation_forest.py
│   └── train_autoencoder.py
│
├── requirements.txt
└── README.md
```

**Note:** Some files in the `models/` and `outputs/` directories are generated when the training scripts are executed. Their availability depends on whether the project has already been run.

## 6. Machine Learning Algorithms

### 6.1 Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm used for anomaly detection.

It identifies unusual transactions by isolating observations that differ from normal transaction patterns.

**Purpose:**

* Detect unusual transaction behaviour.
* Identify potential fraud without requiring labelled data for training.
* Generate anomaly scores for transaction analysis.

### 6.2 Random Forest with SMOTE

Random Forest is a supervised machine learning algorithm that combines multiple decision trees to make predictions.

SMOTE stands for Synthetic Minority Over-sampling Technique. It helps address class imbalance by generating synthetic examples of the minority class in the training data.

**Purpose:**

* Classify transactions as legitimate or fraudulent.
* Improve learning from an imbalanced dataset.
* Evaluate fraud classification performance using multiple metrics.

SMOTE should be applied only to the training data to avoid data leakage into the test set.

### 6.3 AutoEncoder

An AutoEncoder is a neural network that learns to reconstruct its input data.

When trained primarily on normal transactions, it can identify unusual transactions through high reconstruction errors.

**Purpose:**

* Learn patterns in normal transactions.
* Identify transactions that differ from learned patterns.
* Explore an alternative approach to anomaly detection.

The AutoEncoder is an optional component of this project.

## 7. Dataset Description

The project uses a synthetic financial transaction dataset generated for demonstration and educational purposes.

The dataset contains approximately 20,000 transactions.

The main columns include:

| Column        | Description                               |
| ------------- | ----------------------------------------- |
| TransactionID | Unique identifier for each transaction    |
| V1–V10        | Anonymized numerical transaction features |
| Time          | Transaction time                          |
| Amount        | Transaction amount                        |
| Class         | Transaction label                         |

The `Class` column represents the transaction category:

* `0` — Legitimate transaction.
* `1` — Fraudulent transaction.

The synthetic dataset contains approximately 0.6% fraudulent transactions, making it an imbalanced classification problem.

**Important:** The dataset is synthetic and does not represent actual bank transactions.

## 8. Installation and Setup

### Step 1: Download the Project

Download the project ZIP file and extract it to your computer.

Open a terminal or command prompt and navigate to the project directory:

```bash
cd fraud-detection-project
```

### Step 2: Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## 9. How to Run the Project

Execute the following steps from the root project directory.

### Step 1: Generate the Dataset

Run the following command:

```bash
python src/generate_data.py
```

This script generates the synthetic transaction dataset and saves it as:

```text
data/transactions.csv
```

### Step 2: Train the Machine Learning Models

Execute:

```bash
python src/train_isolation_forest.py
```

This script performs the main model-training workflow, including preprocessing, feature scaling, training, and evaluation.

It also saves the trained models and evaluation results.

Expected output files include:

```text
models/isolation_forest.joblib
models/rf_smote.joblib
models/scaler.joblib
outputs/metrics.json
outputs/predictions.csv
```

The exact files generated depend on the implementation of the training script.

### Step 3: Launch the Dashboard

Run the following command:

```bash
streamlit run dashboard/app.py
```

After the application starts, open the local URL displayed in the terminal. It is usually:

```text
http://localhost:8501
```

The dashboard allows users to explore transaction information and investigate potential fraud alerts.

### Step 4: Run the Optional AutoEncoder

To experiment with the AutoEncoder, install TensorFlow if it is not already installed:

```bash
pip install tensorflow
```

Then execute:

```bash
python src/train_autoencoder.py
```

This step is optional and is separate from the main training pipeline.

## 10. Dashboard Features

The Streamlit dashboard provides an interactive interface for exploring transaction data and model predictions.

Its features include:

* Viewing transaction statistics.
* Exploring legitimate and potentially fraudulent transactions.
* Switching between Isolation Forest and Random Forest predictions.
* Filtering transactions by transaction amount.
* Displaying only transactions flagged by a selected model.
* Visualizing anomaly scores and transaction patterns.
* Inspecting flagged transaction records.
* Downloading flagged transactions as CSV files.

The dashboard uses the available dataset and generated model outputs. It is not a live banking transaction monitoring system.

## 11. Model Evaluation

The project uses several metrics to evaluate machine learning performance.

### Precision

Precision measures the proportion of transactions flagged as fraudulent that are actually fraudulent.

Higher precision generally means fewer false fraud alerts.

### Recall

Recall measures the proportion of actual fraudulent transactions that the model successfully identifies.

Higher recall means fewer fraudulent transactions are missed.

### F1-Score

The F1-score is the harmonic mean of precision and recall. It helps evaluate the balance between these two metrics.

### ROC-AUC

ROC-AUC measures how effectively a model distinguishes between legitimate and fraudulent transactions across different classification thresholds.

### Average Precision

Average precision summarizes precision-recall performance and is particularly useful when the dataset is highly imbalanced.

### Confusion Matrix

A confusion matrix summarizes correct and incorrect predictions for each class.

It helps identify:

* True positives: Fraudulent transactions correctly detected.
* True negatives: Legitimate transactions correctly identified.
* False positives: Legitimate transactions incorrectly flagged as fraud.
* False negatives: Fraudulent transactions incorrectly classified as legitimate.

Because fraudulent transactions are rare, accuracy alone is not sufficient to evaluate this project. Precision, recall, F1-score, and the confusion matrix should also be considered.

## 12. Exploratory Data Analysis

The project includes a Jupyter Notebook for exploratory data analysis.

Notebook location:

```text
notebooks/EDA.ipynb
```

The notebook can be opened using Jupyter Notebook or the Jupyter extension in Visual Studio Code.

To install and launch Jupyter Notebook:

```bash
pip install jupyter
jupyter notebook
```

Exploratory data analysis helps users understand transaction distributions, identify patterns, and examine relationships between transaction features.

## 13. Applications

Potential applications of fraud detection techniques include:

* Credit and debit card fraud monitoring.
* Digital payment fraud analysis.
* Online banking transaction monitoring.
* Suspicious transaction identification.
* Financial risk analysis.
* Anomaly detection in financial datasets.

These are potential applications of the techniques demonstrated by the project, not verified capabilities of a deployed production system.

## 14. Limitations

* The project uses synthetic rather than real-world financial transaction data.
* Synthetic fraud patterns may not represent actual fraudulent behaviour.
* Model performance depends on the quality and characteristics of the dataset.
* Fraud detection models may generate false positives and false negatives.
* The dashboard does not process live banking transactions.
* Production use would require extensive validation, security controls, privacy protections, and continuous monitoring.

## 15. Future Enhancements

The project could be improved by:

* Evaluating models on suitable real-world anonymized datasets.
* Improving feature engineering and data preprocessing.
* Comparing additional classification and anomaly-detection algorithms.
* Tuning model parameters and decision thresholds.
* Adding explainable AI techniques to help investigate alerts.
* Developing a secure API for transaction scoring.
* Integrating real-time transaction streams.
* Implementing model monitoring and periodic retraining.

## 16. Conclusion

The Fraud Detection in Financial Transactions project demonstrates how machine learning can be used to identify potentially fraudulent transactions.

By combining Isolation Forest, Random Forest with SMOTE, and an optional AutoEncoder, the project explores both supervised classification and anomaly-detection approaches.

The interactive Streamlit dashboard makes it easier to explore transaction data, review suspicious records, and understand model predictions.

Overall, this project provides a practical introduction to machine learning, imbalanced classification, anomaly detection, and financial data analytics.

## 17. License

No license is currently specified for this project. A suitable license should be added before distributing or reusing the project under specific licensing terms.

