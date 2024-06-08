import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load preprocessed data
data = np.load('../data/data.npy')

# Define the models to be trained
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "GradientBoosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "BernoulliNB": BernoulliNB()
}

# Define the labels and their corresponding files
label_files = {
    "Attention_a1": "../data/labels_Attention_a1.npy",
    "Attention_a2": "../data/labels_Attention_a2.npy",
    "Attention_a3": "../data/labels_Attention_a3.npy",
    "Certainty_a1": "../data/labels_Certainty_a1.npy",
    "Certainty_a2": "../data/labels_Certainty_a2.npy",
    "Certainty_a3": "../data/labels_Certainty_a3.npy",
    "Effort_a1": "../data/labels_Effort_a1.npy",
    "Effort_a2": "../data/labels_Effort_a2.npy",
    "Effort_a3": "../data/labels_Effort_a3.npy",
    "Pleasant_a1": "../data/labels_Pleasant_a1.npy",
    "Pleasant_a2": "../data/labels_Pleasant_a2.npy",
    "Pleasant_a3": "../data/labels_Pleasant_a3.npy",
    "Responsibility_a1": "../data/labels_Responsibility_a1.npy",
    "Responsibility_a2": "../data/labels_Responsibility_a2.npy",
    "Responsibility_a3": "../data/labels_Responsibility_a3.npy",
    "Control_a1": "../data/labels_Control_a1.npy",
    "Control_a2": "../data/labels_Control_a2.npy",
    "Control_a3": "../data/labels_Control_a3.npy",
    "Circumstance_a1": "../data/labels_Circumstance_a1.npy",
    "Circumstance_a2": "../data/labels_Circumstance_a2.npy",
    "Circumstance_a3": "../data/labels_Circumstance_a3.npy"
}

# Function to train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    return accuracy, report

# Prepare directories for saving models and results
os.makedirs("../models", exist_ok=True)
os.makedirs("../results/evaluation_reports", exist_ok=True)
os.makedirs("../results/figures", exist_ok=True)

# Train and save the best model for each label, and generate reports
summary = []

for label, file in label_files.items():
    labels = np.load(file)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    best_report = None

    for model_name, model in models.items():
        accuracy, report = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
        summary.append({
            "Label": label,
            "Model": model_name,
            "Accuracy": accuracy
        })
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
            best_report = report

    # Save the best model
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f"../models/{label}.pkl")

    # Save the classification report
    report_df = pd.DataFrame(best_report).transpose()
    report_df.to_csv(f"../results/evaluation_reports/{label}_report.csv")

    # Generate and save the accuracy plot
    plt.figure(figsize=(10, 6))
    plt.barh([model for model in models.keys()], [entry['Accuracy'] for entry in summary if entry['Label'] == label], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title(f'Model Performance for {label}')
    plt.savefig(f"../results/figures/{label}_performance.png")
    plt.close()

# Save the summary of all models
summary_df = pd.DataFrame(summary)
summary_df.to_csv("../results/evaluation_reports/summary.csv", index=False)
