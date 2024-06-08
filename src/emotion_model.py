import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('../data/enISEAR_appraisal.tsv', sep='\t')
X = data[['Attention_majority', 'Certainty_majority', 'Effort_majority', 'Pleasant_majority', 'Responsibility_majority', 'Control_majority', 'Circumstance_majority']]
y = data['Prior_Emotion']

# Encode the labels
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y_encoded = y.map(label_mapping)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.35, random_state=32)

# Classifiers to test
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'MLP': MLPClassifier(max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Bernoulli NB': BernoulliNB()
}

# One-Hot Encode the Emotion Labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
emotion_labels = encoder.categories_[0]

# Compute Point-Biserial Correlation for Each Appraisal Dimension with Each Emotion
correlations = {}
for i, emotion in enumerate(emotion_labels):
    correlations[emotion] = []
    for col in X.columns:
        corr, _ = pointbiserialr(X[col], y_onehot[:, i])
        correlations[emotion].append(corr)

# Convert the correlation results to a DataFrame for better visualization
corr_df = pd.DataFrame(correlations, index=X.columns)

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Appraisal Dimensions and Emotions')
plt.savefig('../results/figures/appraisal_emotion_correlation.png')

# Train and evaluate each classifier for each emotion
best_results = {}
for emotion in label_mapping.keys():
    y_train_emotion = (y_train == label_mapping[emotion]).astype(int)
    y_test_emotion = (y_test == label_mapping[emotion]).astype(int)
    
    best_accuracy = 0
    best_model = None
    best_name = None
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train_emotion)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test_emotion, y_pred)
        report = classification_report(y_test_emotion, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_name = name
            best_report = report

        print(f"Classifier: {name} for Emotion: {emotion}")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
    
    # Save the best model
    joblib.dump(best_model, f'../models/{emotion}_best_model.pkl')
    best_results[emotion] = (best_name, best_accuracy)
    print(f"Best Model for {emotion}: {best_name} with accuracy: {best_accuracy}")

# Plot the performance
emotion_names = [f"{emotion} - {best_results[emotion][0]}" for emotion in best_results]
accuracies = [best_results[emotion][1] for emotion in best_results]

plt.figure(figsize=(14, 10))
sns.barplot(x=accuracies, y=emotion_names, palette='viridis')
plt.xlabel('Accuracy')
plt.title('Classifier Performance Comparison for Each Emotion')
plt.tight_layout()  # Adjusts the plot to ensure everything fits
plt.savefig('../results/figures/emotions_classifier_performance.png')
