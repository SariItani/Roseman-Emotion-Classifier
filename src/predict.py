import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the tokenizer
with open('../data/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the labels for appraisal models
appraisal_labels = [
    "Attention_a1", "Attention_a2", "Attention_a3",
    "Certainty_a1", "Certainty_a2", "Certainty_a3",
    "Effort_a1", "Effort_a2", "Effort_a3",
    "Pleasant_a1", "Pleasant_a2", "Pleasant_a3",
    "Responsibility_a1", "Responsibility_a2", "Responsibility_a3",
    "Control_a1", "Control_a2", "Control_a3",
    "Circumstance_a1", "Circumstance_a2", "Circumstance_a3"
]

# Load the best models for each appraisal label
best_appraisal_models = {label: joblib.load(f'../models/{label}.pkl') for label in appraisal_labels}

# Define the emotion labels
emotion_labels = ['Fear', 'Shame', 'Guilt', 'Disgust', 'Sadness', 'Anger', 'Joy']

# Load the best models for each emotion
best_emotion_models = {emotion: joblib.load(f'../models/{emotion}_best_model.pkl') for emotion in emotion_labels}

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    return padded_sequences

def predict_appraisal(sentence):
    processed_text = preprocess_text(sentence)
    predictions = {}

    for label in appraisal_labels:
        model = best_appraisal_models[label]
        predictions[label] = model.predict(processed_text)[0]
    
    print(f"Appraisal predictions: {predictions}")

    majority_vote = [
        int(np.round(np.mean([predictions["Attention_a1"], predictions["Attention_a2"], predictions["Attention_a3"]]))),
        int(np.round(np.mean([predictions["Certainty_a1"], predictions["Certainty_a2"], predictions["Certainty_a3"]]))),
        int(np.round(np.mean([predictions["Effort_a1"], predictions["Effort_a2"], predictions["Effort_a3"]]))),
        int(np.round(np.mean([predictions["Pleasant_a1"], predictions["Pleasant_a2"], predictions["Pleasant_a3"]]))),
        int(np.round(np.mean([predictions["Responsibility_a1"], predictions["Responsibility_a2"], predictions["Responsibility_a3"]]))),
        int(np.round(np.mean([predictions["Control_a1"], predictions["Control_a2"], predictions["Control_a3"]]))),
        int(np.round(np.mean([predictions["Circumstance_a1"], predictions["Circumstance_a2"], predictions["Circumstance_a3"]])))
    ]

    print(f"Majority vote for appraisal: {majority_vote}")

    return majority_vote, predictions

def predict_emotion(appraisal_array):
    emotion_predictions = {}
    for emotion in emotion_labels:
        model = best_emotion_models[emotion]
        prediction = model.predict([appraisal_array])[0]
        emotion_predictions[emotion] = prediction
        print(f"Prediction for emotion '{emotion}': {prediction}")
    return emotion_predictions

def getAppraisal(text):
    majority_vote, predictions = predict_appraisal(text)
    emotion_predictions = predict_emotion(majority_vote)
    
    # Translate emotion predictions back to emotion names
    predicted_emotions = [emotion for emotion, pred in emotion_predictions.items() if pred == 1]
    
    return {
        'appraisal': majority_vote,
        'emotions': predicted_emotions
    }

if __name__ == "__main__":
    test_sentence = input("Enter the sentence to be predicted: ")
    result = getAppraisal(test_sentence)
    print(f"\nSentence: {test_sentence}")
    print(f"\nPredicted Appraisal Majority Vote: {result['appraisal']}")
    print(f"\nPredicted Emotions: {result['emotions']}")
