import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import f1_score


def mse(real, predicted):
    return tf.keras.backend.mean(tf.keras.backend.square(real-predicted), axis=1)


def calculate_metrics(real, predicted):
    accuracy = accuracy_score(real, predicted)
    precision = precision_score(real, predicted)
    recall = recall_score(real, predicted)
    f1score = f1_score(real, predicted)
    
    return accuracy, precision, recall, f1score


def print_metrics(real, predicted, LOAD_MODEL=False, FILENAME=None, LOAD_MODEL_FILENAME=None):
    accuracy_best, precision_best, recall_best, f1_score_best = calculate_metrics(real, predicted)
    if LOAD_MODEL==True:
        print(f"saved_models/{LOAD_MODEL_FILENAME}\n----------------------------------")
    else:
        print(f"{FILENAME}\n----------------------------------")
    print(f"Accuracy: {accuracy_best}")
    print(f"Precision: {precision_best}")
    print(f"Recall: {recall_best}")
    print(f"F1-score: {f1_score_best}")
    
    