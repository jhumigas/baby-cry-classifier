from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data.
    Returns dictionary with accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    
    return {
        "accuracy": acc,
        "classification_report": report
    }
