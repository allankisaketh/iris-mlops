import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop("species", axis=1)
    y = df["species"]
    
    clf = joblib.load(model_path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    evaluate_model("data/processed/iris_processed.csv", "model.joblib")