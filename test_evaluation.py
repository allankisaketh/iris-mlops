from src.evaluate_model import evaluate_model

def test_model_accuracy():
    acc = evaluate_model("data/processed/iris_processed.csv", "model.joblib")
    assert acc > 0.7, "Model accuracy is too low"